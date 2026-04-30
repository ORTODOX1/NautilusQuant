# NQX-Core ASIC Floor-plan (TSMC N7, 50 mm²)

Целевой пакет: 50 mm² die в TSMC 7nm (≈ 1.0 GHz), один power island на compute,
второй на SRAM, третий на HBM2e PHY и off-chip I/O.

## 1. Top-level layout (схематично, 7.07 × 7.07 mm)

```
+--------------------------------------------------+ 7.07 mm
|  HBM2e PHY  (1 stack, 256 GB/s)         IO ring  |
|  ------------------------------------------------ |
|        |  GU.L1   |  GU.L2   |  GU.L3   |        |
|  ROM   |  64 lane |  63 lane | ≤32 lane | SRAM_in|
|  LUT   |          |          |          |  24 KB |
|  4 KB  |  PU      |  QU      |  QJL     | SRAM_out|
|        |  CORDIC  |  Lloyd-M |  fused   |  24 KB |
|        |----------+----------+----------|        |
|        |   PACK   |   VRF (16 × 512 B)  | ATTN_DOT|
|        |   3+1    |        8 KB         |  64 lane|
|  ------------------------------------------------ |
|  Frontend (Fetch/Decode/Issue) | CSR | DMA engine |
|  ------------------------------------------------ |
|  PCIe Gen5 x8 SerDes       Power mgmt + clk gen   |
+--------------------------------------------------+ 7.07 mm
```

## 2. Block area & gate count budget (TSMC N7)

| Block          | Gate count | Area est. | Comment                                        |
|----------------|-----------:|----------:|------------------------------------------------|
| GU.L1 (64 lane × 6 ops, FP32 mul/add) | 1.6 M  | 3.2 mm² | Each lane = 4 mul + 2 add + LUT broadcast bus |
| GU.L2 (63 lane)        | 1.55 M | 3.1 mm² | Same micro-arch as L1                          |
| GU.L3 (≤32 lane)       | 0.8  M | 1.6 mm² | Stride dim/4 butterfly, fewer lanes            |
| Polar Unit (CORDIC ×64)| 1.0  M | 2.0 mm² | sqrt 4-stage + atan2 8-stage                   |
| Quant Unit             | 0.6  M | 1.2 mm² | min/max reduction tree depth 7 + Lloyd-Max     |
| QJL Unit               | 0.2  M | 0.4 mm² | sign + scaled add                              |
| Pack Unit (3+1)        | 0.1  M | 0.2 mm² | bit-serial pack, 4-cycle latency               |
| Attention Unit         | 0.7  M | 1.4 mm² | Polar dot fused, MUL+cos LUT lookup            |
| MX block-quant         | 0.4  M | 0.8 mm² | 8-bit shared exponent reduce                   |
| Sub-bit unit           | 0.15 M | 0.3 mm² | radius/angle split-quant                       |
| ROM (golden LUT)       | 4 KB   | 0.04 mm²| Hard-macro                                     |
| SRAM_in (24 KB)        | —      | 1.0 mm² | Single-port, 1 GHz                             |
| SRAM_out (24 KB)       | —      | 1.0 mm² | Single-port, 1 GHz                             |
| VRF (16 × 512 B)       | —      | 0.8 mm² | 4-port (3 read, 1 write)                       |
| Frontend + CSR + DMA   | 1.2  M | 2.4 mm² | F/D/I, dispatcher, DMA engine                  |
| HBM2e PHY              | hard   | 12.0 mm²| 1 stack, 8 ch, ≈16 GB/s/ch                     |
| PCIe Gen5 x8 SerDes    | hard   | 6.0 mm² | board-side host link                           |
| PLL/clock/power-mgmt   | —      | 1.5 mm² | 1 GHz core + 600 MHz HBM clk                   |
| Routing & spare        | —      | ≈12 mm² | tracks, pad ring, decap                        |
| **Total**              |        | **≈ 49 mm²** | fits 50 mm² floor-plan with margin       |

## 3. Power islands

| Island | Voltage | Blocks                                    | Power gating |
|--------|---------|-------------------------------------------|--------------|
| `vdd_compute`  | 0.75 V | GU/PU/QU/QJL/PACK/ATTN/MX/Subbit, VRF | yes          |
| `vdd_sram`     | 0.80 V | SRAM_in, SRAM_out, ROM                | retention    |
| `vdd_phy`      | 1.05 V | HBM2e PHY, PCIe SerDes                | always-on    |
| `vdd_io`       | 1.80 V | pad ring, JTAG, board interface       | always-on    |

DVFS levels: 600 MHz / 0.65 V (low-power), 1.0 GHz / 0.75 V (nominal),
1.2 GHz / 0.85 V (turbo, < 5% duty).

## 4. Power budget @ 1 GHz

| Block         | Dynamic | Leakage | Total |
|---------------|--------:|--------:|------:|
| Compute (GU+PU+QU+QJL+PACK+ATTN+MX+Subbit) | 4.5 W | 0.4 W | 4.9 W |
| SRAM (in/out + VRF)                        | 0.7 W | 0.1 W | 0.8 W |
| HBM2e PHY                                   | 4.0 W | 0.2 W | 4.2 W |
| PCIe SerDes                                 | 1.5 W | 0.1 W | 1.6 W |
| Frontend + DMA                              | 0.4 W | 0.1 W | 0.5 W |
| **Total**                                   |       |       | **≈ 12 W** |

Energy per encoded vector at steady-state ≈ 5.1 nJ — see `docs/architecture.md §5`.

## 5. Routing constraints

- ROM_LUT broadcast bus to GU lanes: ≤ 1 mm signal length, registered every
  2 mm to keep flops at ≤ 700 ps slack at 1 GHz.
- VRF read ports R0/R1/R2 routed to GU/PU/QU side; write port from
  PACK/QJL/QU output.
- HBM2e PHY parked at the top edge with the 1024-bit DDR bus running through a
  dedicated routing channel to SRAM_in/SRAM_out (≤ 1.5 mm).
- ATTN_DOT block placed adjacent to VRF + ROM-cos LUT to share the polar bus.

## 6. Open issues (E5 follow-up)

- **CORDIC vectoring depth** — current 8-stage may need to grow to 12 to keep
  atan2 error < 1 ULP at 1 GHz.
- **Lloyd-Max range register** — sharing min/max accum across batches vs
  per-batch reset; affects QU.r area by ~5%.
- **HBM/PCIe arbiter** — single DMA engine vs split read/write engines;
  decision driven by timing closure (see `asic/timing.md`).
