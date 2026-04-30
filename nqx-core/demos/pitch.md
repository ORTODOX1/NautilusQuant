# NautilusQuant / NQX-Core — pitch deck

10 slides. Read top-to-bottom. Each section is one slide.

---

## 1. Problem

**KV-cache eats 80% of HBM at long context.**

Llama-3-70B at 128 K context = **320 GB FP16 KV cache**. That's 4 × H100s
just to *hold the cache*, before any compute. Operators are paying for
HBM, not for FLOPs. As context windows trend toward 1 M tokens, the
problem grows linearly.

---

## 2. State of the art

**TurboQuant (Google, ICLR 2026) and friends quantise the KV cache by
randomly rotating it first.** That works mathematically — random
rotation flattens outlier-heavy distributions — but it costs:

- A `dim × dim` rotation matrix per layer (32 KB for dim=128 in FP16).
- A CSPRNG state, ≥ 4 cycles per random word.
- **Non-deterministic output**: two runs on identical inputs produce
  different bytes. Bad for caching, bad for verification, bad for audit.

GPU-bound. Memory-bound. Not reproducible.

---

## 3. Insight

**The golden angle is provably more uniform than random.**

Weyl 1916: `{k · 2π/φ²}` is the most equidistributed deterministic
sequence on the circle. Three-distance theorem gives discrepancy
`D*_N = O(log N / N)` versus `O(1/√N)` for random.

We measured slope **−0.93 vs −0.48** in
`bench/angular_uniformity.md`. The math is on our side.

---

## 4. Solution — NautilusQuant

```
KV vector  →  Givens × φ (3 layers)  →  polar  →  3-bit Lloyd-Max  →  1-bit QJL  →  packed bytes
```

The rotation `T` is **fixed** at design time, identical for every model,
every layer, every device. State on chip: a 1.9 KB ROM (dim=128). No
training. No PRNG. No per-layer matrix.

---

## 5. Hardware fit — NQX-Core

The pipeline is a **static dataflow** — no runtime branches, no
register reuse, no varying dependencies. That is the textbook workload
for an ASIC, not a GPU.

50 mm² in TSMC 7 nm, 1 GHz, 12 W TDP. Steady-state: 1 vector per cycle
after 7-cycle prefill.

```
LDV → GU.L1 → GU.L2 → GU.L3 → POLAR → QUANT → QJL → PACK → STV
   1c     1c      1c      1c       1c      8c     1c     1c    DMA
```

We aren't porting an algorithm to silicon. The algorithm *is* the
silicon description.

---

## 6. Numbers — head-to-head

From `demos/side_by_side.md`, identical inputs (1024 vec × dim=128):

| Metric | TurboQuant | **NautilusQuant** | Δ |
|---|---:|---:|---:|
| RMSE roundtrip | 0.282 | **0.286** | +1.5% |
| Cycles per vector | 65.1 | **2.02** | **32× lower** |
| Energy nJ/vec | 80.6 | **8.3** | **10× lower** |
| LUT/PRNG state | 32 KB | **1.9 KB** | **17× smaller** |
| Determinism | 0/30 match | **30/30 match** | absolute |
| Compression | 4× | **4×** | equal |

Quality parity. Cost asymmetry.

---

## 7. Architecture — NQX-Core block diagram

```
+--------------------------------------------------+
|  HBM2e PHY  (1 stack, 256 GB/s)         IO ring  |
| ------------------------------------------------ |
|        |  GU.L1  |  GU.L2  |  GU.L3  |          |
|  ROM   | 64 lane | 63 lane |≤32 lane | SRAM_in  |
|  4 KB  |---------+---------+---------|  24 KB   |
|        |  PU     |  QU     |  QJL    | SRAM_out |
|        | CORDIC  | Lloyd-M | fused   |  24 KB   |
|        |---------+---------+---------|          |
|        | PACK    | VRF (16 × 512 B)  | ATTN_DOT |
|        | 3+1     |       8 KB        | 64 lane  |
| ------------------------------------------------ |
|  Frontend   | CSR | DMA engine                  |
|  PCIe Gen5 x8                                    |
+--------------------------------------------------+
```

Three power islands, one HBM stack, no exotic IP. See
`asic/floorplan.md`.

---

## 8. Roadmap

| Phase | Milestone | Status |
|---|---|---|
| E1 — Software emulator | 1.1 K LoC NumPy, 173 tests, bit-exact vs upstream | **DONE** |
| E2 — RTL / Verilator | `rtl/*.sv` skeleton + bit-exact testbench | **scaffolded** |
| E3 — FPGA Alveo U280 | ≥ 100 MHz, ≥ 10 K vec/s | 3 months |
| E4 — Integrations | vLLM plugin + llama.cpp adapter | 6 months |
| E5 — ASIC tape-out | TSMC 12 nm/7 nm, full chip | 12 months |
| E6 — Board bring-up | PCIe Gen5 card, Linux driver | 18 months |

---

## 9. Money

| Line item | TSMC 12 nm | TSMC 7 nm |
|---|---:|---:|
| Mask set | $1.5 M | $5.0 M |
| Tape-out + first wafer | $0.8 M | $1.6 M |
| Per-chip cost at 30 K-chip volume | $35 | $80 |
| Per-chip TCO amortised, 30 K hr | $0.05 / hr | $0.10 / hr |
| Equivalent H100 spot cost | $2.50 / hr | $2.50 / hr |
| **Break-even per chip** | **560 hours** | **1 250 hours** |

Per `demos/scaling_demo.md`: each Llama-3-70B serving slot saves
≈ $9.80 / hour vs an H100 stack. A 4-chip NQX server pays back its
silicon in **~ 1.5 months of continuous load**.

---

## 10. Ask

We're raising to:

1. **$2 M — FPGA dev (3 months)**: Alveo U280 cards, 2 RTL engineers,
   Vivado licences, bring-up + integration into vLLM.
2. **$3 M — tape-out reserve**: TSMC 12 nm shuttle slot Q3 2026, mask
   set + first wafer, packaging + interposer.
3. **$1.5 M — team for 12 months**: 4 senior eng (RTL, software, ML,
   ops), runway through tape-out.

Total: **$6.5 M**. Outcome: a working FPGA demo on a real LLM in 3
months, a single-chip NQX-Core PCIe board in 18 months, design files and
integrations open-sourced under Apache 2.0 from day one.

The hardware is small, the math is settled and the comparison table is
already built. Now we ship it.
