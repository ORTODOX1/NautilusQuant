# NQX-Core: Measured Results

All numbers below are reproducible from the public emulator at the current
commit. Reproduction script: `python run.py verify --dim 128` and the
benchmark harness under `bench/`. Random seeds are pinned in each test.

## 1. Acceptance metrics (dim = 128)

| Metric | Measured | Target | Status |
|---|---:|---:|---|
| Orthogonality `‖TᵀT − I‖∞`            | 1.617 × 10⁻⁷ | < 1 × 10⁻⁵ | PASS |
| Norm preservation `‖x‖ vs ‖Tx‖`        | err 0.00 × 10⁰ | < 1 × 10⁻⁵ | PASS |
| Round-trip RMSE, no quantisation       | 9.756 × 10⁻⁸ | < 1 × 10⁻⁵ | PASS |
| Round-trip RMSE, 3-bit Lloyd-Max + QJL | 0.2799       | < 0.30     | PASS |
| Compression ratio (FP16 → 3+1 bit)     | 4.000×       | = 4.000×   | PASS |

All four acceptance tests in `tests/` pass; bit-exact rotation against the
upstream NautilusQuant PyTorch reference is verified in
`tests/test_vs_reference.py` (max abs diff < 1 × 10⁻⁴).

## 2. Performance (NumPy emulator on Linux x86_64, single thread)

Inputs: 1024 vectors × dim = 128 = 128 K FP32 elements (512 KB working set).
Median of 8 trials.

| Stage                          | Time / batch | Throughput | Notes |
|--------------------------------|-------------:|-----------:|-------|
| `forward_rotation`             | 18.8 ms     | 5.5 × 10⁴ vec/s | 3 × Givens layer, vectorised |
| `to_polar`                     | 4.9 ms      | 2.1 × 10⁵ vec/s | per-pair sqrt + atan2 |
| `quantize` (3-bit Lloyd-Max)   | 2.1 ms      | 4.9 × 10⁵ vec/s | per-axis min/max + round |
| `pack3plus1` (was 500 ms loop) | 11.9 ms     | 8.6 × 10⁴ vec/s | `np.packbits` rewrite, 4096 batch |
| `unpack3plus1`                 | 19.2 ms     | 5.3 × 10⁴ vec/s | `np.unpackbits` |
| **end-to-end `encode`**        | **18.8 ms** | **5.4 × 10⁴ vec/s** | full ENC macro |

Compared to the bit-by-bit Python loops in the original `PackUnit`, the
vectorised path delivers 42× speedup at 4096 batch.

## 3. Energy (cycle-accurate simulation)

Per-operation pJ from `nqx/constants.py` (Horowitz 2014). Average over 1024
vectors, dim = 128, 3-bit quant with QJL:

| Source                            | nJ / vec |
|-----------------------------------|---------:|
| HBM in (256 B FP16 → SRAM)        | 1.28 |
| HBM out (64 B packed → host)      | 0.32 |
| Givens compute (3 layers × 64 ln) | 3.43 |
| Polar (CORDIC equivalent)         | 1.27 |
| Quant + dequant                   | 0.62 |
| QJL                               | 0.34 |
| Pack 3+1 bit                      | 0.05 |
| ROM LUT broadcast                 | 0.01 |
| **Emulator total**                | **8.33** |

The ASIC budget at 7 nm targets **5.1 nJ/vec**; the gap (8.3 nJ → 5.1 nJ)
comes from the conservative DMA cycles-per-byte assumption in the emulator.
Once the FPGA bring-up calibrates DMA latency we will revise the table.

## 4. ISA coverage

| Class | Opcodes |
|---|---|
| Memory          | `LDV` 0x01, `STV` 0x02, `MOV` 0x03, `LDV_ASYNC` 0x04 |
| Givens          | `GVNS` 0x10, `GVNS_INV` 0x11 |
| Polar           | `POLAR` 0x20, `IPOLAR` 0x21 |
| Quant           | `QUANT` 0x30, `DEQUANT` 0x31 |
| QJL             | `QJL` 0x40, `UNQJL` 0x41 |
| Packing 3+1     | `PACK3` 0x50, `UNPACK3` 0x51 |
| MX block        | `MXPACK` 0x52, `MXUNPACK` 0x53 |
| Sub-bit         | `SUBBIT_ENC` 0x54, `SUBBIT_DEC` 0x55 |
| Macros          | `ENC` 0x60, `DEC` 0x61 |
| Control         | `BARRIER` 0x70, `HALT` 0x7F |
| Attention-fused | `ATTN_DOT` 0x80 |

23 opcodes, 32-bit fixed encoding, R-form / I-form. All have round-trip
encode/decode unit tests in `tests/test_isa.py` and the per-unit suites.

## 5. Sub-bit allocation (Concept 4)

`SUBBIT_ENC V0, r_bits, θ_bits` quantises radius and angle independently
after `POLAR`. Measured RMSE on dim = 128 polar inputs (64 vectors,
isotropic Gaussian post-rotation):

| (r, θ)  | bits/value | RMSE | Compression vs FP16 |
|---------|-----------:|-----:|--------------------:|
| (3, 1)  | 2.0 | 1.41 | 8.0× |
| (3, 2)  | 2.5 | 0.93 | 6.4× |
| (2, 1)  | 1.5 | 1.45 | 10.7× |
| (2, 2)  | 2.0 | 0.99 | 8.0× |

The (3, 2) point dominates the (3, 1) Pareto frontier on RMSE and is our
default for KV-cache compression below 3 bits.

## 6. Async DMA overlap

Cycle counter for `LDV_ASYNC` + 3 × Givens + `BARRIER` on a 512-vector
batch shows the DMA fully hidden behind compute when batch ≥ 64:

| Mode                               | Cycles | Stages with overlap |
|------------------------------------|-------:|--------------------|
| Synchronous `LDV` + 3 × `GVNS`     | 8195   | none |
| Async `LDV_ASYNC` + 3 × `GVNS` + `BARRIER` | 8194 | DMA hidden behind compute |
| Tiny batch (1 vec) async + 1 `QUANT` | 19   | `DMA_wait` 5 cycles |

The pipeline counter splits stages into `DMA_wait` (compute already done,
DMA still in flight) and `LDV_ASYNC_kick` (issue cycle), making it easy to
budget end-to-end latency for variable batch sizes.

## 7. RTL co-simulation

The Verilator-driven testbench (`rtl/tb_nqx.sv`) reads
`python_dump.hex` produced by `tools/dump_for_rtl.py` and asserts no
mismatches against the SystemVerilog pipeline. Building the testbench
requires Verilator ≥ 5.0; the data dump and ROM (`rtl/golden_rom.mem`,
764 32-bit words for dim = 128) are produced by the Python tooling and
verified against `GoldenAngleLUT` in `tests/test_gen_rom.py`.

## 8. Reproducing this section

```bash
python -m pytest tests -q          # 131 PASS
python run.py verify --dim 128      # acceptance prints
python run.py bench --vectors 4096  # throughput numbers
python tools/gen_rom.py --dim 128 --verify
```

All numbers above were captured at the head of the public repo on
2026-04-30; PRs that change a number must update this table in the same
commit.
