# LUT size proof — φ-Givens vs Random rotation matrix

Random rotations need to store the full `dim × dim` orthogonal matrix `T` (FP16 minimum). φ-Givens stores a **fixed** ROM with the rotation pair indices and (cos, sin) per pair, across three layers. Bytes per pair: 10 = 1B pair_i + 1B pair_j + 4B cos (FP32) + 4B sin (FP32). pair indices fit in uint8 up to dim=256.

## Per-dim LUT budget

| dim | φ-LUT total | per layer (L1 / L2 / L3) | random `T` (FP16) | random `T` (FP32) | ratio φ vs random FP16 |
|---:|---:|---|---:|---:|---:|
| 64 | **950 B** | 32p/320 B · 31p/310 B · 32p/320 B | 8.00 KB | 16.00 KB | **9×** |
| 128 | **1.87 KB** | 64p/640 B · 63p/630 B · 64p/640 B | 32.00 KB | 64.00 KB | **17×** |
| 256 | **3.74 KB** | 128p/1.25 KB · 127p/1.24 KB · 128p/1.25 KB | 128.00 KB | 256.00 KB | **34×** |
| 512 | **7.49 KB** | 256p/2.50 KB · 255p/2.49 KB · 256p/2.50 KB | 512.00 KB | 1.00 MB | **68×** |

## Headline numbers

- φ-LUT growth across dim ∈ {64, 128, 256, 512}: linear in `dim` (≈ 15 B/dim). Max in this sweep: **7.49 KB** at dim=512. The 4 KB ROM in `asic/floorplan.md` is sized for dim ≤ 256 (NQX-Core target); dim=512 bumps the ROM macro to 8 KB, still on-die and still independent of model count.
- Random `T` at dim=512: **512.00 KB** (FP16) per layer. A 32-layer model needs 16 MB just for rotation matrices — that lives in HBM and burns one extra full HBM read per attention pass. At dim=128 (the NQX-Core default) the ratio is **17×** φ vs random FP16.

## Implication

Random rotation hardware must either (a) recompute `T` from a PRNG seed every dispatch, or (b) carry a per-layer FP16 matrix in HBM/PCIe-mapped memory. φ-Givens reduces this to a **single 4 KB ROM** that is identical for every model, every layer and every device. This is the architectural reason NQX-Core can sit on a 50 mm² die with no off-chip rotation state.

## Reproduction

```bash
python bench/lut_budget.py --out bench/lut_budget.md
```
