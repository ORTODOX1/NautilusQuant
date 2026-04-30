# φ-Givens vs Random rotation — head-to-head

Three metrics on identical synthetic KV-like inputs (Gaussian + 1/64 outliers ~6σ). Random rotation is a fresh QR-orthonormal `dim × dim` matrix per run; cycle counts include PRNG latency at 8 cycles/vector. φ-Givens uses the three-layer structure with `cycles_dma_per_byte = 0` (we count compute only, since DMA is identical between the two approaches).

## Per (dim, bits) results

| dim | bits | φ RMSE | random RMSE (μ ± σ) | φ wall ms | random wall ms | φ cycles | random cycles |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3 | 0.2009 | 0.1832 ± 0.0026 | 1.78 | 4.30 | 523 | 4680 |
| 64 | 4 | 0.0940 | 0.0854 ± 0.0011 | 1.72 | 3.40 | 523 | 4680 |
| 128 | 3 | 0.1887 | 0.1781 ± 0.0020 | 4.54 | 17.65 | 523 | 4744 |
| 128 | 4 | 0.0879 | 0.0831 ± 0.0009 | 4.79 | 15.74 | 523 | 4744 |

## Headline numbers

- Average RMSE: φ **0.1429** vs random 0.1325
- Average wall ms: φ **3.21** vs random 10.27
- Average cycles: φ **523** vs random 4712

## Verdict

**φ-Givens wins.** Quality (RMSE) is within +7.9% of random rotation, while cycles are 88.9% lower. The static three-layer Givens topology amortises across batch and spends zero PRNG cycles.

## Reproduction

```bash
python bench/phi_vs_random.py --out bench/phi_vs_random.md
```
