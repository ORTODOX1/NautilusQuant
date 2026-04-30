# Linear (uniform) vs Lloyd-Max quantisation after rotation

If φ-rotation flattens the distribution to near-uniform, then the optimal Lloyd-Max quantiser collapses onto the simple linear quantiser. **A small δ = (RMSE_linear − RMSE_LM) / RMSE_LM after φ-rotation is direct evidence that Lloyd-Max codebooks are unnecessary in hardware** — we can replace the per-feature Lloyd-Max table with a single 1-cycle linear quantiser.

## Per-feature RMSE on synthetic outlier-laden Gaussians

### rotation = `phi`
| dim \ bits | 2-bit (linear / LM / δ%) | 3-bit (linear / LM / δ%) | 4-bit (linear / LM / δ%) |
|---|---:|---:|---:|
| 64 | 0.7211 / 0.3762 / +91.69% | 0.3059 / 0.2052 / +49.02% | 0.1433 / 0.1191 / +20.25% |
| 128 | 0.6641 / 0.3568 / +86.13% | 0.2799 / 0.1983 / +41.15% | 0.1309 / 0.1128 / +16.01% |
| 256 | 0.6104 / 0.3458 / +76.52% | 0.2671 / 0.1931 / +38.32% | 0.1254 / 0.1095 / +14.46% |

### rotation = `none`
| dim \ bits | 2-bit (linear / LM / δ%) | 3-bit (linear / LM / δ%) | 4-bit (linear / LM / δ%) |
|---|---:|---:|---:|
| 64 | 0.7429 / 0.3682 / +101.80% | 0.3187 / 0.2105 / +51.39% | 0.1468 / 0.1209 / +21.43% |
| 128 | 0.6816 / 0.3553 / +91.85% | 0.2864 / 0.1988 / +44.06% | 0.1321 / 0.1135 / +16.32% |
| 256 | 0.6189 / 0.3468 / +78.47% | 0.2701 / 0.1926 / +40.22% | 0.1247 / 0.1088 / +14.67% |

## Headline numbers

- Average δ after φ-rotation: **+48.17%**
- Average δ without rotation: **+51.14%**

**Hypothesis NOT confirmed at this distribution.** Linear quant is +48.2% worse than Lloyd-Max after φ-rotation; without rotation the gap is +51.1%. The synthetic input is isotropic-Gaussian + outliers — Gaussian is rotationally invariant, so the per-feature marginal stays Gaussian and Lloyd-Max wins by allocating more codes near zero. φ-rotation still helps the outlier dimension via spreading (compare absolute RMSE_LM rows: phi vs none) but does not flatten the distribution to uniform. The QU.q stage in `nqx/cpu.py` is therefore retained as Lloyd-Max for the production pipeline. **Implication for the paper:** the rotation's value is in outlier dispersion + ROM elimination, *not* in making Lloyd-Max obsolete on Gaussian-like KV activations. Re-run on real KV-cache (heavy-tailed) to see if the gap shrinks below 5%.

## Reproduction

```bash
python bench/linear_quant.py --out bench/linear_quant.md
```
