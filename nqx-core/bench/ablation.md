# Rotation × dim × bits — RMSE ablation

Synthetic isotropic Gaussians with 1/32 outliers of magnitude 8σ. Per-axis min/max quantisation after rotation; rotation is identity for `none`. RMSE measured against the original FP32 input after the inverse rotation. Lower is better.

## dim = 64
| rotation \ bits | 2 | 3 | 4 |
|---|---:|---:|---:|
| `phi` | 1.0102 | 0.4254 | 0.1991 |
| `random` | 0.7182 | 0.3027 | 0.1414 |
| `hadamard` | 0.6885 | 0.2908 | 0.1359 |
| `none` | 1.1229 | 0.4659 | 0.2177 |

## dim = 128
| rotation \ bits | 2 | 3 | 4 |
|---|---:|---:|---:|
| `phi` | 0.8407 | 0.3662 | 0.1713 |
| `random` | 0.6841 | 0.2882 | 0.1346 |
| `hadamard` | 0.6776 | 0.2865 | 0.1336 |
| `none` | 0.8851 | 0.3852 | 0.1798 |

## dim = 256
| rotation \ bits | 2 | 3 | 4 |
|---|---:|---:|---:|
| `phi` | 0.7601 | 0.3310 | 0.1547 |
| `random` | 0.6752 | 0.2854 | 0.1330 |
| `hadamard` | 0.6714 | 0.2845 | 0.1327 |
| `none` | 0.7828 | 0.3418 | 0.1595 |

## Reproduction

```bash
python bench/ablation.py --out bench/ablation.md
```
