# Benchmarks

## Encode+decode pipeline

`python run.py bench --vectors N --dim D` (CPU backend, FP32 sim).

### Cycles

| dim \ vectors | 256 | 1024 | 4096 |
|:---:|:---:|:---:|:---:|
| 32 | 530 | 2066 | 8210 |
| 64 | 530 | 2066 | 8210 |
| 128 | 530 | 2066 | 8210 |
| 256 | 530 | 2066 | 8210 |

### Throughput (vec/cycle)

| dim \ vectors | 256 | 1024 | 4096 |
|:---:|:---:|:---:|:---:|
| 32 | 0.95 | 0.99 | 1.00 |
| 64 | 0.95 | 0.99 | 1.00 |
| 128 | 0.95 | 0.99 | 1.00 |
| 256 | 0.95 | 0.99 | 1.00 |

~1 vec/cycle steady-state across all dims, pipeline saturates at ≥1024 vectors.

### Energy per vector (nJ)

| dim \ vectors | all batch sizes |
|:---:|:---:|
| 32 | 3.635 |
| 64 | 7.304 |
| 128 | 14.641 |
| 256 | 29.316 |

Energy scales linearly with dim: ≈0.114 nJ/dim/vec.

### RMSE (encode→decode)

| dim \ vectors | 256 | 1024 | 4096 |
|:---:|:---:|:---:|:---:|
| 32 | 2.52 | 2.44 | 2.48 |
| 64 | 1.91 | 1.99 | 1.96 |
| 128 | 1.64 | 1.65 | 1.68 |
| 256 | 1.19 | 1.18 | 1.22 |

RMSE decreases with larger dim (more signal averaging).
