# Angular uniformity — φ vs random vs Hadamard

Each method produces a sequence of N angles in [0, 2π). We measure how uniformly distributed the sequence is via the Kolmogorov–Smirnov statistic D = max|F_N(u) − u| (equivalent to the 1D L∞ star discrepancy). Lower is better.

Theory (Weyl 1916, *Über die Gleichverteilung von Zahlen mod. Eins*): the sequence {kα mod 1} is equidistributed iff α is irrational. For α = 1/φ² (golden ratio) the sequence has the lowest possible discrepancy class — D*_N = O(log N / N) — by the three-distance theorem. For uniformly random samples, D*_N = Θ(√(log log N / N)) ≈ O(1/√N) by Chung's law of iterated logarithm.

## Measured discrepancy

| N | φ-Givens | random (mean ± σ, 8 seeds) | Hadamard pairs |
|---:|---:|---:|---:|
| 64 | 0.02897 | 0.10077 ± 0.02198 | 0.21875 |
| 256 | 0.00656 | 0.04336 ± 0.01082 | 0.16406 |
| 1024 | 0.00185 | 0.02746 ± 0.00676 | 0.13867 |
| 4096 | 0.00060 | 0.01383 ± 0.00599 | 0.12866 |
| 16384 | 0.00015 | 0.00652 ± 0.00136 | 0.12610 |

## Empirical scaling — fit log D vs log N

- φ-Givens slope: **-0.929** (theoretical -1.0 for O(1/N))
- random slope:  **-0.478** (theoretical -0.5 for O(1/√N))

Conclusion: the empirical slope of φ-Givens is closer to −1, confirming the Weyl O(log N / N) bound. Random rotations regress to the −0.5 slope predicted by the law of iterated logarithm. Hadamard's pair-angle distribution is fixed and not a sequence in the equidistribution sense, so its row is informational only.

## Reproduction

```bash
python bench/angular_uniformity.py --out bench/angular_uniformity.md
```

## References

- H. Weyl, *Über die Gleichverteilung von Zahlen mod. Eins*, Math. Ann. 77 (1916).
- L. Kuipers, H. Niederreiter, *Uniform Distribution of Sequences*, Wiley 1974.
- K. F. Roth, *On irregularities of distribution*, Mathematika 1 (1954).
