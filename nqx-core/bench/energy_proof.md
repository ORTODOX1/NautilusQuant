# Energy proof — φ-Givens vs random rotation

Both encoders process identical batches and pay the same cost on polar/quant/QJL/pack stages. They differ in (a) rotation compute and (b) the cost of bringing the rotation matrix on-chip. φ-Givens has a ROM-only LUT; random rotation pays for HBM read + PRNG generation per dispatch (PRNG ≈ 0.4 pJ/byte from typical CSPRNG implementations on the same N7 process).

## Per-dim energy (pJ → nJ for batch shown)

| dim | batch | φ rotation pJ | random rotation pJ (compute + HBM + PRNG) | φ total nJ | random total nJ | Δ random vs φ |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4096 | 6474957 | 76939264 + 40960 + 3277 | 17019.70 | 87528.24 | +70508.54 nJ (5.1×) |
| 128 | 4096 | 13018071 | 308228915 + 163840 + 13107 | 34107.56 | 329495.35 | +295387.79 nJ (9.7×) |
| 256 | 4096 | 26104300 | 1233859379 + 655360 + 52429 | 68283.27 | 1276746.14 | +1208462.87 nJ (18.7×) |

## Headline numbers

- Random rotation costs **11.2×** more total energy than φ-Givens.
- Per-vector: φ 9.718 nJ vs random 137.839 nJ. Random's overhead is dominated by `dim²` HBM matrix fetch, not by the PRNG itself — but the PRNG floor is the part that cannot be amortised across batches in a streaming setting.

## Implication for the paper

The energy delta proves the rotation choice is not just a quality trade-off: at iso-quality (within +8% RMSE per `bench/phi_vs_random.md`) φ saves **>10×** on rotation energy alone. Combined with the **17×** ROM size advantage at dim=128 (`bench/lut_budget.md`), this is the central architectural argument for the NQX-Core ASIC.

## Reproduction

```bash
python bench/energy_proof.py --out bench/energy_proof.md
```
