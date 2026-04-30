# Side-by-side — TurboQuant vs NautilusQuant

Both pipelines processed identical inputs: `1024 vectors × dim=128` synthetic Gaussians + 1/64 outliers ~6σ. `30` repeated encodes for the determinism column.

| Metric | TurboQuant | NautilusQuant | Δ |
|---|---:|---:|---|
| RMSE roundtrip | 0.2818 | 0.2860 | +0.0042 (+1.5%) |
| Cycles total | 66,698 | 2,066 | +96.9% |
| Cycles per vector | 65.1 | 2.02 | 32× lower for NQX |
| Energy nJ/vec | 80.56 | 8.33 | 9.7× lower for NQX |
| LUT / PRNG state | 32,768 B | 1,910 B | 17.2× smaller for NQX |
| Determinism (`30 runs`) | 30/30 unique → 0% match | 1/30 unique → **100% match** | absolute |
| Compression ratio | 4.00× | 4.00× | equal |

## Headline

- **32× fewer cycles** at **10× lower energy** per vector,
- with a **17× smaller** rotation state (ROM only, no PRNG, no per-layer matrix),
- and **bit-identical** output across runs (TurboQuant is non-deterministic by construction).

## Reproduction

```bash
python demos/side_by_side.py
```
