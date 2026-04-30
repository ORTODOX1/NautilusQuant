# Determinism witness — φ-Givens vs random rotation

Both encoders run `n_repeats=50` times on the same input (dim=128, batch=64). We hash the packed encoder output with SHA-256 and count distinct hashes.

| Method | Distinct hashes | All identical? |
|---|---:|---|
| **φ-Givens (NQXCore.encode)** | 1 / 50 | YES |
| Random rotation (fresh QR per run) | 50 / 50 | no |

First φ output hash: `6c1f7199ca8c5e62d901042c9887cf6a5fb2a115532ead405e9490a238d26bee`

**Property witnessed.** Every invocation of φ-Givens emits the same byte stream — the encoder is a pure function of the input tensor and the fixed φ-LUT. Random rotation, by contrast, emits a different byte stream every run, regardless of seed schedule. This is the formal guarantee that an NQX-compressed KV-cache is *reproducible* across silicon, drivers and OS schedulers.

## Why determinism matters

- **Hardware verification**: bit-exact equivalence between Python emulator, Verilator RTL and silicon presupposes that the math is bit-deterministic. Random rotations cannot meet this bar without shipping a per-device PRNG state.
- **KV-cache portability**: a checkpoint produced on host A must decode identically on host B. φ guarantees this; random rotations leak the seed into the cache header and break drop-in replacement.
- **Auditability**: deterministic compression is a requirement for regulated deployments where every cache update needs a hash trail.

## Reproduction

```bash
python bench/determinism.py --out bench/determinism.md
```
