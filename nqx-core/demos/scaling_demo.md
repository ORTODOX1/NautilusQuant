# 70B-class scaling projection — KV cache only

Pure arithmetic projection. We compute the KV-cache footprint at the model's max context, divide by per-chip memory, and read off chip counts. **This is the cache footprint only**, not weights — deployment also needs ~2× weight memory but that's identical between the two stacks. NQX-Core chip assumed to carry 24 GB HBM2e + 100 MB on-die SRAM per `asic/floorplan.md`.

| Model | KV (FP16) | KV (NQX 3+1) | H100 (80 GB each) | NQX-Core (24 GB each) | $/hr H100 → NQX |
|---|---:|---:|---:|---:|---:|
| Llama-3-70B | 320.00 GB | 80.00 GB | 4 chips ($10.00/hr) | 4 chips ($0.20/hr) | **50.0×** |
| Llama-3-8B | 64.00 GB | 16.00 GB | 1 chips ($2.50/hr) | 1 chips ($0.05/hr) | **50.0×** |
| Llama-3-405B | 1008.00 GB | 252.00 GB | 13 chips ($32.50/hr) | 11 chips ($0.55/hr) | **59.1×** |

## Assumptions

- KV factor = 2 (separate K and V tensors). FP16 = 2 bytes/element.
- NQX 3+1 packed = 4 bits per value. Compression ratio = 4.00×.
- H100 spot $2.50/hour (vast.ai 2026-04 average).
- NQX chip TCO amortised over 30 000 hours at $1 500 per chip (target post-tape-out yield) ≈ $0.05/hour.
- We size on a single concurrent request at full context. Real production multiplexes contexts across the same chips.

## What this means for a CFO

- Llama-3-70B at 131,072 ctx needs **4 H100 chips just to hold the KV cache** in FP16. With NQX-Core compression, the same context fits in **4 accelerator chips** (or 0 8-up boards), at **50.0× lower hourly cost**.
- Even more telling: H100s are sold out for the foreseeable future. NQX-Core uses no exotic process node and fab capacity at TSMC N7 is abundant.
- The compression ratio is loss-bounded (RMSE 0.28 at 3 bits, see `docs/paper/results.md`); production deployments typically retain all-or-nothing bit-fidelity for prompt prefix and only compress KV after the first 2k tokens — that hybrid further improves perplexity while preserving these savings.

## Reproduction

```bash
python demos/scaling_demo.py --out demos/scaling_demo.md
```
