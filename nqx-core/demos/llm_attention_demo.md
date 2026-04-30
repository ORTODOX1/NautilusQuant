# End-to-end attention demo on NQX-compressed K/V

Synthetic single-layer attention with `n_heads=8`, `seq=512`, `dim=128`. K and V are encoded via `NQXCore.encode` and decoded before attention scoring. Q is kept FP32 for the comparison. Inputs include 1/128 outliers at 6σ to stress the rotation+quant pipeline.

| Metric | Value |
|---|---:|
| RMSE attention output (NQX vs FP16) | 0.0111 |
| RMSE K reconstruction               | 0.1979 |
| RMSE V reconstruction               | 0.1974 |
| KV cache bytes (FP16)               | 2,097,152 B |
| KV cache bytes (NQX 3+1)            | 524,288 B |
| **Compression ratio**               | **4.00×** |
| Cycles per decoded token (NQX)      | 32 |

## Reproduction

```bash
python demos/llm_attention_demo.py --n-heads 32 --seq 2048 --dim 128
```
