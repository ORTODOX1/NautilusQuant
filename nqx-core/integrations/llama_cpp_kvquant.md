# llama.cpp KV-quant adapter — design doc

Status: **specification only**. No C++ in this directory yet.
Scope: hand a llama.cpp engineer enough interface to wire NQX KV-cache
quantisation into ggml-cuda without re-deriving the math from `nqx/`.

## 1. Goal

Allow `llama-cli` and `llama-server` to enable NQX KV-cache compression with:

```bash
./build/bin/llama-cli -m model.gguf --kv-quant nqx --kv-bits 3
./build/bin/llama-server -m model.gguf --kv-quant nqx-mxfp4
```

`--kv-quant nqx` selects the φ-Givens + 3-bit + QJL pipeline.
`--kv-quant nqx-mxfp4` selects the OCP-MX block fallback.

Compression target: 4× on KV cache vs FP16, < 2% perplexity delta on
WikiText-2 perplexity for Llama-3.2-1B.

## 2. Where the code lives

| Path | Role |
|---|---|
| `ggml/include/ggml-nqx.h`             | C API surface (NQX KV pack/unpack) |
| `ggml/src/ggml-cuda/nqx_kvquant.cuh`  | CUDA kernel headers |
| `ggml/src/ggml-cuda/nqx_kvquant.cu`   | Kernels: rotate, polar, quant, QJL, pack |
| `ggml/src/ggml-cpu/nqx_kvquant_cpu.c` | Reference path; uses NQXCore via FFI is **not** in scope (we re-derive in C). |
| `src/llama-kv-cache-nqx.cpp`          | KV-cache adapter (similar to `llama-kv-cache-quantized.cpp`) |
| `common/arg.cpp`                      | Add `--kv-quant nqx[-fmt]`, `--kv-bits` flags |

The adapter does **not** depend on Python — math is re-derived from the
same constants exposed in this doc and `nqx/constants.py`.

## 3. C API exposed by ggml-nqx

```c
// ggml/include/ggml-nqx.h

typedef enum {
    GGML_NQX_FORMAT_3PLUS1 = 0,   // 3-bit Lloyd-Max + 1-bit QJL sign
    GGML_NQX_FORMAT_MXFP4  = 1,   // OCP MXFP4, block_size = 32
    GGML_NQX_FORMAT_MXFP6  = 2,
    GGML_NQX_FORMAT_MXFP8  = 3,
    GGML_NQX_FORMAT_SUBBIT_3_1 = 4,  // r=3 bit, theta=1 bit
} ggml_nqx_format;

// Compute packed buffer size in bytes for `n_vec` vectors of `dim`.
size_t ggml_nqx_packed_size(int dim, int n_vec, ggml_nqx_format fmt);

// Encode a contiguous FP16 buffer (n_vec * dim) into packed bytes.
// Out buffer must have ggml_nqx_packed_size() bytes.
// Returns negative ggml-style error code on failure.
int ggml_nqx_encode_f16(const ggml_fp16_t * src,
                        int dim, int n_vec,
                        ggml_nqx_format fmt,
                        void * dst);

// Decode packed bytes back into FP16. dst must hold n_vec * dim halfs.
int ggml_nqx_decode_f16(const void * src,
                        int dim, int n_vec,
                        ggml_nqx_format fmt,
                        ggml_fp16_t * dst);

// Optional fused attention dot product in polar domain.
// q_packed and k_packed must come from the same fmt.
// Result is n_q * n_k FP32 scores.
int ggml_nqx_attn_dot_polar(const void * q_packed,
                            const void * k_packed,
                            int dim, int n_q, int n_k,
                            ggml_nqx_format fmt,
                            float * scores);
```

Errors mirror existing ggml conventions (`GGML_STATUS_FAILED`, etc.).

## 4. Math constants to hard-code in C

These are the same as `nqx/constants.py`. Engineers should treat them as
canonical:

```c
#define NQX_PHI            1.6180339887498948482045868343656381177
#define NQX_GOLDEN_ANGLE   (2.0 * M_PI / (NQX_PHI * NQX_PHI))
#define NQX_DEFAULT_DIM    128
#define NQX_DEFAULT_BITS   3
#define NQX_QJL_ALPHA      0.5
#define NQX_MX_BLOCK_SIZE  32
```

Pair indices for layers L1/L2/L3 are derived once at startup from the
same algorithm as `GoldenAngleLUT._build` (`nqx/lut.py`). Pre-bake into
constant arrays per supported `dim ∈ {64, 128, 256}`. Cos/sin tables are
FP32; total ROM ≤ 2 KB per dim.

## 5. KV-cache adapter wiring

`llama-kv-cache-nqx.cpp` plugs into `llama_kv_cache_unified` and overrides:

- `set_input_kq_mask`: writes a NQX-aware mask that knows about packed
  layout (each 4-byte word is `1 vector / dim / 4` packed bits).
- `cpy_k`, `cpy_v`: copy from FP16 → packed via `ggml_nqx_encode_f16`.
- `get_k`, `get_v`: lazy decode on demand into a `f16` view tensor; see
  prefetch hint below.
- `compute_attn_score`: optional fast path through
  `ggml_nqx_attn_dot_polar`, gated behind `--nqx-fused-attn`.

Per-layer flag `kv_quant_mode` is preserved in the GGUF header so that
re-loaded checkpoints continue with the same fmt.

## 6. Determinism and bit-exactness

The C kernels MUST be bit-exact against `nqx/cpu.py::NQXCore.encode`
when fmt = `GGML_NQX_FORMAT_3PLUS1`, dim = 128, bits = 3, alpha = 0.5.

Test fixture: `tests/nqx_vectors.bin` — pre-quantised reference produced
by `tools/dump_kv_reference.py` (Python tool to be added during T11).
A llama.cpp unit test loads this binary and asserts byte-for-byte
equality with the C encoder output.

For MX formats, bit-exactness must hold against
`nqx/mx_unit.py::MXQuantizer.quantize` (block 32, formula upstream).

## 7. Performance budget

| Path | Latency target (GH200, dim=128) | Notes |
|------|--------------------------------:|-------|
| `ggml_nqx_encode_f16`, batch 1024 | ≤ 200 µs | 3 GVNS + POLAR + QUANT + QJL + PACK |
| `ggml_nqx_decode_f16`, batch 1024 | ≤ 150 µs | UNPACK + DEQUANT + IPOLAR + 3 GVNS_INV |
| `ggml_nqx_attn_dot_polar`, 1×1024 | ≤ 80 µs  | fused, no decode |
| End-to-end attention pass slowdown | ≤ 8% vs FP16 | measured on `llama-bench -p 1024 -n 32` |

If the dot-product fast path is enabled, NQX must outperform plain
dequant + matmul; otherwise the flag is disabled by default.

## 8. CLI integration

```
--kv-quant nqx | nqx-mxfp4 | nqx-mxfp8 | nqx-subbit-3-1
--kv-bits N    (only valid with --kv-quant nqx; default 3)
--nqx-fused-attn (opt-in, requires fmt to be a polar-aware NQX fmt)
```

GGUF metadata fields written into the cache header:

```
nqx.fmt       (uint32)
nqx.dim       (uint32)
nqx.bits      (uint32)
nqx.qjl_alpha (float32)
```

These must be re-read on load and verified against the model's
`hparams.n_embd_head_k`.

## 9. Out of scope for this design doc

- The actual `.cu` kernels (covered in T11 once we have GPU access).
- KV-cache spilling to host RAM (separate llama.cpp PR).
- Streaming-LLM rolling buffer interaction (revisit after T11).
- Quantisation-aware fine-tuning of the model weights.

## 10. Open questions for the C++ engineer

1. Should encode happen on the same stream as the matmul that produced
   the K/V tensor (overlap), or be a synchronous post-step?
   *Recommendation:* same stream, pipelined; latency matches the
   `LDV_ASYNC` model in `nqx/cpu.py`.
2. Is there a path to share the rotation `T` across layers, or do we
   instantiate one ROM per layer?
   *Recommendation:* one ROM globally; `T` is the same for every layer
   because NautilusQuant is data-independent.
3. Do we need a `--kv-quant nqx-cpu-fallback` for AMD ROCm / CPU-only
   builds? *Recommendation:* yes, plain CPU loop as in the Python
   emulator; fast enough for ≤ 1B models.
