# NQX Programming Guide

## 1. When to Use ENC Macro vs Manual Pipeline

| Criteria | ENC macro | Manual (LDV+GVNS+POLAR+...) |
|----------|-----------|------------------------------|
| **Code size** | 1 instruction per batch | 10 instructions per vector |
| **Throughput** | 1 vec/cycle (steady) | 1 vec/cycle (same pipeline) |
| **Per-vector control** | None — uniform params | Full — different bits, alpha per vector |
| **Intermediate access** | None — fused internally | Any stage can be read (e.g. polar for ATTN_DOT) |
| **Debugging** | Opaque — no insight | Each stage visible in VRF |
| **MX formats** | Not supported | MXPACK/MXUNPACK after POLAR |
| **Sub-bit quant** | Not supported | SUBBIT_ENC after POLAR |
| **Best for** | Production, large uniform batches | Prototyping, mixed-precision, special modes |

**Rule of thumb:** start with manual pipeline during development (visible state), switch to ENC macro for deployment (smaller code, same perf).

---

## 2. Core Patterns

### 2.1 Batched Encode (ENC macro)

Single instruction processes `cnt` vectors from HBM `[src]` through the full pipeline and writes packed result to HBM `[dst]`.

```
source: docs/examples/batch_enc_macro.nqasm
```

See also `programs/encode_dim128.nqasm` for the equivalent manual form.

### 2.2 Manual Per-Vector Encode

Explicit pipeline stages: LDV → GVNS×3 → POLAR → QUANT → QJL → PACK3 → STV. Each vector is independently controlled — different `QUANT` bits or `QJL` alpha per vector.

```
source: docs/examples/streaming_encode.nqasm
```

Shows three vectors at 2-, 3-, and 4-bit quantization.

### 2.3 Encode + Decode Round-Trip

Validate reconstruction quality: ENC writes packed bytes, DEC reads the internal result and reconstructs Cartesian vectors.

```
source: docs/examples/encode_then_decode.nqasm
```

### 2.4 Polar-Domain Attention

Both Q and K are rotated and converted to polar. `ATTN_DOT` computes `r_q·r_k·cos(θ_q−θ_k)` per pair without decoding back to Cartesian — saves the full decode→re-encode round trip.

```
source: docs/examples/attention_score.nqasm
```

For a 2×2 attention matrix with 2 Q and 2 K vectors. See also `programs/attn_dot_dim128.nqasm`.

### 2.5 Sub-Bit Split Quantization

After POLAR, quantize radius and angle with different bit widths. Angles concentrate around golden-ratio multiples, so fewer bits suffice. Standard formula: `bits/value = (r_bits + θ_bits) / 2`.

```
source: docs/examples/subbit_split_quant.nqasm
```

See also `programs/subbit_3_1_dim128.nqasm` (r=3, θ=1) and `programs/encode_subbit_dim128.nqasm`.

### 2.6 MX Block Quantization

OCP Microscaling format: 32-element blocks share an 8-bit exponent. Lower compute cost than Lloyd-Max + QJL, slightly lower quality. Use for latency-sensitive scenarios.

```
source: docs/examples/mxfp4_encode.nqasm
```

See also `programs/mxfp4_dim128.nqasm` (round-trip) and `programs/encode_mx_dim128.nqasm`.

---

## 3. Common Mistakes

### 3.1 Writing to V0 After ENC Without MOV

```nqasm
; WRONG — ENC overwrites V0 internally during pipeline stages
ENC [0x0], [0x10000000], 1
GVNS V0, 0             ; V0 now holds ENC's internal state, not original data

; RIGHT — ENC is self-contained, no MOV needed before it
ENC [0x0], [0x10000000], 1
```

The ENC macro manages registers internally. Manual pipeline: **always** `MOV V_save, V0` before `QUANT`/`QJL`/`PACK3` because those stages modify V0.

### 3.2 Forgetting BARRIER Before STV

```nqasm
; RISKY — no guarantee DMA finished
LDV_ASYNC V0, [0x10000000]
; ... compute ...
STV V0, [0x20000000]   ; V0 might not be fully loaded yet

; SAFE
LDV_ASYNC V0, [0x10000000]
; ... compute ...
BARRIER                  ; wait for all pending DMA
STV V0, [0x20000000]
```

**Rule:** after any `LDV_ASYNC`, insert `BARRIER` before `STV` or before the next `LDV` to the same register. After `BARRIER` + `LDV` (synchronous), no barrier needed before STV.

### 3.3 DEQUANT Without Prior Metadata

```nqasm
; WRONG — UNPACK3 doesn't store mins/maxs needed by DEQUANT
UNPACK3 V0
DEQUANT V0, 3             ; RuntimeError: no metadata

; RIGHT — use DEC macro instead of manual UNPACK3+DEQUANT
DEC [src], [dst], cnt

; Or run QUANT first (in encode pipeline)
QUANT V0, 3               ; stores mins/maxs in S0/S1
DEQUANT V0, 3             ; OK
```

### 3.4 Different dim Between Instructions

Set `dim` in `NQXConfig` before assembly. All instructions in a program operate on the same `dim` from CSR. Do not mix dim=128 and dim=256 vectors in the same batch.

### 3.5 HBM Address Alignment

All LDV/STV addresses must be aligned to `dim × 2` bytes (256 bytes for dim=128). Unaligned addresses cause undefined behavior.

---

## 4. Pipeline Cycle Model

| Stage | Cycles (dim=128) | Note |
|-------|-----------------|------|
| LDV (DMA) | varies | HBM latency, pipelined |
| GVNS ×3 (L1+L2+L3) | 3 | 1 cycle per layer |
| POLAR | 1 | 64 pairs/cycle |
| QUANT (minmax) | 7 | tree reduce over 128 features |
| QUANT (round) | 1 | |
| QJL | 1 | |
| PACK | 1 | |
| STV (DMA) | varies | HBM latency |
| **Total latency** | **~18 cycles** | first vector |
| **Steady-state** | **1 vec/cycle** | pipeline filled |

ENC macro: `cycles = 18 + cnt - 1`. For 16 vectors: 33 cycles.

---

## 5. Naming Conventions

- **V0..V15**: vector registers (each holds `dim` FP32 elements)
- **S0..S7**: scalar registers (scale, zero-point, min, max)
- **HBM**: `0x0000_0000` — external memory, accessed via LDV/STV
- **SRAM_in**: `0x1000_0000` — internal buffer, DMA staging
- **SRAM_out**: `0x1001_0000` — packed output buffer
- **CSR**: `0x3000_0000` — control/status + performance counters
