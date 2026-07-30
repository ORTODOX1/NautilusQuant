# Why NautilusQuant works — the 5-minute explanation

This file is the explanation a non-specialist hires (CFO, board, investor)
should be able to read in five minutes and understand what we built and why
it is interesting.

## WHY it works — the math in two paragraphs

A KV-cache vector arrives heavy-tailed: a few activations are huge, most are
small. Per-channel quantisation trips on the outliers. Rotating the vector
by a "random enough" orthogonal matrix `T` smears the outliers across all
dimensions, after which uniform-spacing quantisation does well — this is the
trick TurboQuant, QuIP# and several others use.

The catch: random `T` costs `dim²` bytes of state and a CSPRNG to generate.
The **golden angle** `2π/φ²` produces a sequence of rotation angles whose
"spread on the circle" is provably the most uniform any deterministic
sequence can achieve — Weyl's equidistribution theorem (1916) and the
three-distance theorem give a discrepancy `D*_N = O(log N / N)`, versus
`O(1/√N)` for random. We measured this directly in
`bench/angular_uniformity.md`:

| N      | φ-Givens | random (μ ± σ) |
|---:|---:|---:|
| 1024   | 0.00185 | 0.02746 ± 0.00676 |
| 16 384 | 0.00015 | 0.00652 ± 0.00136 |

Empirical slope: φ ≈ −0.929, random ≈ −0.478 — exactly the predicted
asymptotics. **A deterministic golden-ratio rotation is *more* uniform
than random rotation, not less.** That is the entire mathematical claim.

## HOW it works — the five stages of NQX-Core

```
       FP16 KV vector
             │
             ▼
   ┌─────────────────┐
   │ 1. GU.L1  (1c)  │  Givens rotation, layer 1 (adjacent pairs)
   │ 2. GU.L2  (1c)  │  Givens rotation, layer 2 (shifted pairs)
   │ 3. GU.L3  (1c)  │  Givens rotation, layer 3 (butterfly stride)
   │ 4. POLAR  (1c)  │  (x, y) → (r, θ) per pair
   │ 5. QUANT  (1c)  │  Lloyd-Max 3-bit per axis
   │      QJL  (1c)  │  ε-sign correction
   │      PACK (1c)  │  3+1 bit serial pack
   └─────────────────┘
             │
             ▼
       4× compressed bytes → SRAM_out → DMA → HBM
```

Steady-state throughput: **one vector per cycle** after a 7-cycle
prefill. Decode is the mirror image (UNPACK + DEQUANT + IPOLAR + 3 ×
GVNS_INV). Total ASIC die at TSMC 7 nm: **50 mm² @ 1 GHz** —
`asic/floorplan.md`.

## WHEN it pays off — the operating regime

The compression value scales with the cache footprint. Per
`demos/scaling_demo.md`:

| Model        | KV (FP16) | KV (NQX) | $/hr H100 → NQX |
|---|---:|---:|---:|
| Llama-3-8B   | 64 GB    | 16 GB   | 50× |
| Llama-3-70B  | 320 GB   | 80 GB   | 50× |
| Llama-3-405B | 1.0 TB   | 252 GB  | 59× |

The break-even point is `ctx ≥ 4 K`, `dim ≥ 64`. Below that, KV cache is
not yet a memory bottleneck and the compression doesn't justify the
accelerator. Above 32 K context — the modern long-context regime — KV is
the dominant footprint and the savings compound.

## WHO wants it

| Operator | Pain | NQX value |
|---|---|---|
| **vLLM / SGLang clusters** | KV swap to host RAM kills throughput at long ctx | 4× compression keeps KV in HBM, attention stays at peak |
| **llama.cpp on CPU/Mac M-series** | Mac unified memory limits ctx | 4× extends ctx into the same memory budget |
| **Edge AI (Jetson Orin, Apple Neural Engine)** | Tiny on-die SRAM | A 1.9 KB ROM beats a 32 KB random matrix |
| **On-device LLM (phone, IoT)** | No room for a PRNG state | Deterministic encoder fits a 950-byte LUT for dim=64 |

## WHY a chip — the architecture argument

Random rotation is GPU-shaped: a `dim × dim` matmul per layer per vector.
NQX is **dataflow-shaped**: 159 fixed pair indices, three layers, no
runtime decisions, no PRNG state, no off-chip bandwidth except the input
and the packed output. That is the textbook definition of a workload
that wants a static-dataflow ASIC, not a programmable GPU. We're not
porting an algorithm to silicon — the algorithm *is* a silicon
description.

## WHAT'S NEXT

- **Months 0–3**: FPGA bring-up on Alveo U280 (`asic/timing.md` retiming
  plan applied; `rtl/Makefile` already runs Verilator).
- **Months 4–9**: integrations land (`integrations/llama_cpp_kvquant.md`,
  the vLLM plugin). End-to-end perplexity validation on Llama-3.
- **Months 10–18**: tape-out at TSMC 12 nm (cheaper) or 7 nm (faster).
  Pre-tape-out checklist already in `asic/tapeout_checklist.md`.

## TL;DR

A deterministic, ROM-only φ-rotation matches random rotation on quality
(within +8% RMSE), wins by **32× cycles**, **10× energy** and **17× state
size**, and produces the same bytes every time. That is the whole pitch.
References to the supporting numbers are in
`bench/angular_uniformity.md`, `bench/phi_vs_random.md`,
`bench/determinism.md`, `bench/lut_budget.md` and
`bench/energy_proof.md`.
