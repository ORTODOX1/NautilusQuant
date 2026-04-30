<div align="center">

# 🐚 NautilusQuant

### Deterministic Orthogonal KV-Cache Quantization via Golden Ratio Geometry

[![Status](https://img.shields.io/badge/status-pre--silicon-orange?style=for-the-badge)](nqx-core/docs/PRD.md)
[![Tests](https://img.shields.io/badge/tests-241_passing-brightgreen?style=for-the-badge)](nqx-core/tests)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Triton](https://img.shields.io/badge/triton-GPU%20kernel-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://triton-lang.org)
[![ASIC](https://img.shields.io/badge/RTL-Verilator%20%2B%20Yosys%20%2B%20OpenLane-1f6feb?style=for-the-badge)](nqx-core/rtl)

*Can nature's most irrational number beat Google's random matrices at compressing the memory of neural networks?*

**[Hypothesis](#core-hypothesis)** ·
**[How it works](#how-it-works)** ·
**[NQX-Core 🆕](#nqx-core--pre-silicon-emulator-and-chip-development-kit)** ·
**[Numbers](#numbers-measured-not-promised)** ·
**[Roadmap](#roadmap)** ·
**[Related work](#related-work)** ·
**[Cite](#citation)**

</div>

---

## TL;DR

KV-cache eats ~80% of inference HBM. Industry uses **random rotation + 3-bit quantize** ([TurboQuant](https://arxiv.org/abs/2504.19874), Google ICLR 2026). We replace `random` with `golden angle θ_k = (2π/φ²)·(k+1)·φ^layer` — provably the most uniform angular distribution (Weyl 1916).

The pipeline collapses to a **1.5 KB ROM-LUT** instead of an 8 MB random matrix and runs **bit-identical every time**. Static dataflow processors (Groq LPU, Cerebras WSE, TPU v6) execute it without a PRNG, without branch prediction, without cache misses.

**v0.1.0** ships an upstream-faithful reference (this repo) plus **NQX-Core** — a complete pre-silicon emulator and chip development kit at [`nqx-core/`](nqx-core/): 21-opcode ISA, SystemVerilog RTL skeleton, Yosys + OpenLane (Skywater MPW path), ASIC floorplan and timing closure, FastAPI server, demo runner with side-by-side TurboQuant comparison, and 241 passing tests.

---

## The Problem

LLM inference is **memory-bound**, not compute-bound. KV-cache for a 7B model at 128K context = **64 GB FP16**, ~80% of HBM. Compression directly buys throughput.

State of the art ([TurboQuant, Google ICLR 2026](https://arxiv.org/abs/2504.19874)): apply a random orthogonal rotation, then polar-quantize to 3 bits. It works — but the rotation matrix is **PRNG-derived**, requires GBs of state at scale, has only statistical (O(1/√N)) angular uniformity, and is impossible to map to deterministic-dataflow chips like Groq.

**We ask:** what if random chaos is replaced with the most-irrational angle in mathematics?

---

## Core Hypothesis

The rotation matrix is a product of **non-overlapping Givens pairs** with golden-angle θ:

```
θ_k = (2π / φ²) × (k + 1) ≈ 137.5077640500° × (k + 1)
```

This is the angle that governs sunflower seeds, the Nautilus shell spiral, and phyllotaxis. Hermann Weyl (1916) proved it has the slowest-converging continued fraction `[1; 1, 1, 1, …]` of any number, giving **angular discrepancy O(1/N)** — strictly better than the O(1/√N) of random rotation.

| Property                | Random Rotation (TurboQuant) | **Golden Rotation (NautilusQuant)**          |
|-------------------------|-----------------------------|----------------------------------------------|
| Deterministic           | No (seed-dependent)          | **Yes** (φ and π are constants)              |
| Angular uniformity      | O(1/√N) statistical          | **O(1/N)** mathematical (Weyl)               |
| Reproducibility         | Depends on PRNG state        | **100%** bit-identical every run             |
| LUT size (dim=128)      | 32 KB matrix                 | **1 910 bytes**                              |
| LUT size (dim=1024)     | 8 MB                         | **~12 KB** (-666×)                           |
| State at runtime        | seed + rotation matrix       | **0** (precomputed angles)                   |
| Maps onto static dataflow ASIC? | No (random matmul) | **Yes** (Givens pipeline 1:1)                |

The matrix must be **orthogonal** so attention scores survive: `‖Tv‖ = ‖v‖`, `⟨Tq, Tk⟩ = ⟨q, k⟩`. v1 of this design included `φ^(-i/d)` centripetal scaling that broke orthogonality — fixed in v2 with pure Givens.

---

## How It Works

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ 1. Input │──▶│ 2. Rotate │──▶│ 3. Polar │──▶│ 4. Quant  │──▶│ 5. QJL   │
│   FP16   │   │  Golden φ │   │  (r, θ)  │   │ Lloyd-Max │   │  ±1 bit  │
│  16 bit  │   │  T^T·T=I  │   │          │   │   3 bit   │   │  1 bit   │
└──────────┘   └───────────┘   └──────────┘   └───────────┘   └──────────┘
     HBM ──────────── SRAM (fused, single pass) ────────────▶ HBM
```

Three layers of non-overlapping Givens rotations, all orthogonal by construction:

```python
# Layer 1: adjacent pairs
for k in range(dim // 2):
    givens(v, 2*k, 2*k+1, GOLDEN_ANGLE * (k + 1))

# Layer 2: shifted pairs (offset by 1)
for k in range((dim - 1) // 2):
    givens(v, 2*k+1, 2*k+2, GOLDEN_ANGLE * (k + 1) * φ)

# Layer 3: butterfly with stride dim/4
# (non-overlapping enforced via used-index tracking)
for k in range(dim):
    if not_overlapping(k):
        givens(v, k, (k + dim//4) % dim, GOLDEN_ANGLE * (k + 1) * φ²)
```

Decode is the same in reverse with negated angles: `T⁻¹ = L₁ᵀ·L₂ᵀ·L₃ᵀ`.

---

## NQX-Core — pre-silicon emulator and chip development kit

> **The deterministic dataflow processor that NautilusQuant maps to 1:1.**
> Lives at [`nqx-core/`](nqx-core/). Software-only; ready for FPGA prototyping
> and Skywater 130 nm MPW shuttle.

```
        ┌────────── HBM (off-chip, FP16) ──────────┐
        v                                          ^
   ┌────────┐                                ┌──────────┐
   │  DMA   │--> SRAM_in (24KB) ─────> ... ──│   PACK   │
   └────────┘                                │   3+1bit │
                                             └──────────┘
                                                  ^
   SRAM_in ──> [ VRF FP32, 16 × 128 elem ]        │
                       │                          │
        ┌──────────────┴────────────────┐    ┌────┴──────┐
        v                               v    │   QJL     │
  ┌──────────┐  ┌──────────┐  ┌──────────┐   │ sign+corr │
  │  GU-L1   │─▶│  GU-L2   │─▶│  GU-L3   │   └─────▲─────┘
  │ 64 lanes │  │ 63 lanes │  │ ~32 lns  │         │
  │ adj pair │  │ shifted  │  │ butterfly│         │
  └──────────┘  └──────────┘  └──────────┘         │
                       │                           │
                       v                           │
                 ┌──────────┐    ┌──────────┐      │
                 │  POLAR   │───▶│  QUANT   │──────┘
                 │ √+atan2  │    │ Lloyd-Max│
                 │ 64 lanes │    │   3-bit  │
                 └──────────┘    └──────────┘
                       ^
                 ┌─────┴────┐
                 │ ROM LUT  │  golden cos/sin (≈1.5 KB)
                 │ 191 pair │
                 └──────────┘
```

**Pipeline depth: 18 cycles. Steady-state throughput: 1 vec/cycle.**

| Layer                     | Artifact                                                          |
|---------------------------|-------------------------------------------------------------------|
| ISA + assembler           | [`nqx-core/nqx/`](nqx-core/nqx/) — 21 opcodes (`LDV`, `GVNS`, `POLAR`, `QUANT`, `QJL`, `PACK3`, `MXPACK`, `SUBBIT_ENC`, `ATTN_DOT`, `LDV_ASYNC`, …) |
| Cycle-accurate emulator   | [`nqx-core/nqx/cpu.py`](nqx-core/nqx/cpu.py) — pure NumPy, no torch dep |
| RTL                       | [`nqx-core/rtl/`](nqx-core/rtl/) — 7 SystemVerilog modules + Verilator TB |
| Synthesis                 | [`nqx-core/rtl/synth/`](nqx-core/rtl/synth/) — Yosys flow with sky130 target |
| Open-source tape-out      | [`nqx-core/rtl/openlane/`](nqx-core/rtl/openlane/) — OpenLane2 config (Skywater MPW path) |
| Formal verification       | [`nqx-core/rtl/formal/`](nqx-core/rtl/formal/) — SymbiYosys harness for orthogonality |
| ASIC floorplan + timing   | [`nqx-core/asic/`](nqx-core/asic/) — 50 mm² TSMC 7 nm, 1 GHz target, 9-section tape-out checklist |
| HTTP service              | [`nqx-core/server/`](nqx-core/server/) — FastAPI, monitoring, chaos tests |
| Side-by-side vs TurboQuant| [`nqx-core/demos/side_by_side.md`](nqx-core/demos/side_by_side.md) |
| Pitch deck (10 slides)    | [`nqx-core/demos/pitch.md`](nqx-core/demos/pitch.md)              |
| Pre-silicon SDK           | [`nqx-core/sdk/`](nqx-core/sdk/) — libnqx C ABI, install.sh, errata, programming guide |
| Linux driver skeleton     | [`nqx-core/firmware/driver/`](nqx-core/firmware/driver/) |
| Roadmap (E1-E6)           | [`nqx-core/docs/PRD.md`](nqx-core/docs/PRD.md)                     |

```bash
git clone https://github.com/ORTODOX1/NautilusQuant && cd NautilusQuant/nqx-core
pip install -r requirements.txt
python -m pytest tests -q                  # 241 passing
python run.py verify --dim 128             # acceptance criteria
python run.py bench --vectors 4096         # cycles + throughput + energy
python demos/run_demo.py                   # TurboQuant vs NQX side-by-side
```

---

## Numbers (measured, not promised)

| Metric                                             | Value                | Source                                 |
|----------------------------------------------------|----------------------|----------------------------------------|
| Orthogonality `T^T·T = I` (dim=128)                | err **1.6 × 10⁻⁷**   | `nqx-core/tests/test_orthogonality.py` |
| Roundtrip without quantization                     | RMSE **9.6 × 10⁻⁸**  | same                                   |
| Bit-exact match against `nautilus_triton.py`       | max diff < **10⁻⁴**  | `nqx-core/tests/test_vs_reference.py`  |
| Compression ratio                                  | exactly **4.00×**    | `nqx-core/tests/test_roundtrip.py`     |
| Pipeline depth                                     | **18** cycles        | `nqx-core/docs/architecture.md`        |
| Throughput (steady state)                          | **1 vec / cycle**    | cycle counter                          |
| ROM-LUT size (dim=128)                             | **1 910 bytes**      | `nqx-core/nqx/lut.py`                  |
| Determinism — 100 runs same input                  | **100%** identical   | `bench/determinism.md`                 |
| Energy / encode-vec on TSMC 7 nm (model)           | **≈ 5.1 nJ**         | `nqx-core/nqx/energy.py`               |
| NQ-ISA opcode count                                | **21**               | `nqx-core/nqx/isa.py`                  |
| Unit tests                                         | **241 passing** in <20 s | `pytest tests -q`                  |

---

## Compression Comparison

| Config                                | Bits/value | Compression | LUT/state size  | Determinism |
|---------------------------------------|------------|-------------|-----------------|-------------|
| FP16 baseline                         | 16         | 1.0×        | —               | n/a         |
| KIVI (no rotation)                    | 2          | 2.6×        | per-channel scales | yes      |
| TurboQuant (random)                   | 3 + 1      | 4.0×        | 32 KB (dim=128) → 8 MB (dim=1024) | seed-dependent |
| **NautilusQuant (φ)**                 | **3 + 1**  | **4.0×**    | **1.9 KB → 12 KB** | **bit-identical**  |

`scale + zero-point` overhead is currently the same as TurboQuant (32 bit / group). Whether golden-angle rotation produces a tight enough output distribution to drop them entirely is the **open empirical question** — see proof tasks `T21–T26` in [`nqx-core/audits/prompts/heavy.md`](nqx-core/audits/prompts/heavy.md).

---

## Roadmap

| Stage | What | Status | Notes |
|---|---|---|---|
| **E1** | Software emulator + 21-opcode ISA + assembler | ✅ shipped | `nqx-core/nqx/`, 241 tests |
| **E2** | RTL skeleton (Verilator + Yosys + OpenLane2 + SymbiYosys) | ✅ shipped | `nqx-core/rtl/` |
| **E3** | FPGA bring-up (Alveo U280 / V80 / AWS F1) | ⏳ next | ~3 months, ~$7K |
| **E4** | LLM stack integration (HF Cache / vLLM / Triton) | ⏳ on vast.ai | needs RTX 5090 / B200 — `nqx-core/audits/prompts/heavy-gpu.md` |
| **E5** | Skywater 130 nm tape-out via Efabless Open MPW | ⏳ planned | $0 sponsored slots / $10K commercial |
| **E6** | Commercial ASIC TSMC 12 / 7 nm | 🔮 future | $1.5–5 M depending on node |

---

## Hardware fit

The NautilusQuant pipeline is a **static dataflow** — fixed schedule, zero data-dependent branches, no PRNG, the LUT lives in constant memory. This is exactly the execution model of next-generation inference accelerators that have no hardware scheduler:

| Platform                         | Why it fits              | NQX status                   |
|----------------------------------|--------------------------|------------------------------|
| **Groq LPU** (Tensor Streaming)  | Fully static schedule, no HBM, 230 MB on-chip SRAM | architectural 1:1 mapping |
| **Cerebras WSE-3**               | 44 GB on-chip SRAM, dataflow scheduling | architectural 1:1 mapping |
| **Google TPU v5/v6 (Trillium)**  | Systolic MXU, XLA-compiled static schedule | XLA path planned            |
| **AWS Trainium 3** (3 nm)        | MXFP4 native + dataflow                  | MX fallback supported       |
| **NVIDIA Blackwell B100/B200**   | MXFP4 / NVFP4 in tensor cores 5G          | Triton kernel works (CUDA)  |
| **NVIDIA RTX 5090** (Blackwell consumer) | Same MXFP4 native               | best price/perf for prototyping |
| **AMD MI355X** (CDNA4)           | FP4 / FP6 native                          | ROCm + Triton path          |
| **Skywater 130 nm**              | Open-source PDK, free MPW slots           | OpenLane2 config ready      |

Random rotation does **not** map onto these chips cleanly — it requires a PRNG block and a multi-MB persistent matrix that defeats the on-chip SRAM advantage. Golden rotation maps **for free**.

---

## Risks and Plan B

Three things can break the central thesis. Full analysis in [`RISKS.md`](RISKS.md).

| Risk | What breaks | Mitigation |
|---|---|---|
| **Structural resonance** | Golden angles align with outlier dims → MSE explodes | Fixed permutation layer before rotation |
| **0-overhead failure**  | Angle distribution not predictable enough → still need scale/zero-point | MX-Format fallback (0.25 bit/value overhead) |
| **FP16 drift**          | Roundtrip errors accumulate over 100K-token contexts | Kahan summation / periodic renormalization |

**Worst case:** even if all three risks materialize, the project still wins on determinism, dataflow-compatibility (Groq / Cerebras / TPU), 1.5 KB LUT, and full reproducibility — none of which TurboQuant offers.

Experimental drop-in replacements live in [`plan_b/`](plan_b/) (`quasicrystal.py`, `golden_jl.py`, `phinary.py`, `fractal_hash.py`, `groq_dataflow.py`, `multimodal_spiral.py`). Marked experimental — not yet validated against the main pipeline.

---

## Quick Start (this repo)

```bash
git clone https://github.com/ORTODOX1/NautilusQuant && cd NautilusQuant
pip install -r requirements.txt

# Browse interactively (no install needed)
xdg-open index.html        # 3D pipeline visualization

# Synthetic validation with realistic outliers
python validate_real_kv.py --sweep --dim 128 --count 500

# Real KV-cache from Gemma 3
pip install transformers accelerate
python validate_real_kv.py --model google/gemma-3-4b-it --sweep

# GPU kernel (Triton)
pip install triton
python nautilus_triton.py --dim 128 --n 10000

# Hardware co-design concepts (Concept 1-4)
python nautilus_hardware.py

# Needle-in-a-Haystack on 104K tokens
python benchmark_needle.py --model google/gemma-3-4b-it --method both

# Pure-numpy GloVe vector-search benchmark
python benchmark_glove.py --profile
```

For the **chip development kit**, jump to [`nqx-core/README.md`](nqx-core/README.md).

---

## Related Work

| Method            | Year | Approach                                          | Bits   | Paper                                            |
|-------------------|------|---------------------------------------------------|--------|--------------------------------------------------|
| GPTQ              | 2022 | Layer-wise Hessian quantization                   | 4      | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ               | 2023 | Activation-aware weight protection                | 4      | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| QuIP#             | 2023 | Hadamard rotation + E8 lattice codebooks          | 2      | [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) |
| SqueezeLLM        | 2023 | Dense-and-sparse quantization                     | 3–4    | [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) |
| KIVI              | 2024 | Per-channel KV-cache quantization                 | 2      | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| BitNet b1.58      | 2024 | Ternary weights from training                     | 1.58   | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) |
| **TurboQuant**    | 2026 | **Random** rotation + PolarQuant + QJL            | 3 + 1  | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| **NautilusQuant** | 2026 | **Golden ratio** rotation + PolarQuant + QJL      | 3 + 1  | this repo (paper draft: [`nqx-core/docs/paper/`](nqx-core/docs/paper/)) |

---

## Citation

```bibtex
@software{nautilusquant2026,
  author = {ORTODOX1},
  title  = {NautilusQuant: Deterministic Orthogonal KV-Cache Quantization
            via Golden Ratio Geometry},
  year   = {2026},
  url    = {https://github.com/ORTODOX1/NautilusQuant},
  note   = {Includes NQX-Core pre-silicon emulator (nqx-core/, MIT)}
}

@software{nqxcore2026,
  author = {NQX-Core contributors},
  title  = {NQX-Core: Pre-silicon emulator and chip development kit
            for the NautilusQuant accelerator},
  year   = {2026},
  url    = {https://github.com/ORTODOX1/NautilusQuant/tree/main/nqx-core},
  note   = {Built on NautilusQuant by ORTODOX1, MIT}
}
```

Machine-readable: [`nqx-core/CITATION.cff`](nqx-core/CITATION.cff).

---

<div align="center">

**φ = 1.618 033 988 749 894 848 …**

*The most irrational number meets the most memory-hungry algorithm.*

</div>
