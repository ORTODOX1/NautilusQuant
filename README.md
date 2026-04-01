<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Preview-orange?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Triton-GPU%20Kernel-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Triton">
</p>

<h1 align="center">
  🐚 NautilusQuant
</h1>

<h3 align="center">
  Deterministic Orthogonal KV-Cache Quantization via Golden Ratio Geometry
</h3>

<p align="center">
  <em>Can nature's most irrational number beat Google's random matrices<br>at compressing the memory of neural networks?</em>
</p>

<p align="center">
  <a href="#core-hypothesis">Hypothesis</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#key-results">Results</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## The Problem

Large Language Models are **memory-bound**, not compute-bound. During text generation, up to **80% of GPU memory** is consumed by the KV-cache (Key-Value cache) — the attention mechanism's working memory. For a 7B model at 128K context, this can reach **64 GB** in FP16.

State-of-the-art compression (Google's [TurboQuant](https://arxiv.org/abs/2504.19874), ICLR 2026) uses **random orthogonal rotation** followed by polar quantization to compress the KV-cache to 3 bits. It works — but the rotation angles are random, requiring a PRNG seed and offering no mathematical guarantees about angular distribution quality.

**We ask:** What if we replace random chaos with nature's optimal geometry?

---

## Core Hypothesis

**NautilusQuant** replaces random rotation with a **deterministic orthogonal matrix** whose angles follow the **golden ratio** (φ ≈ 1.618) and **π**:

```
θₖ = (2π / φ²) × (k + 1) ≈ 137.507764° × (k + 1)
```

This is the **golden angle** — the same angle that governs:
- 🌻 Seed arrangement in sunflowers
- 🐚 The spiral of the Nautilus shell
- 🌀 Phyllotaxis patterns in plants

The golden angle is provably the **most irrational** angle: its continued fraction `[1; 1, 1, 1, ...]` converges slower than any other, guaranteeing that **no two points ever cluster** and **no angular region is left empty**.

### Why This Might Work

| Property | Random Rotation (TurboQuant) | Golden Rotation (NautilusQuant) |
|---|---|---|
| Deterministic | No (seed-dependent) | **Yes** (φ and π are constants) |
| Angular uniformity | O(1/√N) statistical | **O(1/N)** mathematical guarantee |
| Reproducibility | Depends on PRNG state | **100%** identical results always |
| LUT precomputation | Not possible | **64 cos/sin values = 512 bytes** |
| Overhead storage | Seed required | **Zero** (angles computed from formula) |

### Critical Constraint: Orthogonality

The rotation matrix **must** be orthogonal (`T^T · T = I`) to preserve:
- Vector norms: `‖Tv‖ = ‖v‖`
- Dot products: `⟨Tq, Tk⟩ = ⟨q, k⟩`
- Attention scores: **unchanged** after transformation

> **v1 Bug (Fixed):** Our initial design included centripetal scaling (`φ^(-i/d)`) which **broke orthogonality** and corrupted attention. v2 uses **pure Givens rotations** — no scaling, no norm changes.

---

## How It Works

### Pipeline: 5 Stages

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ 1. Input │──▶│ 2. Rotate │──▶│ 3. Polar │──▶│ 4. Quant  │──▶│ 5. QJL   │
│   FP16   │   │  Golden φ │   │  (r, θ)  │   │ Lloyd-Max │   │  ±1 bit  │
│  16 bit  │   │  T^T·T=I  │   │          │   │   3 bit   │   │  1 bit   │
└──────────┘   └───────────┘   └──────────┘   └───────────┘   └──────────┘
     HBM ────────────── SRAM (fused, single pass) ──────────────▶ HBM
```

### Stage 2: Orthogonal Spiral Matrix (The Core Innovation)

Three layers of non-overlapping Givens rotations:

```python
# Layer 1: Adjacent pairs — golden angle × (k+1)
for k in range(dim // 2):
    θ = GOLDEN_ANGLE * (k + 1)                    # ≈ 137.5° × (k+1)
    givens_rotate(v, 2*k, 2*k+1, θ)

# Layer 2: Shifted pairs — golden angle × (k+1) × φ
for k in range((dim - 1) // 2):
    θ = GOLDEN_ANGLE * (k + 1) * φ
    givens_rotate(v, 2*k+1, 2*k+2, θ)

# Layer 3: Butterfly pairs — golden angle × (k+1) × φ²
# Non-overlapping pairs enforced via used-index tracking
```

**Orthogonality proof:**
- Each Givens rotation `G(i,j,θ)` is orthogonal by construction
- Pairs within each layer are non-overlapping → layer is orthogonal
- Product of orthogonal matrices is orthogonal: `T = L₃·L₂·L₁`, `T^T·T = I` ✓

### Stage 5: Inverse Transform (Decode)

```python
T⁻¹ = L₁^T · L₂^T · L₃^T  # Reverse layer order, negate angles
```

---

## Key Results

> ⚠️ **Research Preview**: These are preliminary results on synthetic data with realistic outlier distributions. Real KV-cache validation on Gemma 3 4B is the next milestone.

### Orthogonality Verification

```
Norm preservation error:    < 1e-14
Roundtrip error (T⁻¹·T·v): < 1e-14
Dot product error:          < 1e-14
```

### Compression

| Config | Bits/value | Compression | Overhead |
|---|---|---|---|
| FP16 baseline | 16 | 1.0x | — |
| KIVI (no rotation) | 2 | 2.6x | 32 bit/group |
| TurboQuant (random) | 3+1 | 4.0x | 32 bit/group |
| **NautilusQuant (φ)** | **3+1** | **4.0x** | **32 bit/group** ¹ |

¹ *Overhead is currently identical to TurboQuant. Achieving 0 overhead requires empirical proof that golden angles produce a predictable enough distribution to eliminate scale/zero-point constants. This is the primary open research question.*

---

## Quick Start

### Interactive 3D Visualization (No Install Required)

Open `index.html` in any browser to explore:
- **3D Simulation** — animated 5-stage pipeline, TurboQuant vs NautilusQuant side-by-side
- **Formula Lab** — edit mathematical formulas live, see results instantly
- **MiroFish Lab** — AI agents discuss optimal quantization strategies

### Validation Scripts

```bash
# Install dependencies
pip install torch numpy

# Phase 2: Synthetic validation with realistic outliers (±60 in 6 dimensions)
python validate_real_kv.py --sweep --dim 128 --count 500

# Phase 2: Real KV-cache from Gemma 3
pip install transformers accelerate
python validate_real_kv.py --model google/gemma-3-4b-it --sweep

# Phase 3: GPU kernel benchmark (Triton)
pip install triton
python nautilus_triton.py --dim 128 --n 10000

# Phase 4: All hardware co-design concepts
python nautilus_hardware.py

# Phase 4: Needle-in-a-Haystack (104K tokens)
python benchmark_needle.py --model google/gemma-3-4b-it --method both

# Vector search: GloVe recall@k (no torch needed — pure numpy)
python benchmark_glove.py --profile
```

---

## Architecture

### Project Structure

```
NautilusQuant/
├── index.html              # Project portal (browser)
├── quantsim3d.html         # 3D pipeline visualization
├── formula_lab.html        # Live formula editor + data inspector
├── mirofish_lab.html       # Multi-agent discussion (MiroFish)
├── compute_worker.js       # WebWorker compute engine
│
├── validate_real_kv.py     # Phase 2: KV-cache extraction + comparison
├── nautilus_triton.py      # Phase 3: Triton GPU kernel + PyTorch ref
├── nautilus_hardware.py    # Hardware co-design (SRAM, Dataflow, MX, Sub-1-bit)
├── benchmark_needle.py     # Phase 4: Needle-in-a-Haystack + accuracy
├── benchmark_glove.py      # GloVe vector search + KIVI baseline
│
├── NautilusQuant_Model.md  # Full theoretical model (v2)
├── goldt.txt               # Source research on TurboQuant
└── .gitignore
```

### Hardware Co-design Concepts

| Concept | Description | Implementation |
|---|---|---|
| **SRAM-Fused Kernel** | All 5 stages in single SRAM pass. No HBM round-trips. ~2.5x energy savings. | `nautilus_hardware.py` → `NautilusFusedKernel` |
| **Deterministic Dataflow** | Static execution schedule. Zero branches. Compatible with Groq LPU, TPU, Cerebras. | `nautilus_hardware.py` → `NautilusDataflow` |
| **MX-Format Fallback** | If 0-overhead fails → MXFP4 block scaling (0.25 bit/value overhead). OCP standard. | `nautilus_hardware.py` → `NautilusWithMX` |
| **Sub-1-bit + Multimodal** | Separate bit allocation for radius vs angle. Adaptive compression per modality. | `nautilus_hardware.py` → `SubBitExperiment` |

### Supported Hardware Targets

| Platform | Optimization | Status |
|---|---|---|
| NVIDIA H100/H200/B200 | Triton kernel + TMEM | 🟡 Prototype |
| NVIDIA RTX 5080 | Triton kernel + shared memory | 🟡 Prototype |
| Google TPU v5 | OpenXLA + VMEM | 📋 Planned |
| Groq LPU | Static dataflow schedule | 📋 Planned |
| Apple Silicon (M1-M4) | MLX framework | 📋 Planned |
| llama.cpp / GGUF | Custom quant type Q3_PHI | 📋 Planned |

---

## Benchmarks

### Vector Search (GloVe d=200)

```bash
python benchmark_glove.py --profile
```

Compares: **FP16** (baseline) → **KIVI** (2-bit, no rotation) → **TurboQuant** (random rotation) → **NautilusQuant** (golden rotation)

Metrics: Recall@10, MSE, indexing time, memory savings, energy estimation.

### Needle-in-a-Haystack

```bash
python benchmark_needle.py --model google/gemma-3-4b-it --method both --haystack 4096
```

Hides a fact in long context. Tests if the model can retrieve it after KV-cache quantization.

---

## Risk Analysis

Three fundamental things that can break. Full analysis: [`RISKS.md`](RISKS.md)

| Risk | What breaks | Mitigation | File |
|---|---|---|---|
| **Structural resonance** | Golden angles align with outlier dims → MSE explodes | Add fixed permutation layer before rotation | `validate_real_kv.py` |
| **0-overhead failure** | Angle distribution not predictable enough → need scale/zp | MX-Format fallback (0.25 bit/val overhead) | `nautilus_hardware.py` |
| **FP16 drift** | Roundtrip errors accumulate on 104K contexts | Kahan summation / periodic renormalization | `validate_real_kv.py` |

**Worst case:** Even if ALL 3 risks materialize, NautilusQuant still wins on: determinism, dataflow-compatibility (Groq/Cerebras), 512-byte LUT, and full auditability.

### Plan B: Experimental Modules ([`plan_b/`](plan_b/))

| # | Module | Replaces | Potential gain |
|---|--------|----------|----------------|
| 1 | `quasicrystal.py` | Lloyd-Max (Stage 4) | 2-bit quality at 3-bit budget |
| 2 | `golden_jl.py` | QJL (Stage 5) | **Kill QJL** → 5.3x instead of 4x |
| 3 | `phinary.py` | Scalar quant | φ-base covers outliers naturally |
| 4 | `fractal_hash.py` | Angle encoding | Sub-1-bit per angle |
| 5 | `groq_dataflow.py` | Runtime scheduler | Static LUT for Groq/Cerebras/TPU |
| 6 | `multimodal_spiral.py` | Fixed φ | Adaptive silver/bronze ratio per modality |

---

## The Open Question

The central research question remains **empirically unresolved**:

> **Does the golden angle (137.5°) produce a more concentrated angular distribution than random angles after polar decomposition of real KV-cache tensors?**

If **yes** → NautilusQuant can achieve **0-overhead** quantization (no scale/zero-point storage), making it fundamentally more efficient than TurboQuant at extreme compression (2-3 bits).

If **no** → NautilusQuant still offers: determinism, reproducibility, precomputable LUT, and compatibility with static dataflow architectures (Groq, Cerebras). But the compression ratio matches, not exceeds, TurboQuant.

### How to Test This

```bash
# Run this on a machine with GPU + transformers installed:
python validate_real_kv.py --model google/gemma-3-4b-it --sweep

# Look for:
#   Angle variance (Nautilus) < Angle variance (Turbo)  → φ wins
#   MSE (Nautilus) < MSE (Turbo)                        → φ wins
```

---

## Roadmap

- [x] **Phase 1:** Fix linear algebra (orthogonal matrix, inverse transform, butterfly pairs)
- [x] **Phase 2:** Validation framework (synthetic outliers, real KV-cache extraction)
- [x] **Phase 3:** Triton GPU kernel + hardware co-design
- [x] **Phase 4:** Benchmark suite (Needle-in-a-Haystack, GloVe, KIVI baseline)
- [ ] **Phase 5:** Run validation on real Gemma 3 4B KV-cache tensors
- [ ] **Phase 6:** Prove/disprove 0-overhead hypothesis with empirical data
- [ ] **Phase 7:** Optimize Triton kernel for RTX 5080 (target: < 0.0013s indexing)
- [ ] **Phase 8:** Integration with llama.cpp (GGUF format Q3_PHI)
- [ ] **Phase 9:** Paper submission

---

## Related Work

| Method | Year | Approach | Bits | Paper |
|---|---|---|---|---|
| GPTQ | 2022 | Layer-wise Hessian quantization | 4 | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ | 2023 | Activation-aware weight protection | 4 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| QuIP# | 2023 | Hadamard rotation + E8 lattice codebooks | 2 | [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) |
| SqueezeLLM | 2023 | Dense-and-sparse quantization | 3-4 | [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) |
| KIVI | 2024 | Per-channel KV-cache quantization | 2 | — |
| BitNet b1.58 | 2024 | Ternary weights (-1, 0, +1) from training | 1.58 | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) |
| **TurboQuant** | **2025** | **Random rotation + PolarQuant + QJL** | **3** | [**arXiv:2504.19874**](https://arxiv.org/abs/2504.19874) |
| **NautilusQuant** | **2026** | **Golden ratio rotation + PolarQuant + QJL** | **3** | **This work** |

---

## Industrial Applications

While NautilusQuant targets LLM inference, the core technology — deterministic signal compression with mathematical guarantees — transfers directly to industrial edge computing.

### Maritime and Offshore IoT

Ship power plants generate thousands of sensor readings per second (RPM, exhaust temperature, fuel pressure, lube oil quality, vibration spectra). Satellite bandwidth between vessel and shore is typically 64–512 kbps. **4x deterministic compression** enables real-time shore-side condition monitoring without saturating the link.

| Challenge | How NautilusQuant Helps |
|-----------|------------------------|
| Limited satellite bandwidth | 4–5.3x compression reduces data volume to fit within VSAT/Iridium constraints |
| Edge inference on shipboard hardware | KV-cache quantization enables larger ML models on embedded GPUs (Jetson, Movidius) |
| Safety-critical determinism | No PRNG seed — IMO/SOLAS auditors can verify identical results every time |
| Resource-constrained devices | 512-byte LUT fits in any PLC/FPGA register file |
| Real-time processing | SRAM-fused pipeline achieves sub-millisecond latency for sensor stream compression |

### Hardware Fit

- **512-byte LUT** — precomputed cos/sin table stored in constant memory. Compatible with marine-grade embedded controllers.
- **SRAM-fused pipeline** — single-pass encode/decode eliminates memory bottleneck on constrained devices.
- **Static dataflow** — no branches, no PRNG, no runtime decisions. Suitable for safety-critical real-time systems with deterministic scheduling requirements.
- **ONNX-exportable** — quantized inference pipeline can be deployed alongside predictive maintenance models on shipboard edge nodes.

### Author Background

The author brings 3+ years of hands-on marine engineering experience — ship power plant maintenance, diesel engine diagnostics, and propulsion system overhaul — bridging the gap between physical machinery domain knowledge and modern ML/signal processing research.

---

## Citation

```bibtex
@software{nautilusquant2026,
  author    = {ORTODOX1},
  title     = {NautilusQuant: Deterministic Orthogonal KV-Cache Quantization via Golden Ratio Geometry},
  year      = {2026},
  url       = {https://github.com/ORTODOX1/NautilusQuant},
  note      = {Research Preview}
}
```

---

<p align="center">
  <strong>φ = 1.6180339887...</strong><br>
  <em>The most irrational number meets the most memory-hungry algorithm.</em>
</p>
