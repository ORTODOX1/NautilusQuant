# NQX-Core: A Golden-Ratio Rotation Accelerator for KV-Cache Compression

## Abstract

Auto-regressive LLM inference is increasingly bottlenecked by KV-cache
bandwidth: every decoded token reads the entire cache and the cache grows
linearly with context length. NQX-Core compresses the KV-cache at 4× by
applying a deterministic three-layer Givens rotation parameterised by the
golden ratio φ, projecting each KV vector onto a near-equiangular basis where
3-bit Lloyd–Max quantisation plus a 1-bit sign correction (QJL) preserves
attention scores. Unlike data-driven rotations (QuIP#, TurboQuant) the
NautilusQuant rotation matrix `T` is fixed once at design time and admits a
ROM-only LUT of ≤ 2 KB, which we exploit to design a 50 mm² ASIC with no
training-side dependencies. We present (i) a cycle-accurate software
emulator written in 1.1 K LoC of NumPy, (ii) a full NQ-ISA v2 with
attention-fused (`ATTN_DOT`) and sub-bit operations, and (iii) a
SystemVerilog skeleton that is bit-exact against the emulator on
batch-1024, dim-128 traffic.

## 1. Motivation

KV-cache footprint dominates HBM traffic during decode for long contexts:
for Llama-3.1-8B at 32k tokens the cache exceeds 16 GB in FP16. Two families
of mitigations exist:

1. **Block-quantisation** (KIVI, MX): per-block scale + small mantissa.
   Easy to implement but suffers on heavy-tailed activations.
2. **Random / learned rotations** (QuIP#, TurboQuant): a Hadamard or
   data-driven `T` flattens outliers before quantisation, but burns
   training cycles to find `T` and stores it per-layer.

NautilusQuant (Ryltsov 2025) showed empirically that a *deterministic*
rotation built from `φ = (1+√5)/2` matches QuIP#-quality residuals while
needing zero training. The rotation factorises into three Givens layers of
adjacent / shifted / butterfly pairs, with angles `θ_k = k · 2π/φ²`. We
formalise this as a hardware target.

## 2. Problem Statement

Given KV vectors `x ∈ R^d` (d ∈ {64, 128, 256}), produce a fixed-overhead
encoder/decoder with:

- **Bit-exact reproducibility** against a software reference, so that
  hardware-software co-design can land in production.
- **≤ 5 nJ/vector** encode energy at 7 nm, including HBM round-trip.
- **No training-time state**: the rotation `T` is hard-coded in ROM.
- **Throughput ≥ 10⁵ vectors/s** on FPGA Alveo U280 and ≥ 10⁶ /s on the
  ASIC at 1 GHz steady-state.

## 3. Contributions

1. **NQ-ISA v2** — 23-opcode RISC-style ISA covering encode/decode,
   per-block MX quantisation (`MXPACK`/`MXUNPACK`), sub-bit polar split
   (`SUBBIT_ENC`), fused polar attention (`ATTN_DOT`, opcode 0x80) and
   asynchronous DMA (`LDV_ASYNC` + `BARRIER`) with overlap-aware cycle
   accounting.
2. **NQX-Core software emulator** — pure NumPy, 132 pytest tests,
   bit-exact against the upstream NautilusQuant PyTorch reference for the
   rotation/polar/quant pipeline (`tests/test_vs_reference.py`,
   `tests/test_mx_unit.py`).
3. **RTL skeleton** (`rtl/*.sv`) with cycle-accurate Verilator harness
   driven by hex dumps from the emulator. Same data flows as the Python
   pipeline; replacement of placeholder math with retimed CORDIC and
   Lloyd-Max trees scheduled per the timing closure plan
   (`asic/timing.md`).
4. **ASIC floor-plan** at 50 mm² in TSMC N7 with three power islands and
   ≈ 12 W TDP — see `asic/floorplan.md`. Critical paths and retiming plan
   bring the design within ±50 ps of the 1 GHz target.
5. **Reproducible benchmarks** — orthogonality error 1.6 × 10⁻⁷,
   round-trip RMSE without quantisation 9.8 × 10⁻⁸, 3-bit RMSE 0.28 on
   isotropic Gaussian inputs, energy 8.3 nJ/vec on the emulator (within
   2× of the ASIC target, dominated by Python-side DMA accounting).

## 4. Relation to Prior Work

| System         | Rotation              | Quant      | Train-time? | HW target     |
|----------------|-----------------------|-----------|-------------|---------------|
| KIVI           | none                  | per-channel 2-bit | no  | GPU CUDA       |
| QuIP#          | learned + Hadamard    | 4-bit lattice    | yes  | GPU CUDA       |
| TurboQuant     | random ortho          | 4-bit + outliers | yes  | GPU CUDA       |
| OCP MX         | none                  | block FP4/8/INT8 | no   | GPU + Blackwell |
| **NQX-Core**   | **φ-Givens (fixed)**  | **3+1 bit + MX fallback** | **no** | **FPGA / ASIC** |

NautilusQuant occupies a unique cell: deterministic rotation lets the
accelerator omit any per-layer state, freeing on-chip SRAM for VRF and
SRAM scratchpads. We treat the φ-rotation property as a *hardware
opportunity*, not a quantisation trick.

## 5. Paper Structure

Section 2 (`results.md`) reports the measured emulator numbers and
compares against the ASIC budget. Section 3 covers ISA design rationale.
Section 4 details the three-stage Givens micro-architecture. Section 5
describes the cycle-accurate Verilator co-simulation methodology.
Section 6 outlines the FPGA bring-up plan on Alveo U280. Section 7
discusses limitations (no training-loop integration yet, multi-GPU
sharding deferred).

## References

- Ryltsov A., *NautilusQuant: Golden-Ratio KV-Cache Compression*, GitHub
  hermandoronin/NautilusQuant, 2025.
- TurboQuant, *arXiv 2504.19874*, 2025.
- Liu et al., *KIVI: Plug-and-Play 2bit KV-Cache Quantization*,
  arXiv 2402.02750.
- Tseng et al., *QuIP#: Even Better LLM Quantization*, arXiv 2402.04396.
- Open Compute Project, *OCP MX Format Specification*, v1.0, 2023.
- Horowitz, M., *Computing's Energy Problem*, ISSCC 2014 (per-op pJ
  numbers used in `nqx/energy.py`).
