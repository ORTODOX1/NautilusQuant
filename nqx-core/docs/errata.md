# NQX Errata

## Known Limitations

### L1. Max Batch = 1024 Vectors

**Scope:** ENC/DEC macros, manual LDV pipeline  
**Root cause:** HBM address space is 4 GB (software emulator) / 16 GB (ASIC target). At 64 bytes packed per vector (dim=128, 3+1 bit), theoretical max exceeds 16M, but internal SRAM staging buffers limit in-flight batch to 1024 vectors.

**Workaround:**
- Split batches larger than 1024 into multiple ENC calls
- Use streaming pattern (one LDV → process → STV per vector) for incremental processing
- In LLM inference, KV-cache chunks rarely exceed 512 vectors per layer

### L2. Max dim = 512

**Scope:** VRF, SIMD lanes, LUT size  
**Root cause:** VRF capacity = 16 × dim × 4 bytes. For dim=512, VRF = 32 KB, which exceeds the target SRAM budget for dim=128 (8 KB). SIMD lanes must scale from 64 to 256.

**Workaround:**
- Stay at dim=128 for production — it matches the reference NautilusQuant pipeline
- dim=256/512 supported in software emulator only; ASIC floorplan requires lane count increase

### L3. No Branches or Loops

**Scope:** NQ-ASM ISA  
**Root cause:** NQX has no branch predictor, no PC-relative jumps, no call/return. All programs are linear sequences that halt.

**Workaround:**
- Unroll loops manually — duplicate pipeline blocks for each vector
- Use ENC macro for uniform batches (single instruction encodes cnt vectors)
- Host CPU orchestrates multi-batch workflows by submitting multiple programs

### L4. No Multi-Core / Multi-Instance

**Scope:** Hardware architecture  
**Root cause:** Single pipeline, single VRF, single DMA engine. No cache coherence, no cross-core synchronization.

**Workaround:**
- Deploy multiple NQX instances via PCIe (one device per NUMA node)
- Partition KV-cache by layer across instances
- Host CPU handles load balancing

### L5. Manual Decode (UNPACK3 + DEQUANT) Requires Encode Context

**Scope:** ISA, software emulator  
**Root cause:** `DEQUANT` reads min/max metadata from the preceding `QUANT` instruction. `UNPACK3` only extracts quantized indices and sign bits from the packed byte stream — it does not restore min/max metadata.

**Workaround:**
- Use `DEC` macro instead of manual `UNPACK3` + `DEQUANT` when possible
- For manual decode, run `QUANT` on a known reference vector first to populate scalar registers
- Alternatively, store min/max as a header in HBM alongside packed data (software convention, not yet automated)

### L6. QJL Alpha Limited to Q1.7

**Scope:** ISA encoding  
**Root cause:** QJL alpha is encoded as an 8-bit unsigned normalized value (Q1.7 format, range [0, 2)). Alpha = imm / 128.

**Limitation:**
- Alpha=0x00 → 0.0 (no correction)
- Alpha=0x80 → 0.5 (default, balanced)
- Alpha=0xFF → 1.992 (aggressive correction)

No support for alpha > 2 or negative alpha.

**Workaround:**
- Default alpha=0.5 (0x80) is optimal for uniform KV-cache distributions
- Adjust per-layer if needed (QJL run separately per vector in manual pipeline)

### L7. Energy Model Not Cycle-Accurate

**Scope:** Software emulator (`nqx/energy.py`)  
**Root cause:** Energy is a post-hoc tally using per-operation pJ constants from Horowitz 2014 (45nm). No signal toggling, no wire capacitance, no clock tree.

**Workaround:**
- Use energy numbers for relative comparison between pipelines (e.g., ENC vs MXFP4)
- Absolute energy requires post-synthesis gate-level simulation

### L8. No Training-Loop Integration

**Scope:** Software ecosystem  
**Root cause:** NQX is an inference-only accelerator. There is no backward pass, no gradient computation, no QAT (quantization-aware training) flow.

**Workaround:**
- NQX quantization is fully deterministic — use KL-divergence or RMSE to evaluate quality offline
- Fine-tune with FP16, deploy with NQX quantization

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-30 | Initial pre-silicon errata. All limitations documented from software emulator and ISA spec. |

---

## ASIC Errata

*This section will be populated after first silicon returns. Expected contents:*

- Functional bugs found in hardware vs software emulator
- Electrical issues (setup/hold violations, IR drop, clock jitter)
- PCIe link training failures
- Thermal throttle thresholds
- Metal mask fixes (ECO) per stepping

---

## Per-Silicon Revision Matrix

| Rev | Status | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|-----|--------|----|----|----|----|----|----|----|----|
| v1.0 (emulator) | Active | W | W | W | W | W | W | W | W |

`W` = software workaround available. Empty = fixed.
