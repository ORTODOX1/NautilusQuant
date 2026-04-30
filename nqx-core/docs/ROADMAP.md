# NQX-Core Roadmap

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  E1 ✅ Software emulator     ─────────────────────  v0.1.0 (NOW)    │
│         │                                                           │
│  E2 ✅ RTL skeleton          ─────────────────────  v0.1.0 (NOW)    │
│         │                    (synthesizable, not yet placed&routed) │
│         ▼                                                           │
│  E3 ⏳ FPGA bring-up          ▒▒▒▒▒▒░░░░░░░░░░  3 months  ($7K)     │
│         │                                                           │
│         ▼                                                           │
│  E4 ⏳ LLM stack integration  ▒▒▒▒░░░░░░░░░░░░  3-6 months  ($)     │
│         │                                                           │
│         ▼                                                           │
│  E5 ⏳ Skywater MPW shuttle   ▒▒░░░░░░░░░░░░░░  6-9 months  ($0!)   │
│         │                    (open-source path through Efabless)    │
│         ▼                                                           │
│  E6 ⏳ Commercial ASIC        ░░░░░░░░░░░░░░░░  12-18 months ($1.5-5M)│
│                              (TSMC 12nm or 7nm)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## E1 — Software emulator ✅ DONE (v0.1.0)

| Artifact | Location |
|---|---|
| Functional ISS | `nqx/cpu.py` |
| Cycle-accurate emulator | `nqx/pipeline.py` + counters |
| 21-opcode ISA | `nqx/isa.py` |
| Assembler ↔ Disassembler | `nqx/assembler.py`, `nqx/disassembler.py` |
| 7 functional units | `nqx/functional_units.py` |
| Energy model | `nqx/energy.py` |
| 241 tests | `tests/` |

**Acceptance**: orthogonality 1.6e-7, roundtrip 9.6e-8, bit-exact vs upstream.

## E2 — RTL skeleton ✅ DONE (v0.1.0)

| Artifact | Location |
|---|---|
| 7 SystemVerilog modules | `rtl/*.sv` |
| Verilator testbench | `rtl/tb_nqx.sv` |
| Yosys synth flow | `rtl/synth/` |
| OpenLane config | `rtl/openlane/` |
| SymbiYosys formal | `rtl/formal/` |

**Acceptance**: skeleton compiles in Verilator. `make synth` outputs gate
count. `make formal` proves orthogonality property.

## E3 — FPGA bring-up ⏳ NEXT (3 months, ~$7K)

| Step | Tool | Timeline |
|---|---|---|
| Verilator + waveform analysis | open-source | 1 week |
| Yosys synth → gate count target | open-source | 1 week |
| Vivado place & route on Alveo U280 | AMD Vivado | 2 weeks |
| Bring-up + perf benchmark | Alveo U280 ($7K) или AWS F1 ($1.6/час) | 4 weeks |
| Real KV-cache validation | with E4 partial | 4 weeks |

**Acceptance**: encode 4096 vec dim=128 ≥ 10K vec/s on FPGA at ≥ 100 MHz.

## E4 — LLM stack integration ⏳ NEXT (3-6 months, $)

Pending tasks in `audits/prompts/heavy-gpu.md`:

| ID | Adapter | What |
|---|---|---|
| T10 | HuggingFace Cache hook | `integrations/hf_kv_hook.py` for `transformers` |
| T11 | vLLM plugin | `integrations/vllm_kvquant.py` for vLLM `kv_cache_quantization_methods` |
| T12 | Triton kernel in server/ | replace PyTorch reference with `NautilusQuantTriton` |

**Hardware**: vast.ai RTX 5090 ($0.6/hr) or B200 ($5/hr) for benchmarks.

**Acceptance**: Llama-3.2-1B perplexity delta < 5% with NQX KV-quant
vs FP16 baseline. vLLM with `--kv-cache-quant nqx` produces ≥ FP16 throughput.

## E5 — Skywater 130nm MPW ⏳ NEXT (6-9 months, $0)

The **open-source** path to real silicon. No commercial NDA needed.

| Step | Tool | Timeline |
|---|---|---|
| OpenLane2 full flow | docker openlane2 | 1 week |
| Floor-plan + place&route + DRC | open-source | 2 weeks |
| Submit to Efabless Open MPW shuttle | Efabless Caravel | 1 month |
| Manufacturing | Skywater Foundry | 4-5 months |
| Bring-up on Caravel testbench | shipped board | 2 weeks |

**Cost**: $0 for Open MPW shuttle (sponsored slots) or $10K for commercial slot.

**Acceptance**: live silicon executing NQ-ISA programs at 100 MHz, bit-exact
to Verilator simulation.

## E6 — Commercial ASIC ⏳ FUTURE (12-18 months, $1.5-5M)

Volume production path. Requires foundry NDA and team.

| Process | Tape-out cost | Target frequency | Use case |
|---|---|---|---|
| GlobalFoundries 12nm | $1.5-2M | 500 MHz | Mid-volume edge |
| TSMC 12nm | $2-3M | 700 MHz | Mid-volume datacenter |
| TSMC 7nm | $5M+ | 1 GHz (our spec) | Hyperscale inference |
| TSMC 5nm/3nm | $15-30M | edge of art | Compete with NVIDIA directly |

**Acceptance**: PPA targets met (perf, power, area), DRC/LVS/STA clean,
yield > 80% on first run.

## Parallel tracks (non-blocking on E3-E6)

| Track | Status |
|---|---|
| **Pre-silicon SDK Часть B** (libnqx C ABI, Linux driver skeleton, boot ROM, programming guide, errata, install.sh, SDK README) | 🔄 In progress (DeepSeek) |
| **Paper draft** (`docs/paper/`) | ✅ Initial draft, needs benchmarks update post-E3 |
| **Programming guide** (`docs/programming_guide.md`) | 🔄 Pending SDK12 |

## Out of scope (we're not doing this)

- ❌ Training new LLMs (we're inference compression only)
- ❌ Compete with NVIDIA at 3nm (too expensive, not the target)
- ❌ General-purpose CPU (we're domain-specific)
- ❌ Quantum hardware (the name is φ-irrationality, not quantum)
- ❌ GUI applications (web UI / API only)
