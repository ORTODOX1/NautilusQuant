# Contributors

NQX-Core was built by humans + AI agents working together. Every commit and
every artifact is auditable.

## Core idea (the math)

| Person | Role | Contribution |
|---|---|---|
| **[@ORTODOX1](https://github.com/ORTODOX1)** | Originator | NautilusQuant insight — using the golden angle (Weyl equidistribution, 1916) instead of random rotation for KV-cache compression. Reference math in [`nautilus_triton.py`](https://github.com/ORTODOX1/NautilusQuant/blob/main/nautilus_triton.py) and [`nautilus_hardware.py`](https://github.com/ORTODOX1/NautilusQuant/blob/main/nautilus_hardware.py). NQX-Core is fundamentally an emulator and hardware design kit on top of his algorithm. |

## Engineering build (this repo)

The full chip development kit was built using parallel AI agents
(Claude + DeepSeek-V4) coordinated through structured task lists in
[`audits/prompts/`](../audits/prompts/). Every task, every prompt, every result
is preserved for auditability.

### Heavy track (architecture, RTL, ASIC, demo, proof tasks)

**Agent**: Anthropic Claude 4.7
**Tasks**: T1-T26 from `heavy.md`, D1-D9 from `demo.md`, SDK1-SDK8 from `sdk.md`

| Domain | What was built |
|---|---|
| ISA design | NQ-ISA v2 with 21 opcodes, encoding format, assembler, disassembler |
| Functional units | GivensUnit (vectorized), PolarUnit, QuantUnit, QJLUnit, PackUnit (`np.packbits`-based), MXQuantizer, SubBitUnit, AttentionUnit |
| Performance | Sub-millisecond encode hot path, async DMA model, perf counters |
| RTL | 7 SystemVerilog modules + Verilator testbench + ROM mem-file generator |
| Synthesis | Yosys flow + OpenLane2 config (Skywater MPW path) |
| Formal | SymbiYosys harness with SVA assertions for orthogonality |
| ASIC docs | Floor-plan (50 mm² TSMC 7nm) + timing closure (1 GHz) + tape-out checklist (9 sections) |
| Proof tasks | Angular uniformity, linear-vs-Lloyd-Max, φ-vs-random, determinism, LUT budget, energy delta |
| Demo | TurboQuant baseline emulation, end-to-end attention demo, 70B scaling projection, side-by-side comparison, why-it-works narrative, 10-slide pitch deck, ASCII visualizations, demo runner |
| Server | FastAPI app with auto CPU/GPU backend, structured errors |
| CI/CD | `.github/workflows/ci.yml`, multi-arch Docker (amd64+arm64) |
| Pre-silicon SDK | Random Instr Generator, coverage tracking, JTAG model, perf counters |

### Routine track (tests, scenarios, fixtures, monitoring, debug, docs)

**Agent**: DeepSeek-V4 Flash (via Anthropic-compatible endpoint)
**Tasks**: R1-R28 from `routine.md`, S1-S24 from `scenarios.md`, SDK9-SDK15 (in progress)

| Domain | What was built |
|---|---|
| Test coverage | 7 dedicated test files for individual units (givens, polar, quant, qjl, pack, memory, assembler) — went from 32 tests at start to 241 |
| Scenarios | Multi-turn chat session, long-context (64K), variable batch, multimodal, edge inputs (NaN/Inf) |
| Fixtures | Realistic Llama-like KV generator, golden snapshots, adversarial inputs |
| Monitoring | Access log middleware (JSON Lines), Prometheus metrics endpoint, deep health check (LUT hash + T^T·T self-test), structured error responses |
| Chaos / fault-injection | OOM behaviour, corrupt payloads, concurrent requests, slow client |
| Debug tools | `nqx-debug` CLI with stage-by-stage encode inspector, replay harness, automatic failure snapshots |
| Smoke / load | `deploy/smoke.sh` for post-deploy validation, `deploy/load_test.py` (100 concurrent, p50/p95/p99) |
| Notebooks | Jupyter intro, comparison vs TurboQuant, attention demo |
| Bench / dataset | Synthetic KV-cache generator with realistic outliers, full-config bench runner |
| Linting / style | Ruff + black sweep on all of `nqx/`, `server/`, `tests/`, `tools/` |
| Status tooling | `nqx-status`, `nqx-stats`, `tools/clean.sh` |
| Performance | HBM lazy paging optimization (100 MB read in 0.113 s), `--quiet` mode, `--summary` flag for nqx-audit |
| Documentation | `docs/benchmarks.md` with results across all dim×batch combinations, README updates |

### Architectural review

**Agent**: OpenAI Codex (via `/second` second-opinion calls)
**Role**: Independent review on heavy-track changes before merging.

### Coordination layer

| Tool | Purpose |
|---|---|
| `audits/prompts/heavy.md` | 26-task list for architecture/RTL/ASIC track |
| `audits/prompts/routine.md` | 28-task list for tests/docs/tooling track |
| `audits/prompts/demo.md` | 9-task list for pitch and demo materials |
| `audits/prompts/scenarios.md` | 24-task list for production-readiness |
| `audits/prompts/sdk.md` | 15-task list for pre-silicon SDK |
| `audits/prompts/heavy-gpu.md` | 3-task list for vast.ai GPU integrations |
| `tools/cli/nqx-launch-all` | Spawns Heavy + Routine agents in parallel windows |
| `tools/cli/nqx-status` | Live progress dashboard |

## How to contribute

See [`CONTRIBUTING.md`](../CONTRIBUTING.md). Open work items are visible in
the prompt files above as `[ ]` checkboxes.

## Theoretical references

- **Hermann Weyl** (1916). *Über die Gleichverteilung von Zahlen mod. Eins*.
  Mathematische Annalen, 77(3), 313–352. — proves that φ has the most-irrational
  continued fraction expansion, giving angular discrepancy O(1/N) — the
  mathematical bedrock of NautilusQuant.
- **OCP Microscaling Spec** (2023). *MX format for FP4/FP6/FP8/INT8 block
  quantization*. — used as Plan B in `nqx/mx_unit.py`.
- **TurboQuant** (2026). Online Quantization for KV-Cache via Random Orthogonal
  Rotation. — the random-rotation predecessor we benchmark against.

---

_If your name should be here and isn't, open an issue._
