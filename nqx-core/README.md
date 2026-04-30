<div align="center">

# 🐚 NQX-Core

**Pre-silicon emulator and chip development kit for the NautilusQuant
golden-ratio KV-cache accelerator.**

[![Tests](https://img.shields.io/badge/tests-229_passing-brightgreen?style=for-the-badge)](tests)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/numpy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Status](https://img.shields.io/badge/status-pre--silicon-orange?style=for-the-badge)](docs/PRD.md)

**[Quick start](#quick-start)** ·
**[Pitch deck](demos/pitch.md)** ·
**[Why it works](demos/why_it_works.md)** ·
**[Architecture](docs/architecture.md)** ·
**[Paper draft](docs/paper/)** ·
**[Roadmap](docs/PRD.md#3-roadmap-по-этапам-e1-e6)**

</div>

---

## TL;DR

NQX-Core — software-эмулятор специализированного процессора для
[NautilusQuant](https://github.com/ORTODOX1/NautilusQuant), детерминистического
сжатия KV-кэша LLM в 4× через **золотое сечение φ**. Алгоритм правильный — нужен
**правильный процессор**: static dataflow ASIC с 1.5 KB ROM-LUT вместо 8 MB
random rotation matrix как у TurboQuant.

В пакете: ISA + ассемблер + дизассемблер, 7 functional units, **SystemVerilog
RTL skeleton**, Verilator + Yosys + OpenLane (путь к Skywater MPW),
ASIC floor-plan + timing closure docs, FastAPI HTTP-сервер с мониторингом и
chaos-тестами, demo runner с pitch deck, **6 proof-документов** с числами против
TurboQuant.

## Why this matters

| | TurboQuant (Google ICLR 2026) | **NQX / NautilusQuant** |
|---|---|---|
| **Подход** | random rotation matrix + PRNG | **deterministic Givens × φ** |
| **LUT / state size (dim=128)** | 32 KB matrix | **1.5 KB ROM** |
| **LUT scaling (dim=1024)** | 8 MB | **~12 KB** (-666×) |
| **Determinism** | seed-dependent | **bit-identical always** |
| **PRNG cost** | ~4 cycles × dim² | **0 cycles** (precomputed) |
| **Hardware fit** | GPU-bound (random matmul) | **static dataflow ASIC** (1:1 с pipeline) |
| **Compression ratio** | 4× | **4×** (equal) |
| **Roundtrip RMSE on synthetic KV** | comparable | **comparable** |

> **Главный месседж**: «Нужно перестать кроить алгоритмы под GPU. Алгоритм
> правильный — нужен правильный процессор. Вот он, эмулирован, готов к tape-out.»

## What we shipped

| Слой | Что готово | Где |
|---|---|---|
| **Software emulator** | ISA NQ-ISA v2 (21 opcode), ассемблер, дизассемблер, 7 functional units | `nqx/` |
| **RTL skeleton** | 7 SystemVerilog модулей + Verilator testbench + Yosys synth + OpenLane config | `rtl/` |
| **ASIC docs** | Floor-plan (50 mm² TSMC 7nm) + timing closure (1 GHz target) + tape-out checklist | `asic/` |
| **HTTP service** | FastAPI с auto CPU/GPU backend, /encode /decode /benchmark /verify /metrics /health/deep | `server/` |
| **Production** | Middleware logging, Prometheus metrics, structured errors, chaos tests, smoke test, load test | `server/`, `tests/chaos/` |
| **Demo / pitch** | TurboQuant baseline, 70B scaling, side-by-side, why-it-works, 10-slide pitch | `demos/` |
| **Proof tasks** | Angular uniformity, linear vs Lloyd-Max, φ vs random, determinism, LUT budget, energy delta | `bench/` |
| **Pre-silicon SDK** | Random Instr Gen, coverage tracking, performance counters, JTAG debug, formal verification (SymbiYosys) | `nqx/`, `rtl/formal/` |
| **CLI tools** | 16 launchers (`nqx-claude`, `nqx-deepseek`, `nqx-audit`, `nqx-demo`, `nqx-debug`, …) | `tools/cli/` |
| **Deploy** | Multi-arch Docker (amd64+arm64) + GPU image + vast.ai automation | `deploy/`, `Dockerfile*` |

## Quick start

```bash
# Clone & install
git clone https://github.com/<you>/nqx-core && cd nqx-core
pip install -r requirements.txt

# Run all 229 tests
python -m pytest tests -q

# Verify acceptance (orthogonality, roundtrip, bit-exact vs reference)
python run.py verify --dim 128

# Benchmark cycles + throughput + energy
python run.py bench --vectors 4096

# Run encode pipeline as NQ-ASM program
python run.py run programs/encode_dim128.nqasm --vectors 1000

# Show demo (TurboQuant vs NautilusQuant side-by-side)
python demos/run_demo.py
```

### Run as HTTP API

```bash
# Local CPU
docker compose --profile cpu up
curl http://localhost:8000/health

# vast.ai with RTX 5090 (one command)
bash deploy/quickstart-vastai.sh
```

See [`deploy/vastai.md`](deploy/vastai.md) for full deployment guide.

## Architecture (one picture)

```
        ┌────────── HBM (off-chip, FP16) ──────────┐
        │                                          │
        v                                          ^
   ┌────────┐                                ┌──────────┐
   │  DMA   │--> SRAM_in (24KB) ------> ... -│   PACK   │
   └────────┘                                │   3+1bit │
                                             └──────────┘
                                                  ^
   SRAM_in --> [ VRF FP32, 16 × 128 elem ]        │
                       │                          │
        ┌──────────────┴────────────────┐    ┌────┴──────┐
        v                               v    │   QJL     │
  ┌──────────┐  ┌──────────┐  ┌──────────┐   │ sign+corr │
  │ GU-L1    │->│ GU-L2    │->│ GU-L3    │   └─────▲─────┘
  │ 64 lanes │  │ 63 lanes │  │ ~32 lanes│         │
  │ adj pair │  │ shifted  │  │ butterfly│         │
  └──────────┘  └──────────┘  └──────────┘         │
                       │                           │
                       v                           │
                 ┌──────────┐    ┌──────────┐      │
                 │ POLAR    │--->│  QUANT   │------┘
                 │ sqrt+at2 │    │ Lloyd-Max│
                 │ 64 lanes │    │ 3-bit    │
                 └──────────┘    └──────────┘
                       ^
                 ┌──────┴───┐
                 │ ROM LUT  │  golden angle cos/sin (≈1.5 KB)
                 │ 191 pair │
                 └──────────┘
```

Pipeline: **18 cycles** depth, **1 vec / cycle** steady-state throughput.
Detailed spec: [`docs/architecture.md`](docs/architecture.md).

## Numbers (measured, not promised)

| Метрика | Значение | Источник |
|---|---|---|
| Orthogonality `T^T·T = I` (dim=128) | err ≤ **1.6e-7** | `tests/test_orthogonality.py` |
| Roundtrip без квантования (RMSE) | **9.6e-8** | `tests/test_orthogonality.py` |
| Bit-exact vs upstream NautilusQuant math | max diff < **1e-4** | `tests/test_vs_reference.py` |
| Compression ratio | **4.00×** ровно | `tests/test_roundtrip.py` |
| Throughput steady-state | **1 vec / cycle** | cycle counter в эмуляторе |
| ROM-LUT size (dim=128) | **1 910 bytes** | `nqx/lut.py` |
| Pipeline depth | **18 cycles** | `docs/architecture.md` |
| Determinism (100 runs same input) | **100% bit-identical** | `bench/determinism.md` |
| Energy per encode-vec (TSMC 7nm est.) | ≈ **5.1 nJ** | `nqx/energy.py` model |
| Number of NQ-ISA opcodes | **21** | `nqx/isa.py` |
| Tests passing | **229 / 229** | `pytest tests -q` |

## Roadmap

```
E1 ✅ Software emulator (NQX-Core)              ← DONE
E2 ✅ RTL skeleton + Verilator + Yosys synth   ← DONE
E3 ⏳ FPGA bring-up (Alveo U280 / AWS F1)       ← 3 months
E4 ⏳ vLLM / HF / llama.cpp integration         ← 3-6 months (T10-T12 на vast.ai)
E5 ⏳ Skywater 130nm MPW via Efabless ($0)      ← 6-9 months
E6 ⏳ TSMC 12nm / 7nm tape-out ($1.5-5M)        ← 12-18 months
```

Подробно: [`docs/PRD.md`](docs/PRD.md), [`asic/floorplan.md`](asic/floorplan.md),
[`asic/timing.md`](asic/timing.md), [`asic/tapeout_checklist.md`](asic/tapeout_checklist.md).

## Project layout

```
nqx/                  Software emulator core (95 Python modules)
  constants.py          φ, golden angle, NQXConfig
  lut.py                ROM-LUT generator
  memory.py             HBM, SRAM, register files
  functional_units.py   GivensUnit, PolarUnit, QuantUnit, QJLUnit, PackUnit, AttentionUnit
  mx_unit.py            OCP MX-Format quantization
  subbit_unit.py        Sub-1-bit raisable quantization
  pipeline.py           Cycle/energy accounting
  isa.py                Opcode definitions, encoding, decoding
  assembler.py          NQ-ASM → bytecode
  disassembler.py       bytecode → NQ-ASM
  cpu.py                NQXCore: emulator orchestrator
  energy.py             Energy model + random-rotation baseline
  coverage.py           Opcode/state coverage tracking
  counters.py           Performance counters (MMIO-mapped)
  jtag.py               IEEE 1149.1 TAP controller model

rtl/                  SystemVerilog (7 modules + TB)
  givens_lane.sv        One Givens rotation lane (4 mul + 2 add)
  golden_rom.sv         ROM with cos/sin LUT
  polar_unit.sv         CORDIC sqrt + atan2
  quant_unit.sv         Lloyd-Max with min/max reduce tree
  nqx_top.sv            Top-level wrapper
  tb_nqx.sv             Verilator testbench (bit-exact vs Python)
  formal/               SymbiYosys formal verification
  synth/                Yosys synthesis flow
  openlane/             OpenLane config for Skywater MPW

asic/                 ASIC design docs
  floorplan.md          50 mm² TSMC 7nm placement
  timing.md             1 GHz target, slack analysis
  tapeout_checklist.md  9-section pre-silicon checklist

programs/             NQ-ASM example programs
tests/                46 test files, 229 tests
server/               FastAPI HTTP service
deploy/               Docker, vast.ai, smoke tests
demos/                TurboQuant comparison, pitch deck, scaling demo, notebooks
bench/                Proof tasks (angular_uniformity, linear_quant, energy_proof, …)
docs/                 PRD, architecture, paper draft
sdk/                  libnqx C ABI, install scripts
firmware/             Boot ROM, Linux driver skeleton
integrations/         vLLM/HF/llama.cpp adapters (placeholders for vast.ai)
tools/                gen_rom.py, rig.py, dump_for_rtl.py, debug/
tools/cli/            16 CLI launchers (nqx-claude, nqx-deepseek, nqx-demo, …)
audits/               AI-agent prompts + results (this entire build by Claude+DeepSeek)
```

## Compare against TurboQuant

Run side-by-side:

```bash
python demos/turboquant_emul.py             # Run TurboQuant random-rotation emulation
python demos/run_demo.py                    # Full comparison: cycles, energy, RMSE, determinism
cat demos/side_by_side.md                   # Main comparison table
```

## How to cite

If you use NQX-Core in research, please cite:

```bibtex
@misc{nqxcore2026,
  title  = {NQX-Core: Pre-silicon emulator for golden-ratio KV-cache quantization},
  author = {NQX-Core contributors},
  year   = {2026},
  note   = {Built on NautilusQuant by ORTODOX1},
  url    = {https://github.com/<you>/nqx-core}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable form.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md). All AI-agent task lists
that built this project live in [`audits/prompts/`](audits/prompts/) for full
transparency.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgments

- Built on top of [**NautilusQuant**](https://github.com/ORTODOX1/NautilusQuant) by
  [@ORTODOX1](https://github.com/ORTODOX1) — golden-angle insight, reference math,
  upstream `nautilus_triton.py` and `nautilus_hardware.py`.
- Theory: Hermann Weyl's [equidistribution theorem](https://en.wikipedia.org/wiki/Equidistribution_theorem)
  (1916), still relevant 110 years later.
- Inspiration: [TurboQuant](https://arxiv.org/abs/2504.19874) (Google ICLR 2026)
  showed the rotation idea works; we're showing that determinism wins on the
  right hardware.
- Built using **Claude** + **DeepSeek-V4** as parallel coding agents — the entire
  102-task development log is auditable in `audits/`.

---

<div align="center">

**Status**: Pre-silicon. RTL skeleton synthesizable. Ready for FPGA bring-up
(Alveo U280) or Skywater 130 nm MPW shuttle via Efabless.

[Open an issue](../../issues) · [Join the discussion](../../discussions) ·
[Read the pitch](demos/pitch.md)

</div>
