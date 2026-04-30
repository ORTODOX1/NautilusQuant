# NQX Pre-Silicon SDK

SDK for developing and deploying NautilusQuant KV-cache quantization on NQX hardware. Includes assembler, disassembler, emulator, C ABI, firmware, documentation, and test infrastructure.

## Quick Install

```bash
bash sdk/install.sh
```

Add to `~/.bashrc`:
```bash
export PATH="$PATH:$HOME/.local/bin"
export NQX_SDK_DIR="$HOME/.local/share/nqx-sdk"
export PYTHONPATH="$PYTHONPATH:$NQX_SDK_DIR"
```

## Binary Reference

| Binary | Description |
|--------|-------------|
| `nqx-asm` | Assemble `.nqasm` → `.bin` bytecode |
| `nqx-disasm` | Disassemble `.bin` → `.nqasm` text |
| `nqx-sim` | Run `.nqasm` on emulator with cycle/energy reporting |
| `nqx-rig` | Random Instruction Generator — stress-test the ISA |
| `nqx-debug` | Step-by-step encode inspection |
| `nqx-debug-jtag` | JTAG debug interface — VRF/PC/CSR dump, single-step |
| `nqx-doctor` | System diagnostics — check deps, config, device |
| `nqx-status` | Project health — test counts, coverage, milestones |
| `nqx-stats` | Coverage statistics from RIG runs |
| `nqx-heavy` | Run heavy/deep audit tasks |
| `nqx-routine` | Run routine audit tasks |
| `nqx-claude` | Interactive Claude in the project |
| `nqx-deepseek` | Interactive DeepSeek |
| `nqx-flash` | Interactive DeepSeek Flash (cheap) |
| `nqx-codex` | Interactive Codex |
| `nqx-demo` | Start demo server |
| `nqx-audit` | Multi-model audit — prompts × CLIs in parallel |
| `nqx-trio` | tmux 3-pane: Claude + DeepSeek + Codex |
| `nqx-launch-all` | Launch all AI CLIs simultaneously |
| `nqx-sdk-env` | Source to set `NQX_SDK_DIR` and `PYTHONPATH` |

## Architecture

```
User code (C/Python)
    |
nqx_open → libnqx (C ABI, sdk/libnqx/)
    |
NQXCore (nqx/cpu.py) — software emulator
    |
    ├── GivensUnit  — golden-ratio rotation (3 layers)
    ├── PolarUnit   — (x,y) ↔ (r,θ) CORDIC
    ├── QuantUnit   — Lloyd-Max quantization
    ├── QJLUnit     — sign-bit correction
    ├── PackUnit    — 3+1 bit packing
    ├── SubBitUnit  — radius/angle split quantization
    ├── MXUnit      — OCP MX block quantization (MXFP4/6/8)
    └── HBM/SRAM    — memory hierarchy
```

## Hello World

**Step 1:** Write `hello.nqasm`
```nqasm
; Encode one vector from HBM address 0 and store packed result
ENC [0x0], [0x10000000], 1
HALT
```

**Step 2:** Assemble
```bash
nqx-asm hello.nqasm hello.bin
```

**Step 3:** Run on emulator
```bash
nqx-sim hello.nqasm
```
Output:
```
halted: True
cycles: 29
energy: 133.23 nJ
```

**Step 4:** Python API
```python
from nqx.assembler import assemble
from nqx.cpu import NQXCore
from nqx.constants import NQXConfig
import numpy as np

cfg = NQXConfig(dim=128, bits=3)
core = NQXCore(cfg)

# Load 16 FP16 vectors into HBM
vectors = np.random.randn(16, 128).astype(np.float32)
core.load_vectors_to_hbm(0, vectors)

# Assemble and run
with open("hello.nqasm") as f:
    prog = assemble(f.read())
result = core.execute_program(prog)

print(f"Halted: {result['halted']}")
print(f"Cycles: {core.cycles.total}")
print(f"Energy: {core.energy.total_nj():.2f} nJ")
```

## SDK Structure

```
sdk/
  README.md           — this file
  install.sh          — installer → ~/.local/share/nqx-sdk/
  libnqx/             — C ABI (libnqx.h + Python prototype)

nqx/                  — core emulator
  cpu.py              — NQXCore: instruction execution
  isa.py              — opcodes, instruction encode/decode
  assembler.py        — .nqasm → Instruction list
  disassembler.py     — bytecode → .nqasm text
  givens_unit.py      — Givens rotation layers
  polar_unit.py       — polar transform
  quant_unit.py       — Lloyd-Max quantizer
  qjl_unit.py         — sign-bit correction
  pack_unit.py        — 3+1 bit pack/unpack
  subbit_unit.py      — radius/angle split quant
  mx_unit.py          — OCP MX block quant
  lut.py              — golden-angle LUT
  memory.py           — HBM + SRAM + register files
  pipeline.py         — cycle counter
  energy.py           — energy model

tools/
  cli/                — CLI launchers (nqx-*)
  rig.py              — Random Instruction Generator
  debug/              — debug visualization tools

firmware/
  boot/               — boot ROM (boot.nqasm)
  driver/             — Linux PCIe driver skeleton

programs/             — example .nqasm programs
docs/                 — documentation (see below)
tests/                — pytest test suite
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](../docs/architecture.md) | ISA reference, microarchitecture, pipeline timing, address map |
| [Programming Guide](../docs/programming_guide.md) | Patterns, common mistakes, worked examples |
| [Errata](../docs/errata.md) | Known limitations, workarounds, revision history |
| [PRD](../docs/PRD.md) | Product requirements, roadmap, acceptance criteria |
| [Final Report](../docs/FINAL_REPORT.md) | Benchmark results, quality metrics |
| [Install Guide](../docs/INSTALL.md) | Full installation, Docker, GPU setup |

## Acceptance

| Check | Command |
|-------|---------|
| All tests pass | `python -m pytest tests -q` |
| Smoke test | `bash deploy/smoke.sh` |
| Benchmarks | `python run.py bench --vectors 4096` |
| Quality | `python run.py verify --dim 128` |
