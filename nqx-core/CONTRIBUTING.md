# Contributing to NQX-Core

Thanks for considering a contribution. NQX-Core is open to PRs for emulator
features, RTL improvements, integration adapters, documentation, and benchmarks.

## Quick rules

1. **English only** in code, comments, commit messages, branch names.
2. **No new docstrings or inline comments** unless the code clearly demands
   them. Keep code self-explanatory through naming.
3. **All tests must pass** before any PR: `python -m pytest tests -q`.
4. **Bit-exact** compatibility with upstream NautilusQuant math is mandatory
   for the encode pipeline.
5. **No speculative error handling**, no "just in case" code, no refactoring
   beyond the scope of the change.

## How to contribute

### 1. Find or open an issue

Look at [`audits/prompts/`](audits/prompts/) for the original task lists
(`heavy.md`, `routine.md`, `demo.md`, `scenarios.md`, `sdk.md`, `heavy-gpu.md`).
Items still marked `[ ]` are open work.

For new ideas, open an issue first to discuss the design.

### 2. Branch & commit

```bash
git checkout -b feat/short-name           # for new features
git checkout -b fix/short-name            # for bugfixes
git checkout -b rtl/short-name            # for RTL changes
git checkout -b chore/short-name          # for maintenance
```

Commit messages: imperative mood, English, one logical change per commit.

```
Add MXFP4 backend with bit-exact upstream parity

- nqx/mx_unit.py: new MXQuantizer with 32-element block, 8-bit shared exp
- nqx/isa.py: opcodes 0x52 MXPACK, 0x53 MXUNPACK
- nqx/assembler.py: parsing for MXPACK/MXUNPACK
- tests/test_mx_unit.py: edge cases for FP4/FP6/FP8/INT8 formats
```

### 3. Test locally

```bash
python -m pytest tests -q                 # all tests
python run.py verify --dim 128            # acceptance
python run.py bench --vectors 4096        # performance check
ruff check nqx/ server/ tools/            # style
black --check nqx/ server/ tools/         # formatting
```

### 4. Open the PR

Fill in the PR template. Include:
- What changed (one paragraph)
- How to verify (command + expected output)
- Any acceptance criteria affected

## Code style

- Python 3.11+, type hints on public APIs
- Black with `--line-length 100`
- Ruff for linting (config in `pyproject.toml`)
- NumPy 2.x; PyTorch / Triton are optional deps (only in `server/backends.py`,
  `nautilus_triton.py`, `integrations/`)
- SystemVerilog: 2-space indent, lowercase signal names, `_n` suffix for
  active-low resets

## What to NOT do

- **Don't add features** that aren't in an existing task or open issue.
- **Don't refactor** beyond the scope of your change.
- **Don't add docstrings** to existing code unless the function is genuinely
  non-obvious.
- **Don't introduce dependencies** without discussion (NumPy + FastAPI +
  Pydantic + Torch+Triton are the only allowed runtime deps).
- **Don't break bit-exact reference math** — if your change moves the encode
  output even by 1e-5 from upstream NautilusQuant, the tests will fail and
  the PR will be rejected unless explicitly justified.

## RTL contributions

For SystemVerilog changes:
- Run Verilator on `tb_nqx.sv` and confirm bit-exact match against Python
  golden output (use `tools/dump_for_rtl.py`)
- For larger designs, run SymbiYosys formal verification
  (`make formal` in `rtl/formal/`)
- Update `asic/timing.md` if your change affects critical paths

## Reporting issues

Use the templates in `.github/ISSUE_TEMPLATE/`:
- **Bug report**: include reproducer (Python snippet + expected vs actual)
- **Feature request**: scope, motivation, proposed interface

## License

By contributing, you agree your work is released under the MIT License (see
[`LICENSE`](LICENSE)).
