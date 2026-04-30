# NQX-Core RTL skeleton

SystemVerilog backbone for the encode pipeline.
This directory mirrors the Python emulator's functional units (`nqx/`).

## Files

| File | Role |
|---|---|
| `givens_lane.sv` | One Givens rotation lane (4 mul + 2 add, FP32 placeholder) |
| `golden_rom.sv`  | ROM, init from `$readmemh("golden_rom.mem")` (cos/sin/pair indices) |
| `polar_unit.sv`  | sqrt + atan2 CORDIC stub, 4-stage pipelined |
| `quant_unit.sv`  | Lloyd-Max quant w/ min/max reduction tree |
| `nqx_top.sv`     | Top-level wrapper (LDV→GU.L1..L3→PU→QU) |
| `tb_nqx.sv`      | Verilator testbench: reads `python_dump.hex`, asserts no mismatches |
| `Makefile`       | `make sim` runs Verilator and the testbench |

## Build

```bash
make rom            # generate golden_rom.mem (calls tools/gen_rom.py)
make dump           # generate python_dump.hex from a Python-emulated rotation
make sim            # build + run Verilator + tb_nqx
```

Requires Verilator ≥ 5.x. Without Verilator the makefile falls through cleanly;
the regenerated mem files are still useful for downstream Vivado synthesis.

## Status

This is the E2 milestone scaffold. The pipeline is **structurally** wired but
the math is currently pass-through — Givens lanes, polar CORDIC and Lloyd-Max
ranges still need their iteration networks. The upcoming RTL work fills these
in lane-by-lane while keeping the testbench bit-exact against `nqx/cpu.py`.
