# OpenLane2 / Skywater 130 nm flow for NQX-Core

This directory drives the open-source RTL → GDSII flow on Skywater
sky130A (Google × SkyWater). The endgame is a free shuttle slot in the
**Efabless Open MPW Program** — $0 if your design is open-source under
Apache 2.0 / OSI-approved, ≈ $10 K for commercial.

## Quick start

```bash
# 1. install OpenLane2 (Docker is the simplest)
pip install openlane

# 2. point it at our config
openlane --pdk sky130A rtl/openlane/config.json

# 3. once it finishes, the run lives under
ls runs/RUN_<timestamp>/
ls runs/RUN_<timestamp>/final/gds/    # → nqx_top.gds
```

For an interactive flow (recommended on first run):

```bash
openlane --pdk sky130A --last-run rtl/openlane/config.json --interactive
```

You can stop after `floorplan` to sanity-check area and abort before
the long routing step:

```bash
openlane rtl/openlane/config.json -from synthesis -to floorplan
```

## What the config sets

| Key | Value | Why |
|---|---|---|
| `DESIGN_NAME`         | `nqx_top` | Top-level wrapper in `rtl/nqx_top.sv` |
| `CLOCK_PERIOD`        | 10 ns (100 MHz) | Conservative target for sky130 standard cells; bump to 20 ns if STA fails |
| `FP_CORE_UTIL`        | 50 | Half the die filled with logic, the other half left for routing |
| `PL_TARGET_DENSITY`   | 0.55 | Slightly higher than util to keep buffers reachable |
| `DIE_AREA`            | 1.5 × 1.5 mm | Below Caravel user-project area (2.92 × 3.52 mm) |
| `PDK` / `STD_CELL_LIBRARY` | sky130A / sky130_fd_sc_hd | Free, open, MPW-eligible |
| `RUN_HEURISTIC_DIODE_INSERTION` | 1 | Adds antenna diodes during routing |
| `GRT_REPAIR_ANTENNAS` | 1 | OpenROAD post-route fix |

## What `runs/<run>/` contains after a full pass

| Path | Role |
|---|---|
| `final/gds/nqx_top.gds` | GDSII layout — what the foundry ingests |
| `final/lef/nqx_top.lef` | Abstract for hierarchical reuse |
| `final/def/nqx_top.def` | Placed/routed DEF |
| `reports/synthesis/`    | Yosys synthesis stats |
| `reports/floorplan/`    | Util / cell density |
| `reports/sta/`          | Timing reports (setup/hold), per corner |
| `reports/drc/`          | Magic / KLayout DRC |
| `reports/lvs/`          | LVS (must be clean for tape-out) |

## Submitting to Efabless Open MPW

1. Fork `efabless/caravel_user_project`.
2. Replace `verilog/rtl/user_project_wrapper.v` to instantiate
   `nqx_top` (treat NQX as the user-project block).
3. Update `openlane/user_project_wrapper/config.json` with the values
   from this file (paths must be relative to the Caravel checkout).
4. Run the project's harness flow:
   ```bash
   make user_project_wrapper
   ```
5. When `final/` is clean of DRC/LVS, open a PR against the next MPW
   shuttle window listed at <https://efabless.com/open-shuttle-program>.
6. Tape-out approval review takes ~2 weeks; chips are silicon-back ~6
   months later.

## Caveats

- Sky130 is a **130 nm** process. Cycle times are 5–10 × slower than
  the TSMC 7 nm target in `asic/floorplan.md`. Treat sky130 as the
  *correctness silicon* and the commercial node as the *throughput
  silicon*.
- The `polar_unit.sv` in this skeleton uses placeholder math — Open
  MPW samples are fine, but production tape-out needs the CORDIC
  iteration network filled in (tracked in `asic/timing.md` open issues).
- Yosys + OpenROAD do not currently support `$readmemh` initialisation
  for hard SRAM macros. The `golden_rom.mem` is loaded post-fabrication
  via the JTAG path (see `nqx/jtag.py` in SDK8).
