# Yosys synthesis of NQX-Core

Open-source synthesis flow for the SystemVerilog skeleton in `rtl/`.
Targets a generic technology so you can run it without a PDK; switch to
sky130 or any other ASAP7-class library by overriding the `abc` step.

## Build

```bash
cd rtl/synth
make synth
```

Requires Yosys ≥ 0.39 (`yosys --version`). On most distros:

```bash
sudo dnf install yosys              # Fedora / RHEL
sudo apt install yosys              # Debian / Ubuntu
brew install yosys                  # macOS
```

`make synth` produces:

| File | Role |
|---|---|
| `synth.log`           | Full Yosys log including stat report |
| `nqx_top_synth.v`     | Flattened gate-level Verilog netlist |
| `nqx_top_synth.json`  | JSON netlist (for downstream OpenLane / ABC) |

## Reading the report

The end of `synth.log` contains a `Number of cells` block. Sample
expected ranges for the current skeleton (placeholders for GU / PU
math):

| Top-level item | Order of magnitude |
|---|---:|
| Wires (`Number of wires`) | a few thousand |
| Wire bits (`Number of wire bits`) | tens of thousands |
| Cells (`Number of cells`) | a few hundred (skeleton; will grow ~50× once GU/PU are filled) |

Use the cell count as a smoke gate, not a final number — the real
gate count after retiming and CORDIC fill-in lives in
`asic/floorplan.md` (≈ 6.7 M gates total at dim=128).

## Targeting a real PDK

To map onto sky130 standard cells:

```tcl
read_liberty -lib /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
synth -top nqx_top
dfflibmap -liberty /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
abc -liberty /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
stat -liberty /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
```

For the Caravel-style OpenLane2 flow that wraps Yosys with
floorplanning + routing, see `rtl/openlane/README.md`.
