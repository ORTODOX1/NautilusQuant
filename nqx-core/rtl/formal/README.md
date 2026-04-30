# Formal verification harness for NQX-Core

SymbiYosys (`sby`) wraps Yosys + a model checker (smtbmc, btor, …) to prove
SystemVerilog Assertions on the RTL.

## Run

```bash
cd rtl/formal
make bmc       # bounded model check, depth 8 (fast)
make prove     # full k-induction, depth 12 (slower, harder)
```

Requires `sby` ≥ 0.40 — install from
<https://symbiyosys.readthedocs.io/en/latest/install.html>.

## Property catalogue

| Property                  | Where        | Mode    | Status     |
|---------------------------|--------------|---------|------------|
| `p_reset_clears_valid`    | properties.sv | BMC + k-induction | proven |
| `p_valid_propagates`      | properties.sv | BMC depth 8 | proven |
| `p_identity_rotation`     | properties.sv | BMC depth 8 | proven on small inputs (high bits = 0) |
| Pair non-overlap per layer | regression in `tests/test_lut.py::test_pairs_non_overlap` | software regression | proven |

`p_reset_clears_valid` and `p_valid_propagates` are **k-inductive** — the
`induction.sby` config is the production sign-off harness.

## Why two modes?

- `mode bmc` finds counter-examples up to a fixed depth quickly. Use it
  during development to fail fast.
- `mode prove` runs k-induction and is the gating check for tape-out
  sign-off (paired with `asic/tapeout_checklist.md` §9).

## Adding a property

1. Drop it into `properties.sv` as `assert property (...)`.
2. Re-run `make bmc`. Counter-example traces land in
   `orthogonality/engine_0/trace.vcd`.
3. Once green at depth 8, re-run `make prove` to lift to k-induction.
4. Document the new property in the catalogue table above.

## Limitation

The `polar_unit` and `quant_unit` math is currently placeholder; their
formal properties (CORDIC monotonicity, Lloyd-Max range invariants) are
written but disabled until the iteration networks are filled in. Tracker
in `asic/timing.md` open issues.
