# NQX-Core Tape-out Checklist (TSMC N7, 50 mm² die)

Pre-tape-out gating items. Each section lists *what to check* and the
*pass/fail criterion*. Fail in any item ⇒ no tape-out submission until
remedied. Sign-off owners (DFT, PV, STA, …) initial each row in the
official spreadsheet; this file is the canonical specification.

## 1. DFT — Design For Test

| Item | What to check | Pass criterion |
|------|---------------|----------------|
| Scan chains | All flops on a scan chain; ≤ 8 chains, ≤ 2000 flops/chain | scan-shift coverage ≥ 99.0% (Tessent) |
| ATPG fault coverage | Stuck-at + transition-delay | stuck-at ≥ 99.5%, TDF ≥ 95% |
| BIST for SRAM (24 KB ×2 + VRF 8 KB + ROM 4 KB) | March-C+ for read/write SRAMs, RowHammer-aware march | 100% bit coverage, BIST runtime ≤ 1 ms |
| Boundary scan | JTAG IEEE-1149.1 chain over all top-level pads | continuity, BYPASS, IDCODE, EXTEST verified in sim |
| Test mode signals | TM, SE, scan_en routed clean (no clock-glitch) | TM/SE always-low in functional mode (LEC equivalence) |
| Compression | EDT 64:1 or higher | pattern count ≤ 50 K patterns at full coverage |

## 2. IO Ring

| Item | Check | Pass |
|------|-------|------|
| Pad ring closure | Continuous power/ground rings around the die | DRC pad-ring rule clean |
| HBM2e PHY pads | 1024-bit DQ + ECC + CK + control routed to top edge | per-channel skew ≤ 30 ps |
| PCIe Gen5 SerDes | x8 lanes paired (Tx_p/n, Rx_p/n) | impedance 85 Ω diff, lane-to-lane skew ≤ 5 ps |
| Side-band IO (JTAG, UART debug, SMBus) | Slow CMOS pads | ESD CDM 250 V, HBM 2 kV |
| Pad sharing rules | No analog pad shares power island with HBM | confirmed in floor-plan LEF |

## 3. ESD Protection

| Item | Check | Pass |
|------|-------|------|
| HBM model (Human Body) | All pads ≥ 2 kV | TLP curve linear, no snap-back below 2 kV |
| CDM (Charged Device) | All pads ≥ 250 V | per-pad CDM diode ratio ≥ 1.0 |
| MM (Machine Model) | optional but recommended | ≥ 200 V |
| Power-clamp rail-to-rail | One clamp per power island | Idsat ≥ 100 mA, Vt1 ≤ 1.2 × Vdd_max |
| ESD CDM hot-spots | Long-running clock nets and SerDes Tx | per-net Crating clean in PERC deck |

## 4. Package Selection

| Item | Check | Pass |
|------|-------|------|
| Body size | Die 7.07 × 7.07 mm + HBM stack(s) | FCBGA 23 × 23 mm fits with HBM2e on interposer |
| Substrate layers | Routing for 1024-bit HBM bus | ≥ 12 substrate layers, impedance controlled |
| Solder bumps | C4 bumps on 130 µm pitch | bump count ≥ 1500, current/bump ≤ 50 mA |
| Thermal | TDP ≈ 12 W | Theta-JC ≤ 0.5 °C/W with heatsink |
| Mechanical | Warpage in JEDEC reflow | per-corner warpage ≤ 50 µm @ 260 °C |

## 5. Reticle / Die Size Limits

| Item | Check | Pass |
|------|-------|------|
| Maximum die area | TSMC N7 reticle 26 × 33 mm | die 7.07 × 7.07 = 49.98 mm² ≪ reticle |
| Wafer dicing | Edge keep-out + scribe lanes | scribe ≥ 80 µm; die placed within wafer mask |
| Multi-project wafer (MPW) compatibility | If shuttle is used | block fits MPW slot, no exotic mask layers |
| Photomask cost (15 layers) | Mask set count | ≤ 15 critical mask layers (N7 standard) |

## 6. Multi-corner Timing Sign-off

Setup + hold across PVT × on-chip variation (OCV).

| Corner | Voltage | Temp | Library | Setup target | Hold target |
|--------|---------|------|---------|--------------|-------------|
| `ssg_0p65v_125c` (slow-slow) | 0.65 V | 125 °C | tt0p75v25c | TNS = 0  | WNS ≥ 0 |
| `ffg_0p85v_m40c` (fast-fast)  | 0.85 V | -40 °C | ff0p85v-40c | hold WNS ≥ 0 | TNS = 0 |
| `tt_0p75v_85c` (typical)     | 0.75 V | 85 °C | tt0p75v85c | sign-off ref | sign-off ref |
| `ssg_0p7v_125c` (LP island)  | 0.70 V | 125 °C | ssg0p70v125c | TNS = 0 | WNS ≥ 0 |

Pass: TNS ≤ 0 ps in all corners with OCV margins applied; aging derating
≤ 2% per critical path; clock-skew sign-off across H-tree leaves ≤ 30 ps.

## 7. IR Drop

| Item | Check | Pass |
|------|-------|------|
| Static IR drop (vector-less) | Per power island | ≤ 5% of Vdd at any region |
| Dynamic IR drop (transient) | Worst-case attention burst (1024 batch) | ≤ 10% of Vdd at any flop |
| Decap capacity | On-die + package | Cd / Cload ≥ 5× |
| Power TSV (HBM interposer) | For HBM pin currents | ≤ 1 mA per via |
| Hot spot map | Thermal-aware floor-plan rerun | max ΔT ≤ 25 °C across compute island |

## 8. Electromigration (EM)

| Item | Check | Pass |
|------|-------|------|
| Power-grid wires | Average current density | ≤ TSMC N7 Jmax (per-layer) at 105 °C, 10-yr life |
| Signal wires (clock, HBM bus) | Peak current + RMS | within library `iavg`, `irms`, `ipeak` |
| Vias on power straps | EM-aware via array | ≥ 4× redundancy on critical via stacks |
| MTBF report | Calibre PERC EM deck | sign-off pass with no waivers |

## 9. Formal Verification (LEC)

| Item | Check | Pass |
|------|-------|------|
| RTL ↔ Synthesised netlist | Synopsys Formality / Cadence Conformal | LEC clean, 0 unmapped, 0 not-equivalent |
| Synth ↔ Post-CTS netlist | Account for ICG / clock buffers | LEC clean |
| Post-CTS ↔ Post-Route | Final | LEC clean; any ECO logged |
| Constant-propagation waivers | List of justified differences | reviewed and signed by RTL lead |
| Property checking (SVA) | Key invariants in `nqx_top.sv` (e.g. "BARRIER drains pending DMA") | bounded model check 30+ cycles, no CEX |

## Sign-off ledger

Each section has a sign-off line in the project tracker
(`linear:NQX-TAPEOUT`). Status flips from yellow (in flight) to green
(pass) only after the assigned owner attaches the corresponding tool log
(Calibre, Tessent, Tempus, Voltus, Conformal). No tape-out without all 9
sections green.

## Out-of-scope, deferred to NQX-Core v2

- Multi-die / chiplet integration via UCIe.
- 3D-stacked SRAM via TSV (current SRAM is on-die).
- Side-channel attack hardening (DPA / EM probes).
