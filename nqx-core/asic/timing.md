# NQX-Core Timing Closure (TSMC N7, 1 GHz target)

Целевая частота `f_clk = 1.0 GHz` ⇒ доступный clock period `T = 1.00 ns`.
Для setup мы оставляем 100 ps под clock skew/jitter ⇒ `t_max(comb) ≤ 0.90 ns`.

## 1. Critical paths (top-5)

| # | Path                                                                | Length | Slack | Status |
|---|---------------------------------------------------------------------|-------:|------:|--------|
| 1 | `GU.lane.in_a → mul0 → sub → out_a_reg`                             |  920 ps | -20 ps | **fail** — нужно retime |
| 2 | `PU.CORDIC stage 0 → stage 1 → stage 2`                             |  870 ps | +30 ps | borderline |
| 3 | `QU.minmax_tree[d=128] → range_calc → q_round`                      |  950 ps | -50 ps | **fail** — добавить регистр на середине дерева |
| 4 | `ROM_LUT.broadcast → GU.lane[63].cos`                               |  720 ps | +180 ps | OK |
| 5 | `HBM_PHY.rx_data → SRAM_in.write_data`                              |  860 ps | +40 ps | OK (HBM PHY domain crossing) |

## 2. План retiming / pipelining

### Path #1 — GU lane FP32 multiply-add

Сейчас: одно такт от регистра `in_a/in_b` до `out_a/out_b`. Включает:
`mul (FP32, 380 ps) + sub (FP32 add, 320 ps) + setup 220 ps`.

**Решение:** разбить на **две стадии** — `mul → reg → add → reg`:
- Pipeline depth GU.Lk возрастает с 1 → 2 такта.
- Steady-state throughput не падает (1 vec/такт после prefill).
- Total pipeline depth для encode: 18 → 21 такт. Acceptable.

В RTL это внутренний регистр `s1_a, s1_b, s1_cos, s1_sin` уже добавлен в
`rtl/givens_lane.sv` (см. always_ff на стадии s1). Synthesis это retime.

### Path #3 — Lloyd-Max min/max reduction tree

Дерево 7-уровневое для dim=128 (`log2(128) = 7`). Перепад mux + compare на каждом
уровне ≈ 130 ps ⇒ `7 × 130 = 910 ps` без регистров.

**Решение:** balance tree → `4 + 3` уровня с регистром после уровня 4.
Pipeline depth QU.r: 1 → 2 такта. `cycles_quant_minmax` в Python модели
уже = 7 (битовая reduction depth), для cycle-accurate матчинга поднять до 8.

### Path #2 — CORDIC stage chain

Marginal. Решается переходом на **redundant signed-digit CORDIC** в R1 итерации,
но в скоупе E5 floor-plan'а — оставляем 4-стадийный pipeline + 8-стадийный
atan2, как уже задано в `rtl/polar_unit.sv`.

## 3. Hold timing

Все critical paths имеют запас ≥ 80 ps на минимальный пробег (min-cell delay в
N7 ≈ 30–40 ps на ячейку). Hold-violations не ожидаются на скан-цепях после
обычного buffering pass'а.

## 4. Clock tree

- Single H-tree от center die. Insertion delay ≤ 250 ps; Skew ≤ 30 ps.
- `vdd_compute` и `vdd_sram` островки — независимые leaf-clocks с программируемой
  фазой (≤ ±50 ps adjustment).
- HBM2e PHY имеет свой PLL @ 600 MHz; FIFO domain crossing у SRAM_in/out.

## 5. CDC (Clock Domain Crossing)

| Source domain   | Sink domain    | Тип      | Решение |
|-----------------|----------------|----------|---------|
| HBM PHY 600 MHz | core 1 GHz     | data + valid | gray-code FIFO depth=8 |
| PCIe 250 MHz    | core 1 GHz     | CSR ack/req | 2-flop synchronizer |
| JTAG ~10 MHz    | core 1 GHz     | scan     | clock mux на boundary scan |

## 6. Static timing margins (после retiming)

| Path | New slack | Comment |
|------|----------:|---------|
| GU.lane (split into mul/add) | +120 ps | safe |
| QU.minmax (4+3 split)        | +90 ps  | safe |
| PU CORDIC chain              | +30 ps  | borderline; reroute LUT bus to remove 20 ps |
| ROM broadcast                | +180 ps | already OK |
| HBM PHY domain crossing      | +40 ps  | OK |

## 7. Open issues to track

- `cycles_quant_minmax` в `nqx/constants.py` нужно поднять с 7 → 8 после
  утверждения retiming-плана пути #3, иначе Python pipeline counter будет
  расходиться с RTL.
- При раздельном `vdd_compute` / `vdd_sram` — добавить level-shifter cells на
  границе VRF read ports → GU/PU/QU. Площадь: ~0.05 mm² (учтена в "Routing &
  spare" в `asic/floorplan.md`).
- DVFS turbo @ 1.2 GHz требует rerun setup на путях #1 и #3 после retiming;
  возможно понадобится дополнительный регистр в QU range stage.
