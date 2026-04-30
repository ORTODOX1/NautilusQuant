# DEMO/PITCH задачи — для показа «когда / как / почему»

Этот раунд — переход от **proof-of-concept** к **демонстрации**. Цель — чтобы любой
человек (инвестор / партнёр / коллеги по AI) за 5 минут увидел side-by-side:
TurboQuant (random + GPU) vs NautilusQuant (φ + dataflow ASIC) — **на цифрах**.

## Перед стартом

1. Прочитай `docs/PRD.md`
2. Удостоверься что T17 (paper draft) и T20 (ablation) сделаны — оттуда тянем числа
3. `python -m pytest tests -q` должно зеленить

## Список задач (бери первый `[ ]`, делай, отмечай `[x]`)

- [x] **D1. TurboQuant baseline эмуляция.** `demos/turboquant_emul.py` — pure-numpy реализация
  TurboQuant подхода (без torch):
  - Random orthogonal rotation matrix (через `scipy.stats.ortho_group` или Householder QR)
  - Polar conversion (как у нас)
  - 3-bit polar quantize + 1-bit QJL (тот же финал что Nautilus)
  - Encode/decode симметрично
  - Метрики: RMSE roundtrip, simulated cycles (PRNG = 4 cycles/random + dim² muls), energy nJ
  Tests: `tests/test_turboquant_emul.py` — sanity checks.

- [x] **D2. End-to-end LLM attention demo.** `demos/llm_attention_demo.py` — эмуляция одного
  attention forward на NQX-Core (без torch):
  - synthetic Q, K, V: shape (n_heads=32, seq=2048, dim=128), generated via `np.random` с
    realistic outliers (как в `_make_data`)
  - compress K, V через `NQXCore.encode`
  - simulated decompress при attention scoring (используй `AttentionUnit.dot_polar` если умеет
    напрямую; иначе decode→inverse rotate→dot)
  - метрики: cycles per token, HBM bytes saved (4×), RMSE на финальный attention output
  Артефакт: `demos/llm_attention_demo.md` — таблица.

- [x] **D3. 70B-модель projection.** `demos/scaling_demo.py` — числовая прогноз-эмуляция,
  без реального запуска модели:
  - Llama-3-70B params: 80 layers × 64 heads × dim=128, ctx=128K
  - FP16 KV-cache total = 80 × 64 × 128 × 128K × 2 (K+V) × 2 bytes = ?
  - NQX-compressed = `?` / 4
  - Сколько H100 нужно сейчас (по 80 GB HBM) vs сколько NQX-class ASIC (по 100 MB SRAM)
  - $$ savings: spot price H100 vs прогноз ASIC TCO
  - Output: `demos/scaling_demo.md` — таблица для CFO.

- [x] **D4. Side-by-side comparison — главная таблица.** `demos/side_by_side.md`:
  Запусти D1 (turbo) и существующий `NQXCore.encode` на ОДНИХ И ТЕХ ЖЕ 4096 vec dim=128.
  Метрики:
  | Метрика | TurboQuant | NautilusQuant | Δ / комментарий |
  | RMSE roundtrip | ? | ? | ? |
  | Cycles per encode | ? | ? | ? |
  | Energy nJ/vec | ? | ? | ? |
  | LUT / PRNG state size | ? | 1.5 KB | ? |
  | Determinism (100 runs bit-identical fraction) | ? | 100% | ? |
  | Compression ratio | 4× | 4× | равно |
  Это **одна таблица которая решает** «чем мы лучше». Должна быть вверху pitch.

- [x] **D5. Why-it-works narrative.** `demos/why_it_works.md` — простыми словами + математика:
  - **WHY работает**: golden angle = most-irrational → angular discrepancy O(1/N) → uniform
    distribution after rotation → linear quant работает (не нужен Lloyd-Max таблиц).
    Численно сослаться на T21 (angular uniformity) и T22 (linear vs Lloyd-Max).
  - **HOW работает**: visual data flow через 5 stages (encode) и 4 stages (decode).
    Каждый stage = 1 такт NQX-Core (T8 RTL skeleton).
  - **WHEN profitable**: при ctx ≥ 4K и dim ≥ 64, выигрыш растёт квадратично с ctx.
  - **WHO target**: LLM inference operators (vLLM, llama.cpp), edge AI (Jetson, M-series),
    on-device LLM (phone, IoT с PLC + 512-byte LUT).
  - **WHAT NEXT**: эмулятор → FPGA U280 → ASIC TSMC 12nm/7nm.
  Каждый раздел — числами из proof tasks (T21-T26), не «вода».

- [x] **D6. Pitch deck (markdown).** `demos/pitch.md` — 10 слайдов (по странице каждый):
  1. **Problem**: KV cache eats 80% HBM memory at long context
  2. **State of art**: TurboQuant (Google ICLR 2026) — random rotation, не reproducible, GPU-bound
  3. **Insight**: golden angle is provably most uniform (Weyl equidistribution, 1916)
  4. **Solution**: NautilusQuant — deterministic Givens × φ
  5. **Hardware fit**: pipeline 1:1 mapping на static dataflow ASIC
  6. **Numbers**: 4× compression, 0 PRNG cycles, 1.5KB LUT vs 8MB random matrix (cite D4)
  7. **Architecture**: NQX-Core diagram (ASCII или mermaid)
  8. **Roadmap**: эмулятор (DONE) → FPGA U280 (3 months) → ASIC TSMC 12nm (12 months)
  9. **Money**: FAB cost ($1.5M tape-out 12nm) vs ROI per chip (per 1B token cost)
  10. **Ask**: что нужно — $X на FPGA dev, $Y на tape-out, $Z на team
  Стиль — Y Combinator pitch, минимум воды, максимум цифр.

- [x] **D7. Demo runner.** `demos/run_demo.py` — single-command показ:
  ```
  python demos/run_demo.py
  ```
  Запускает D1 + D2 + D3 + D4 в правильном порядке. Печатает таблицы с ANSI-цветом
  (`\033[32m` для NQX-победы, `\033[31m` для TurboQuant-проигрыша). Создаёт собранный отчёт
  `demos/results-<timestamp>.md`.

- [x] **D8. ASCII visualization.** `demos/viz.py` — без matplotlib (pure stdlib):
  - **Latency jitter histogram**: 100 runs encode, latency в ms, ASCII bar chart.
    TurboQuant покажет разброс, NQX — единая полоса.
  - **Cycle breakdown by stage**: side-by-side bar chart pipeline (load / rotate / polar /
    quant / qjl / pack / store) — clocks per stage.
  - **Timeline pipeline**: ASCII Gantt chart показывает overlap stages.
  Печатается в терминал, fallback markdown в `demos/viz.md`.

- [x] **D9. CLI launcher для demo.** `tools/cli/nqx-demo` — bash скрипт:
  ```bash
  cd /home/user/nautilusquantcore
  python demos/run_demo.py
  echo
  echo "Open demos/pitch.md, demos/why_it_works.md, demos/side_by_side.md to read"
  ```
  Линковать через `tools/cli/install.sh`.

## Acceptance criteria (что значит «демо готово»)

| Критерий | Где |
|---|---|
| `python demos/run_demo.py` отрабатывает за < 30 сек | D7 |
| `demos/side_by_side.md` имеет ≥ 5 численных метрик с конкретными числами | D4 |
| `demos/pitch.md` ≤ 10 страниц, каждый слайд stand-alone | D6 |
| Любой человек за 5 минут понимает «что такое NautilusQuant и зачем процессор» | D5 + D6 |
| `pytest tests -q` всё ещё 131+ PASS | sanity |

## Workflow

1. Один `[ ]` за раз
2. Если стопперится на отсутствии данных — генерируй synthetic, пометь в выводе «synthetic, real KV pending vast.ai»
3. Бит-точность с upstream NautilusQuant — обязательна для encode-side
4. После D1-D9 — финальный pytest, никаких regressions
