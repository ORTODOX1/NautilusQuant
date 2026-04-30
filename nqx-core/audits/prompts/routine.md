# Routine tasks — для DeepSeek-flash (рутина, мелкие фиксы, тесты, доки)

Ты — мидл-инженер на проекте NQX-Core. Делаешь **мелкие точечные** изменения, без архитектурных решений. Прочитай `docs/PRD.md` и `CLAUDE.md`, потом бери первый незакрытый `[ ]` отсюда. Молоти быстро, без воды.

## Перед стартом

1. Прочитай `docs/PRD.md` (особенно §7 «Правила работы»).
2. `python -m pytest tests -q` — все должно зеленить. Если красное — НЕ ТВОЯ ЗАДАЧА, останавливайся и сообщай мне.

## Список задач

### Тесты (повышаем покрытие)

- [x] **R1.** `tests/test_givens_unit.py` — отдельные тесты только для `GivensUnit`: forward + inverse каждого слоя, на синтетике.
- [x] **R2.** `tests/test_polar_unit.py` — `PolarUnit.to_polar` + `from_polar` roundtrip < 1e-5.
- [x] **R3.** `tests/test_quant_unit.py` — Lloyd-Max на edge cases (constant input, single outlier, all-zeros).
- [x] **R4.** `tests/test_qjl_unit.py` — sign correction, alpha boundaries (0, 0.5, 1).
- [x] **R5.** `tests/test_pack_unit.py` — exhaustive pack/unpack на всех `q ∈ [0..7] × sign ∈ {0,1}` × всех bit positions in byte.
- [x] **R6.** `tests/test_memory.py` — HBM lazy paging, граничные адреса, oob чтения/записи.
- [x] **R7.** `tests/test_assembler.py` — все mnemonic encode/decode roundtrip, ошибки ассемблера на bad input.

### Документация / комментарии (только если ясно нужно)

- [x] **R8.** Обнови `README.md` — секция `Что добавилось` с актуальным списком launchers, audits/, server/, deploy/. Не вода — таблица.
- [x] **R9.** `docs/benchmarks.md` — запусти `python run.py bench --vectors N --dim D` для каждой комбинации:
  - dim ∈ {32, 64, 128, 256}
  - vectors ∈ {256, 1024, 4096}
  Сложи результаты в markdown-таблицу: cycles, throughput, energy/vec, RMSE.

### Linting / стиль

- [x] **R10.** Запусти `ruff check nqx/ server/ tests/ tools/` — почини warnings (но не меняй логику и не добавляй docstrings).
- [x] **R11.** Запусти `black --check nqx/ server/ tests/ tools/` — отформатируй где разошлось, line-length 100.
- [x] **R12.** `pyflakes nqx/` — почини неиспользуемые импорты.

### Скрипты / удобство

- [x] **R13.** `tools/cli/nqx-status` — bash-скрипт, выводит:
  - сколько `[x]` vs `[ ]` в `audits/prompts/heavy.md`
  - то же для `audits/prompts/routine.md`
  - последние 5 файлов в `audits/results/`
  - результат `pytest tests -q --tb=no`.
- [x] **R14.** `tools/cli/nqx-stats` — парсит `audits/logs/` и вытаскивает примерное количество токенов / вызовов на CLI. Если в `~/.claude/mcp-servers/multi-ai/.calls.jsonl` есть лог — оттуда.
- [x] **R15.** `tools/clean.sh` — удаляет `audits/results/`, `audits/logs/`, `__pycache__`, `.pytest_cache`. Спрашивает подтверждение `[y/N]`.

### Bench, dataset

- [x] **R16.** `bench/synth_kv_data.py` — генератор синтетического KV-cache (FP16, реалистичные outliers). Параметры: n_layers, n_heads, dim, seq_len. Дамп в `.npy`.
- [x] **R17.** `bench/run_all.py` — прогоняет все эмулятор-конфиги (CPU backend) на синтетике и пишет в `audits/results/bench-<timestamp>.md`.

### Лёгкие фиксы

- [x] **R18.** В `nqx/memory.py` HBM lazy `read_bytes` — вместо `bytearray(n)` + page-by-page копий → проверь работу на 100 MB чтении и оптимизируй если медленно (осторожно: bit-exact).
- [x] **R19.** `run.py` — флаг `--quiet` для bench/verify, чтобы не печатать energy report.
- [x] **R20.** `nqx-audit` — добавь `--summary` режим: после прогона распечатать в чат первые 20 строк каждого результата.

## Round 2 — задачи под новый функционал, что закрыл Claude (T1–T9, T13–T15)

### Тесты для новых юнитов

- [x] **R21.** `tests/test_attention_unit.py` — тесты для `AttentionUnit.dot_polar` (T3): идентичность с обычным dot product через decode→FROM_POLAR→GVNS_INV→dot, проверить на 5×5 batch q/k и dim ∈ {64, 128}, RMSE < 1e-3.
- [x] **R22.** Расширить `tests/test_mx_unit.py` — edge cases для **всех** поддерживаемых форматов: MXFP4 / MXFP6 / MXFP8 / MXINT8. На constant input, all-zeros, single outlier per block. RMSE budget per format.
- [x] **R23.** `tests/test_subbit_unit.py` — все пары `(r_bits, θ_bits)` ∈ {(3,1), (3,2), (2,1), (2,2), (4,2)}. Сравнить compression vs RMSE, записать таблицу в чат.
- [x] **R24.** Расширить `tests/test_assembler.py` — encode/decode roundtrip для новых mnemonic: `MXPACK`, `MXUNPACK`, `SUBBIT_ENC`, `SUBBIT_DEC`, `ATTN_DOT`, `LDV_ASYNC`. Проверить bad input для каждого.

### NQ-ASM примеры под новые opcodes

- [x] **R25.** Создай в `programs/`:
  - `programs/encode_mx_dim128.nqasm` — пример с `MXPACK V0, MXFP4` после rotate+polar.
  - `programs/encode_subbit_dim128.nqasm` — пример с `SUBBIT_ENC V0, 3, 2`.
  - `programs/attention_dot_dim128.nqasm` — пример: encode q (V0), encode k (V1), `ATTN_DOT V0, V1`.
  Каждый ассемблируется + исполняется в `python run.py run programs/<file>.nqasm` без ошибок.

### Bench — новые FU

- [x] **R26.** Расширить `bench/run_all.py` — добавить замеры:
  - `AttentionUnit.dot_polar` на batch (32×32), (128×128), (512×512), dim=128
  - `MXQuantizer` на 4096 vec dim=128 для каждого формата
  - `SubBitUnit` на 4096 vec dim=128 для пар (r=3,θ=1) и (r=3,θ=2)
  Результаты — в `audits/results/bench-extended-<timestamp>.md`.

### README / docs

- [ ] **R27.** В `README.md` добавь короткую таблицу «ISA v2 quick reference» — все opcodes (NOP, LDV, …, ATTN_DOT) в одну строку каждый: hex, mnem, что делает (≤8 слов). Не дублировать `docs/architecture.md`, ссылка туда.

### CLI / tooling

- [x] **R28.** `tools/cli/nqx-doctor` — bash диагностика окружения:
  - наличие `numpy`, `pytest`, `fastapi`, `uvicorn` (через `python -c 'import …'`)
  - наличие `claude`, `codex`, `deepseek-cli` в PATH
  - линки в `~/.local/bin/nqx-*` валидны
  - `~/Desktop/NQX-Core.desktop` существует и executable
  - `python -m pytest tests -q --tb=no` проходит
  Вывод — лаконичный, цветной (✓/✗), один блок status в конце.

## Workflow

1. Один `[ ]` за раз
2. Минимальный diff
3. `pytest tests -q` после каждого изменения — должно зеленить
4. `[x]` обновить в этом файле
5. Не лезь в задачи помеченные `T*` (это для Claude/heavy)

Помни: **минимум диффа, никаких рефакторов, никаких docstrings**.
