# Heavy tasks — для Claude (тяжёлая работа, большие изменения)

Ты — senior systems / hardware engineer на проекте NQX-Core. Работаешь автономно: читаешь PRD, берёшь первый незакрытый таск, делаешь его под ключ (код + тесты + bench-проверка), отмечаешь `[x]`, переходишь к следующему. Не спрашивай разрешения на каждом шаге — **молоти**.

## Перед стартом

1. Прочитай `docs/PRD.md` — roadmap, scope, правила.
2. Прочитай `CLAUDE.md` — стиль работы.
3. Запусти `python -m pytest tests -q` — должно быть 32+ PASS. Если что-то красное — сначала почини.

## Список задач (бери первый `[ ]`, делай, отмечай `[x]`)

### Architecture / ISA расширения

- [x] **T1. MXFP4 backend.** Реализуй `nqx/mx_unit.py` с классом `MXQuantizer` (block size 32, 8-bit shared exponent). Добавь opcodes `MXPACK` (0x52) и `MXUNPACK` (0x53) в `nqx/isa.py`, ассемблер, диспатч в `nqx/cpu.py`. Тесты `tests/test_mx_unit.py`. Проверка vs `nautilus_hardware.py::NautilusWithMX` из upstream — bit-exact.

- [x] **T2. Sub-bit ISA extension.** Concept 4 из upstream: раздельная квантизация radius (3 bit) и angle (1-2 bit). Новый opcode `SUBBIT_ENC V0, r_bits, θ_bits`. Реализуй в `nqx/subbit_unit.py`. Тесты для пар (r_bits, θ_bits) ∈ {(3,1), (3,2), (2,1), (2,2)} с RMSE budget.

- [x] **T3. Attention-fused ISA.** Текущий ISA даёт только encode/decode. Добавь `ATTN_DOT V_q, V_k` — скалярное произведение в polar домене (без декода обратно). Это ключ для интеграции в attention loop. Файл `nqx/functional_units.py::AttentionUnit`, opcode 0x80.

- [x] **T4. NQ-ISA v2 spec.** Обнови `docs/architecture.md`: добавь раздел про MX, sub-bit, attention-fused operations. Заверши таблицу opcodes до полной картины. Дай примеры NQ-ASM для каждой новой инструкции в `programs/`.

### Performance

- [x] **T5. PackUnit hot path.** Сейчас `pack3plus1`/`unpack3plus1` — bit-by-bit Python loops. Перепиши через `np.packbits` / bit manipulation на uint64 chunks. Цель: encode 4096 vec dim=128 < 50 ms (сейчас ~500 ms). Cycle-counter и energy-model не меняй.

- [x] **T6. Vectorized GivensUnit.apply_layer.** Сейчас цикл по парам. Все пары layer не пересекаются → можно vectorized advanced indexing: `out[..., pair_i_arr], out[..., pair_j_arr] = ...`. Цель: x10 speedup. Bit-exact.

- [x] **T7. Async DMA model.** Сейчас LDV блокирующий. Реализуй overlap: `LDV_ASYNC vd, [addr]` начинает DMA, `BARRIER` ждёт. Pipeline counter учитывает overlap с compute.

### RTL (E2)

- [x] **T8. RTL skeleton.** Создай `rtl/`:
  - `givens_lane.sv` — один lane (4 mul + 2 add, FP32)
  - `golden_rom.sv` — ROM, init из `$readmemh "golden_rom.mem"` (генерируй из `nqx/lut.py`)
  - `polar_unit.sv` — sqrt + atan2 CORDIC, 4-stage pipelined
  - `quant_unit.sv` — Lloyd-Max with min/max reduce tree
  - `nqx_top.sv` — top-level wrapper
  - `tb_nqx.sv` — Verilator testbench, читает hex дамп от `nqx/cpu.py`, сравнивает выход
  - `Makefile` — `make sim` запускает Verilator, проверяет bit-exact с Python

- [x] **T9. Generator script.** `tools/gen_rom.py` — Python скрипт, генерирует `rtl/golden_rom.mem` из `GoldenAngleLUT` (FP32 как 4-byte hex). Проверка: то же значение что `cos(θ_k)`.

### LLM integration (E4) — на vast.ai

> T10–T12 требуют CUDA + transformers/vllm/triton. Перенесены в **`heavy-gpu.md`** —
> делать на vast.ai-инстансе, не локально. Их нет в этом списке.

### Server / CI

- [x] **T13. CI workflow.** `.github/workflows/ci.yml` — pytest, ruff, black --check на каждый PR. Cache pip. Matrix: Python 3.11, 3.12.

- [x] **T14. Multi-arch Docker.** Поправь `Dockerfile.cpu` чтобы билдилось на amd64 + arm64. `docker buildx`. Документируй в `deploy/README.md`.

### ASIC (E5) — design docs

- [x] **T15. Floor-plan документ.** `asic/floorplan.md` — диаграмма размещения GU/PU/QU/PACK/SRAM на 50 mm² die в TSMC 7nm. Оценка площади каждого блока (gate count, mm²). Power islands.

- [x] **T16. Timing closure report.** `asic/timing.md` — критические пути, целевая частота 1 GHz, slack analysis. Какие блоки нужно retimed/pipelined.

## Round 2 — без GPU, продолжаем молотить

- [x] **T17. Paper draft.** `docs/paper/intro.md` (1-2 страницы: motivation, problem, contribution) + `docs/paper/results.md` (текущие measured numbers: orthogonality 1.6e-7, throughput, energy/vec, RMSE bounds, compression 4.00×). Стиль академический, для arXiv. Refs: TurboQuant, KIVI, QuIP#, NautilusQuant upstream.

- [x] **T18. Tape-out checklist.** `asic/tapeout_checklist.md` — реальный pre-tape-out checklist по 9 разделам: DFT (scan chains, BIST), IO ring, ESD protection, package selection, reticle / die size limits, multi-corner timing sign-off, IR drop, EM, formal verification (LEC). Каждый пункт — что проверить + критерий пройдено/не пройдено.

- [x] **T19. llama.cpp design doc.** `integrations/llama_cpp_kvquant.md` — design doc для C++ адаптера (без кода, только interface): что реализовывать в `ggml-cuda.cu`, какие функции экспортить, как вызывать через `--kv-quant nqx`. Спецификация под T11 после vast.ai.

- [x] **T20. Ablation study.** `bench/ablation.md` — таблица: φ-rotation vs random rotation vs Hadamard vs no-rotation, на синтетике с outliers. Combinations × dim ∈ {64, 128, 256} × bits ∈ {2, 3, 4}. RMSE per (rotation, dim, bits). Запусти численно, заполни markdown.

## Round 3 — PROOF tasks (доказательственная часть, без неё проект не paper)

Цель этого раунда — превратить implementation в **доказательство** ключевой гипотезы:
*детерминистическое φ-сжатие на static dataflow процессоре ≥ random на GPU при кратно меньшем
LUT и нулевой PRNG-нагрузке.*

- [x] **T21. Angular uniformity proof.** `bench/angular_uniformity.py` — статистический тест:
  для каждого ротора (φ-Givens, random-Givens (несколько seeds), Hadamard) посчитай
  **discrepancy** угловой дистрибуции (Kolmogorov-Smirnov vs uniform на круге, и L∞ star discrepancy).
  Цель: показать что φ даёт O(1/N), random даёт O(1/√N) — это и есть теоретическая база.
  Артефакт: `bench/angular_uniformity.md` с таблицей и ссылкой на seminal Weyl equidistribution.

- [x] **T22. Linear vs Lloyd-Max quant after φ-rotate.** `bench/linear_quant.md` — самый важный тест:
  после φ-rotation замерить RMSE для **uniform (linear)** quantize vs **Lloyd-Max** на одинаковых
  данных. Если разница < 5% — это прямое доказательство что φ-rotation создаёт почти равномерное
  распределение, и Lloyd-Max не нужен (значит можно делать linear-quant в железе на 1 такте,
  без Lloyd-Max таблицы кодов).

- [x] **T23. φ vs Random head-to-head.** `bench/phi_vs_random.md` — три метрики на одинаковых
  данных (synthetic + real KV если есть):
  - RMSE после encode→decode
  - Wall time (ms)
  - Cycle count из NQX-Core (для random rotation добавить PRNG-cycles в модель)
  Цель: показать что φ ≥ random по quality и < random по cycles/energy.

- [x] **T24. Determinism witness.** `bench/determinism.md` — запусти φ-encode 100 раз на одном
  входе с разными timestamps и (для baseline) random-encode с разными PRNG seeds.
  Покажи что φ даёт **bit-identical** выход всегда, random — разный.
  Это формальное доказательство ключевого свойства.

- [x] **T25. LUT size proof.** `bench/lut_budget.md` — таблица: dim × layer × angles × LUT bytes.
  Покажи что для dim ∈ {64, 128, 256, 512} наш LUT остаётся ≤ 4 KB. Сравни с size random
  rotation matrix (dim² × FP16 = 32 KB → 512 KB → 8 MB). **Это уничтожающее преимущество.**

- [x] **T26. Cycle/energy delta vs random.** Расширь `nqx/energy.py` с моделью random rotation
  (PRNG cycles + extra mul по dim²). Сравни total energy для batch=4096 dim=128:
  φ-rotation vs random rotation. Артефакт: `bench/energy_proof.md`.

## Workflow

1. Беру первый `[ ]` ↑
2. План в чат: «делаю T_N, файлы такие-то, оценка минут такая»
3. Glob/Grep по проекту — что уже есть
4. Реализация
5. `python -m pytest tests -q` — должно проходить
6. `python run.py verify` — acceptance ок
7. Отметить `[x]` в этом файле, commit (если git инициализирован)
8. К следующему `[ ]`

Помни: **никаких docstrings/комментариев без явной просьбы**, никакого скоупа за пределами таска. Бит-точность с upstream NautilusQuant — обязательна.
