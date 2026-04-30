# Поиск багов в NQX-Core

Ты — senior verification engineer, ищешь баги. Не оптимизации — баги.

## Что прочитать

- `nqx/lut.py` — генератор golden-angle LUT
- `nqx/functional_units.py` — GivensUnit, PolarUnit, QuantUnit, QJLUnit, PackUnit
- `nqx/cpu.py` — `NQXCore.encode`, `NQXCore.decode`, `_do_enc`, `_do_dec`, `rotation_matrix`
- `nqx/memory.py` — VRF, HBM lazy pages
- `tests/test_*.py` — какие инварианты проверяются и какие — нет

## Ищи (в порядке убывания приоритета)

1. **Численные ошибки**. Где FP32 → FP16 round-trip ломает математику? Что происходит при vector с одним элементом (`mins == maxs`)? Где может случиться `nan/inf`? Что если входные данные содержат `inf` или `nan`?
2. **Off-by-one в LUT**. L1 имеет 64 пары, L2 — 63, L3 — переменное число. Точно ли все индексы валидны для всех `dim`? Что если `dim` нечётный? Что если `dim < 4`?
3. **Несоответствие encode/decode**. Если данные «слабо коррелируют» (constant, all zeros, single outlier) — корректно ли восстанавливаются?
4. **State leakage между вызовами**. `last_pack_meta` — там точно нет утечек между разными `encode()`? Что если вызвать `decode()` без предшествующего `encode()`?
5. **Pack/unpack INT3+1**. Точно ли pack/unpack — обратные операции для всех значений `q ∈ [0..7]` и `sign ∈ {0,1}`? Endianness, padding последнего байта.
6. **HBM lazy pages**. Если запись пересекает границу страницы — точно ли корректно склеивается? Что при чтении out-of-bounds?
7. **Тесты на orthogonality** — достаточно ли строгие? Достаточно ли разнообразные данные? Что если тестировать на реальном KV-cache (не нормально-распределённом)?
8. **rotation_matrix() через apply_layer на eye**. Корректно ли это даёт T или T.T? Как проверить?

## Формат ответа

Список багов, каждый:
- **Severity:** critical / high / medium / low
- **File:line:**
- **Описание бага** (1-2 предложения)
- **Минимальный воспроизводитель** (Python-snippet или test case)
- **Фикс** (diff или текст изменения)

Если багов нет в каком-то файле — скажи прямо «file X clean».
