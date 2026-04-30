# Performance-аудит NQX-Core (Python emulator)

Ты — performance engineer. Текущий эмулятор работает на pure NumPy и сейчас чисто-эмуляционный, но я хочу чтобы:
- Референс-encode 4096 векторов dim=128 на CPU занимал секунды, а не пол-минуты.
- Cycle-counter оставался корректным.
- Hot path был достаточно быстрым, чтобы можно было прогонять real KV-cache на CPU.

## Текущая ситуация

`python run.py bench --vectors 4096 --dim 128` — encode ~500 ms, decode ~25 ms. Decode в 20× быстрее encode — ненормально, encode должен быть симметричен или быстрее.

## Что прочитать

- `nqx/functional_units.py` — особенно `GivensUnit.apply_layer`, `PackUnit.pack3plus1`, `PackUnit.unpack3plus1`
- `nqx/cpu.py` — `_do_enc`, `_do_dec`, `forward_rotation`, `inverse_rotation`
- `run.py` — `cmd_bench`

## Найди и исправь

1. **Где Python-loop там, где должен быть векторный numpy-op.** Особенно в `apply_layer` — он итерирует по парам, но для каждого слоя пары независимы (non-overlapping), их можно применять одной vectorized операцией через advanced indexing.
2. **PackUnit — bit-by-bit Python loop.** Это самое медленное место. Перепиши через `np.packbits` или bit-manipulation на uint64 chunks.
3. **`out[..., i].copy()` — лишние copies?** Можно ли через temp-views избавиться?
4. **Что делать с тем, что quantize делает axis=0 reduce per-batch — это правильно для feature-wise, но точно ли так в reference?** Если данных мало (n=1), деление на zero range — какая стратегия? eps клемп достаточен?
5. **Memory allocations в hot path.** Где `np.zeros_like` можно заменить на pre-allocated buffer?

## Формат ответа

- **Топ-3 хотспота** с измеренной долей времени (если можешь — дай benchmark snippet).
- **Patch-style diff** для каждого хотспота.
- **Ожидаемое ускорение** после фикса (×N).
- **Что НЕ нужно оптимизировать** — где затраты не оправданы.
- В конце — финальный benchmark скрипт, который можно запустить и убедиться.

Cycle-counter и energy-model должны оставаться идентичными до и после, мы не меняем семантику.
