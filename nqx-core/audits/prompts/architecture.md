# Архитектурный ревью NQX-Core

Ты — senior hardware architect, делаешь независимое ревью эмулятора процессора.

## Контекст

Я только что закончил `NQX-Core` — программный эмулятор специализированного процессора (NPU/ASIC-class) под пятистадийный pipeline NautilusQuant: квантование KV-кэша LLM через детерминированные вращения с золотым углом φ. Проект лежит в `/home/user/nautilusquantcore/`.

Прочитай:
- `README.md`
- `docs/architecture.md`
- `nqx/constants.py`, `nqx/lut.py`, `nqx/memory.py`, `nqx/functional_units.py`, `nqx/pipeline.py`, `nqx/cpu.py`, `nqx/isa.py`
- `programs/encode_dim128.nqasm`

## Что хочу услышать

1. **Архитектурные ошибки.** Что не так в выборе SIMD lanes, ширины VRF, глубины pipeline, ROM/SRAM sizing? Что я не учёл? Где будет реальный bottleneck не там, где я указал?
2. **ISA-design**. Достаточен ли набор opcodes? Что лишнее? Чего не хватает (например, attention-fused операции, KV-batch handling, async DMA)? Стоит ли разделить scalar/vector ISA? Достаточно ли 8-bit поля для register index?
3. **Параллелизм**. Как лучше распараллелить L1/L2/L3 Givens? Является ли мой выбор «64 lanes × 2 elements» действительно оптимальным для dim=128, или лучше 128 lanes × 1 element с lane-permute network? Какие costs у permute fabric?
4. **Pipeline depth = 18**. Это много или мало? Где можно forwarding-ом срезать стадии?
5. **Совместимость с реальной плиткой**. Если бы это синтезировалось в RTL и шло на TSMC 7nm — какая получилась бы площадь, частота, мощность? Какие модули плохо ложатся на стандартные cell library?
6. **Сравнение с реальными NPU**. Чем мой дизайн похож/отличается от: Groq TSP, Google TPU MXU, Tenstorrent Tensix, NVIDIA Tensor Core, AMD CDNA Matrix Core, Cerebras CE? Что я невольно повторил, что упустил?
7. **Масштабирование**. Что меняется при dim=256, dim=512, dim=1024? Есть ли скрытое квадратичное поведение?
8. **Что бы ты отрезал, что добавил**, если бы взял этот дизайн и тейпнул его как ASIC?

## Формат ответа

- В начале — **3 главные ошибки** в порядке убывания важности.
- Дальше — **3 главные удачи** дизайна (что сделано правильно).
- Дальше — конкретный numbered punch list изменений: «добавь X в `nqx/foo.py`», «убери Y», «измени параметр Z с A на B», и т.д.
- Без воды, без «в целом неплохо». Жестко.
