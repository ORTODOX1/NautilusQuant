# NQX-Core vs upstream NautilusQuant — точное сравнение

Сверь две реализации:

**Моя:** `/home/user/nautilusquantcore/nqx/` (NQX-Core, pure-numpy эмулятор)
**Upstream:** `https://github.com/hermandoronin/NautilusQuant`, основные файлы:
- `nautilus_triton.py` → `NautilusQuantPyTorch` (reference math)
- `nautilus_hardware.py` → `NautilusFusedKernel`, `NautilusDataflow`, `NautilusWithMX`, `SubBitExperiment`
- `validate_real_kv.py` (как тестируют на реальном KV-cache)

## Что сделать

1. **Бит-точное совпадение математики.**
   - Углы L1/L2/L3 — совпадают (мы показали в `tests/test_vs_reference.py`).
   - Порядок применения rotation внутри слоя — совпадает?
   - Polar/inverse polar — то же определение `r=sqrt(x²+y²), θ=atan2(y, x)`?
   - Lloyd-Max — `min/max` per-feature по batch axis=0, или axis=1?
   - QJL — `quantized + sign(error) * |error| * α` ?
   - Pack — endianness, порядок битов sign-vs-quant?

2. **Чего я НЕ реализовал из upstream**:
   - MX fallback (`NautilusWithMX`) → MXFP4
   - Sub-1-bit experiment (`SubBitExperiment`) — раздельные radius/angle bits
   - Multimodal adapter (text/image/audio configs)
   - Static schedule compilation (`NautilusDataflow._compile_schedule`)
   - Triton kernel path
   Каждое — нужно ли в эмуляторе?

3. **Где я отступил от reference и почему**:
   - hbm_bytes default (16 GB → 256 MB → 4 GB lazy)
   - Cycle counts (никаких в reference, у меня жёстко прибиты)
   - Energy model (никакой в reference)
   Это окей, или вводит несовместимость?

## Формат ответа

- **Bit-exact diff** в виде таблицы: «Stage X: NQX делает Y, reference делает Z, эквивалентно/нет».
- **Список TODO** для добавления в NQX: «реализовать MX fallback в `nqx/mx_unit.py`», «добавить opcode `MXPACK` в `nqx/isa.py`», и т.д. Каждый TODO с описанием новой функции и заголовком файла.
- **Список расхождений, которые НЕ нужно править** (например, ENERGY_MODEL — наш уникальный фичур).

В конце — сводный sanity-check тест, который надо добавить в `tests/test_vs_reference.py`, чтобы поймать любой будущий drift.
