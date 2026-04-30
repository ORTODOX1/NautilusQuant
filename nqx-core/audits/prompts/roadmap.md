# NQX-Core roadmap: от эмулятора к кремнию

У меня сейчас работающий software-эмулятор NQX-Core (pure NumPy) + REST-сервер с CPU/GPU backend. Хочу довести до:
1. **FPGA-прототип на AMD Alveo U280 / V80 или AWS F1**.
2. **ASIC tape-out** (TSMC 7nm или GlobalFoundries 12nm).
3. **Интеграция в LLM inference stack** (vLLM, llama.cpp, transformers).

## Что у тебя есть

Прочитай весь `/home/user/nautilusquantcore/`:
- `README.md`, `docs/architecture.md`
- `nqx/`, `programs/`, `tests/`, `server/`
- `Dockerfile`, `Dockerfile.cpu`, `deploy/`

## Что хочу от тебя

1. **Конкретный roadmap по этапам**, каждый этап с:
   - **Цель** (одно предложение, измеримая)
   - **Артефакт** (что появится в репо: новые файлы / новые папки)
   - **Acceptance criteria** (как проверить что этап завершён)
   - **Время** (часы / дни / недели для разраба-одиночки и для команды 2-3 человек)
   - **Стоимость** (ТД-инстансы / FPGA-карта / EDA-tools / tape-out)
   - **Риски** (что может пойти не так, mitigation)

2. **Список этапов:**
   - **E1**: NQX-Core → Verilator-bit-exact RTL (SystemVerilog) с тестбенчем, который заставляет сравнивать выход RTL и Python pop-by-pop.
   - **E2**: Synthesizable RTL → AMD Vivado synth → AWS F1 / Alveo U280, реальный benchmark.
   - **E3**: интеграция в HuggingFace transformers как KV-cache hook (kv_quant=True).
   - **E4**: интеграция в vLLM как replacement для AWQ/GPTQ KV-quant.
   - **E5**: ASIC tape-out: floor-plan, DFT, scan chains, 7nm PDK.
   - **E6**: PCIe board bring-up, host driver, Linux kernel module, userspace API.

3. **Какие команды специалистов нужны на каждом этапе** (RTL designer, DV engineer, kernel driver, ML engineer, foundry-side support).

4. **Где можно сэкономить** — какие этапы можно объединить, какие пропустить (например, использовать готовые HBM IP вместо своего PHY).

5. **Альтернативные пути**: что, если сразу прыгать в Tenstorrent (open-source ISA, можно компилить свой kernel без RTL), или сразу в Groq Compiler (если они дадут SDK)?

6. **Конкретные числа**:
   - Цена тейпаута на 12nm vs 7nm vs 5nm
   - Перформанс-цели: сколько vec/sec нужно в production
   - Mass-production ROI: при каком объёме ASIC окупается над FPGA

## Формат ответа

Большая структура с таблицами. Не «короткий план», а **руководство которое можно повесить на стену и идти по нему**. Без «это сложно» — конкретно. Прямо и жёстко.
