# Pre-silicon SDK — то что собирают перед tape-out

Это финальный stack который превращает «эмулятор + RTL» в **полноценный chip
development kit** уровня RISC-V Zephyr SDK / Chipyard / Caravel-Efabless.

Без этого можно жить на эмуляторе, но к фабрике не понесёшь.

## Перед стартом

- `python -m pytest tests -q` зеленит
- heavy.md / routine.md / demo.md / scenarios.md — желательно завершены
- `nqx-status` показывает 100% по предыдущим спискам

---

## Часть А — для CLAUDE (heavy / RTL / synthesis)

- [x] **SDK1. Random Instruction Generator (RIG).** `tools/rig.py` — генератор
  валидных рандомных NQ-ASM программ:
  - случайная последовательность из 10-100 инструкций
  - корректные регистры, корректные адреса, корректные параметры
  - запускает через ассемблер + исполняет на NQXCore
  - проверяет invariants: не падает, нет нагрева VRF, нет undefined state
  - 1000 итераций → лог coverage

- [x] **SDK2. Coverage tracking.** `nqx/coverage.py` — instrumentation:
  - какие opcodes были выполнены (× count)
  - какие комбинации pair (например LDV→GVNS→POLAR)
  - какие register pairs читались/писались
  - дамп в `audits/results/coverage-<ts>.md` после каждого RIG-run
  - target: 100% coverage по opcodes после 100 RIG-runs

- [x] **SDK3. Disassembler.** `nqx/disassembler.py` — bytes → ASM:
  - принимает bytecode (от `pack_program`)
  - возвращает текст NQ-ASM, идентичный input ассемблеру
  - roundtrip test в `tests/test_disassembler.py`: assemble → disassemble → assemble
    bit-identical
  - CLI: `python -m nqx.disassembler program.bin`

- [x] **SDK4. Yosys synthesis flow.** `rtl/synth/synth.ys` — open-source synthesis:
  - читает все `*.sv` из `rtl/`
  - целевая cell library — generic / sky130
  - выводит netlist + gate count + estimated area
  - `make synth` в `rtl/synth/Makefile`
  - в `rtl/synth/README.md` — как интерпретировать output, цели по gate count

- [x] **SDK5. OpenLane configuration.** `rtl/openlane/config.json` — Caravel-style:
  - `DESIGN_NAME = nqx_top`
  - `VERILOG_FILES = ["dir::rtl/*.sv"]`
  - `CLOCK_PERIOD = 10` (100 MHz starter target — sky130 предел)
  - `FP_CORE_UTIL = 50`
  - `PL_TARGET_DENSITY = 0.55`
  - `rtl/openlane/README.md` — как запустить через docker openlane2,
    что ожидать в `runs/`, как сделать tape-out submission в Efabless MPW

- [x] **SDK6. Formal verification harness.** `rtl/formal/`:
  - `formal/orthogonality.sby` — SymbiYosys script для проверки `T·T^T = I` через
    SystemVerilog Assertions
  - `formal/properties.sv` — SVA assertions: norm preservation, no-NaN, pair
    non-overlapping per layer
  - `formal/Makefile` — `make formal` запускает SymbiYosys
  - в `formal/README.md` — какие property доказаны, какие BMC vs k-induction

- [x] **SDK7. Performance counters.** `nqx/counters.py` + добавить в RTL:
  - HW perf counters: `cycle_count`, `stall_cycles`, `gu_busy_cycles`,
    `pu_busy_cycles`, `qu_busy_cycles`, `dma_in_bytes`, `dma_out_bytes`,
    `prng_cycles_baseline` (для honest random comparison)
  - MMIO addresses в `docs/architecture.md` (расширить §3.4)
  - read через scalar register `S0..S7`
  - тест `tests/test_counters.py`

- [x] **SDK8. JTAG / debug interface model.** `nqx/jtag.py` + RTL:
  - TAP controller state machine (IEEE 1149.1)
  - DR / IR scan chains
  - команды: read VRF, read PC, read CSR, single-step, breakpoint set/clear
  - software-side: `tools/cli/nqx-debug-jtag` — connect, dump state, step
  - тест `tests/test_jtag.py` — все state transitions

---

## Часть B — для DEEPSEEK (software / docs / tools)

- [x] **SDK9. libnqx (C ABI).** `sdk/libnqx/libnqx.h` + `sdk/libnqx/libnqx.py`:
  - C-style header: `nqx_handle nqx_open(const char* config); int nqx_encode(...); int nqx_decode(...); void nqx_close(...);`
  - Python implementation на основе `NQXCore` для прототипа
  - тест `sdk/libnqx/test_abi.py` — проверяет что functions exposed правильно
  - `sdk/libnqx/README.md` — как линковать в C/C++ приложение
  - В будущем — реальный `.so` через ctypes / pybind11; пока Python proof

- [x] **SDK10. Linux kernel driver skeleton.** `firmware/driver/nqx_driver.c`:
  - PCIe device probe / remove
  - BAR mapping (MMIO области из §3.4 architecture)
  - char device `/dev/nqx0`
  - read/write/ioctl interface (skeleton, not functional)
  - `firmware/driver/Makefile` — кросс-компиляция против Linux headers
  - `firmware/driver/README.md` — как вставить в `dkms`, какие kernel API нужны

- [x] **SDK11. Boot ROM / firmware.** `firmware/boot/boot.nqasm`:
  - bringup sequence: clear VRF, clear SRF, init LUT (если нужно), barrier
  - сравнить с reset state (всё zeros), проверить orthogonality self-test
  - `firmware/boot/Makefile` — assemble в `boot.bin` + hex
  - `firmware/boot/README.md` — bringup протокол

- [x] **SDK12. Programming guide.** `docs/programming_guide.md`:
  - таблица: «когда использовать ENC макрос vs LDV+GVNS+...»
  - паттерны: streaming encode (один LDV → много ENC), batched (много LDV)
  - типичные ошибки: писать в V0 после ENC без MOV, забывать BARRIER перед STV
  - примеры на каждый паттерн (рабочие .nqasm файлы)

- [x] **SDK13. Errata.** `docs/errata.md`:
  - known limitations: max batch=1024 (DMA bound), max dim=512 (VRF size)
  - software workarounds для каждой
  - revision history (v1.0, v1.1, …) когда появятся
  - empty section "ASIC errata" — заполнится после первого silicon

- [x] **SDK14. SDK installer.** `sdk/install.sh`:
  - копирует `nqx/`, `tools/cli/`, `sdk/libnqx/` в `~/.local/share/nqx-sdk/`
  - линкует bin'ы в `~/.local/bin/`: `nqx-asm`, `nqx-disasm`, `nqx-sim`,
    `nqx-rig`, `nqx-debug`
  - устанавливает `python -m pip install -e .` если есть `pyproject.toml`
  - проверяет dependencies (numpy, etc)
  - `sdk/README.md` — что после install будет доступно

- [x] **SDK15. SDK overview.** `sdk/README.md`:
  - один документ: что в SDK, как этим пользоваться
  - таблица всех бинарей (nqx-asm, nqx-sim, ...) с одной строкой описания
  - HelloWorld пример: написать NQ-ASM, ассемблировать, запустить, прочитать
    результат
  - ссылки на каждый sub-document (architecture, programming_guide, errata, …)

---

## Acceptance — что значит «pre-tape-out SDK готов»

| Что | Где |
|---|---|
| Random instr gen прогоняет 1000 программ без падения | SDK1 |
| Coverage достигает 100% opcodes после 100 RIG-runs | SDK2 |
| `assemble → disassemble → assemble` bit-identical | SDK3 |
| `make synth` в Yosys выдаёт gate count | SDK4 |
| `openlane2` запускается с нашим config (можно прервать после floorplan) | SDK5 |
| SymbiYosys доказывает orthogonality assertion | SDK6 |
| Performance counters читаются через scalar regs | SDK7 |
| JTAG модель проходит full IEEE 1149.1 state coverage | SDK8 |
| `gcc test.c -lnqx` компилируется и линкуется | SDK9 |
| Driver skeleton компилируется против `make -C /lib/modules/$(uname -r)/build` | SDK10 |
| `boot.bin` ассемблируется и проходит self-test | SDK11 |
| Programming guide ≥ 5 рабочих примеров | SDK12 |
| Errata актуальна с текущим состоянием | SDK13 |
| `bash sdk/install.sh` за 30 сек, всё в PATH | SDK14 |
| SDK README — индекс всего | SDK15 |

После этого мы готовы:
1. **Сделать MPW shuttle на Skywater 130nm** через Efabless ($0 за участие в Open MPW, или ~$10K за commercial)
2. **Прийти к коммерческому foundry** (TSMC 12nm/7nm) — у нас RTL, верификация, перф/пауэр, errata, доки. Они откроют design review.
3. **Дать SDK партнёрам** — vLLM/llama.cpp интегрируют через libnqx, Linux дистры пакают driver.

## Workflow

1. **Claude (heavy)** берёт SDK1-SDK8
2. **DeepSeek (routine)** берёт SDK9-SDK15
3. После каждой задачи — pytest + acceptance test для конкретного SDK#
4. Помечают `[x]` в этом файле
5. По завершении всех 15 — финальный merge sweep, обновить главный `README.md`
   секцией «Pre-silicon SDK», обновить `docs/PRD.md` секцией §10 (Что готово к tape-out)
