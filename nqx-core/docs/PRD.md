# NQX-Core — Product Requirements Document

## 1. Vision одной строкой

Специализированный процессор (NPU/ASIC-class) под пятистадийный pipeline
NautilusQuant: детерминистическое квантование KV-кэша LLM через золотое сечение φ.
NQX-Core = software-эмулятор → SystemVerilog RTL → FPGA-прототип → ASIC tape-out.

## 2. Что уже сделано (состояние на сейчас)

| Артефакт | Состояние |
|---|---|
| Software-эмулятор `nqx/` (≈1100 LOC, pure NumPy) | ✅ работает, 32 теста PASS |
| ISA NQ-ISA v1 (16 opcodes, ассемблер) | ✅ |
| Программы encode/decode на NQ-ASM | ✅ |
| Acceptance: orthogonality 1.6e-7, roundtrip 9.6e-8 | ✅ |
| Bit-exact vs upstream NautilusQuant math | ✅ (`tests/test_vs_reference.py`) |
| HTTP-сервис (FastAPI, CPU+GPU backends) | ✅ |
| Docker (CPU + CUDA images) + vast.ai инструкция | ✅ |
| CLI launchers (`nqx-claude`, `nqx-deepseek`, `nqx-audit`, …) | ✅ |
| **MXFP4 backend** (Concept 3 из upstream) | ⏳ TODO |
| **Sub-bit ISA extension** (Concept 4) | ⏳ TODO |
| **SystemVerilog RTL** (`rtl/`) | ⏳ TODO |
| **vLLM / HuggingFace integration** | ⏳ TODO |
| **Triton kernel в server/** | ⏳ TODO |
| **CI (GitHub Actions)** | ⏳ TODO |

## 3. Roadmap по этапам (E1-E6)

| ID | Этап | Артефакт | Цель |
|---|---|---|---|
| E1 | Software-эмулятор | `nqx/`, `server/` | DONE |
| E2 | RTL bit-exact | `rtl/*.sv` + Verilator testbench | RTL даёт тот же выход что Python для batch=1024 dim=128 |
| E3 | FPGA bring-up | `rtl/build/` + Vivado project | Synthesizable на Alveo U280, ≥100 MHz, throughput ≥10 K vec/s |
| E4 | LLM stack integration | `integrations/vllm_kvquant.py`, `integrations/hf_kv_hook.py` | Llama-3.1-8B inference с NQX KV-quant без потери качества |
| E5 | ASIC floor-plan | `asic/floorplan.md`, `asic/timing.md` | TSMC 7nm, 50 mm², 1 GHz, готов к tape-out |
| E6 | PCIe board bring-up | `firmware/`, kernel driver | Реальное устройство в Linux хосте через `/dev/nqx0` |

## 4. Архитектурные параметры (фиксированы)

| Параметр | Значение | Можно менять? |
|---|---|---|
| dim (KV-vector) | 128 default; supports {16,32,64,128,256,512} | да, через NQXConfig |
| bits квантизация | 3 (Lloyd-Max) + 1 (QJL sign) | да, ISA рассчитан до 8 |
| φ (golden ratio) | (1+√5)/2 = 1.618… | НЕТ, hard-coded |
| L1 pairs | adjacent, 64 для dim=128 | НЕТ |
| L2 pairs | shifted-by-1, 63 для dim=128 | НЕТ |
| L3 pairs | butterfly stride dim/4 | НЕТ |
| SIMD lanes | 64 (для 64 параллельных пар) | да, для будущего dim=256 ⇒ 128 lanes |
| VRF | 16 × dim × FP32 | да |
| Pipeline depth | 18 (steady-state 1 vec/cycle) | да, при оптимизации |

## 5. Acceptance criteria (что проверяет CI)

| Проверка | Файл | Критерий |
|---|---|---|
| `T^T·T = I` | `tests/test_orthogonality.py` | err < 1e-5 |
| Roundtrip без квантизации | `tests/test_orthogonality.py` | RMSE < 1e-5 |
| Bit-exact rotation vs reference | `tests/test_vs_reference.py` | max diff < 1e-4 |
| ISA encode/decode | `tests/test_isa.py` | bit-exact roundtrip всех opcodes |
| Pipeline cycle counter | `tests/test_pipeline.py` | predicted == measured |
| Pack/unpack 3+1 bit | `tests/test_roundtrip.py` | inverse точно |
| Compression ratio | `tests/test_roundtrip.py` | == 4.00× ровно |
| Все pytest | `pytest tests -q` | 0 failures, < 1 sec |

**После любого PR/edit pytest должен проходить целиком.** Никаких xfail.

## 6. Out of scope (не делаем)

- ❌ GUI (web index достаточно)
- ❌ Тренировка / fine-tuning моделей — только inference KV-cache
- ❌ Quantization < 4 эффективных бит (radius+angle): не помещается в roadmap
- ❌ Поддержка не-NVIDIA / не-AMD GPU в server/backends.py
- ❌ Распределённый inference (multi-GPU sharding) — отдельный проект
- ❌ Visualization/3D (это в upstream `quantsim3d.html`)

## 7. Правила работы (для AI агентов и людей)

### Язык
- Communication / commit messages explanations: **Russian**
- Code / comments / docstrings / branch names: **English**
- README на русском, docstrings (если требуются) — английские

### Стиль кода
- Python ≥ 3.11, type hints обязательны для public API
- NumPy 2.x; PyTorch и Triton — optional dependencies (только в server/backends.py и nautilus_triton.py)
- **Не добавлять docstrings/comments если задача не просит явно**
- Не добавлять speculative error handling
- Не делать рефакторов вне scope таска
- Не добавлять features которые не запрошены

### Тесты
- Каждый новый функциональный юнит → отдельный test_<unit>.py
- Поломал acceptance — сначала почини, потом всё остальное
- Бенчи в `python run.py bench`, не в pytest

### Git
- Бренчи: `feat/short-name`, `fix/...`, `chore/...`, `rtl/...`
- Один логический change = один commit
- Не амендим, не --force-push в main

## 8. Стек

| Слой | Что используется |
|---|---|
| Core эмулятор | Python 3.11+, NumPy 2.x |
| HTTP-сервис | FastAPI, Uvicorn, Pydantic 2.x |
| GPU backend | PyTorch ≥2.2, Triton ≥2.2 (опционально) |
| RTL | SystemVerilog, Verilator (sim), Vivado (synth) |
| Тесты | pytest 8.x |
| Контейнеры | Docker, docker-compose |

## 9. Структура репо

```
nqx/                эмулятор (constants/lut/memory/FU/pipeline/cpu/isa/assembler/energy)
programs/           NQ-ASM программы
tests/              pytest
server/             HTTP API
deploy/             Docker, vast.ai инструкция
docs/               architecture.md, PRD.md (этот файл)
tools/cli/          launchers (nqx-claude, nqx-deepseek, nqx-audit, nqx-heavy, nqx-routine)
audits/             prompts + результаты от AI-агентов
rtl/                SystemVerilog (E2)            ← TODO
integrations/       vLLM / HF / llama.cpp adapters (E4)  ← TODO
asic/               floor-plan, timing reports (E5)      ← TODO
firmware/           kernel driver, board bring-up (E6)   ← TODO
```

## 10. Метрики успеха к концу года

| Метрика | Цель | Сегодня |
|---|---|---|
| Throughput (RTX 5090) | > 1 M vec/s | TBD |
| Throughput (FPGA Alveo U280) | > 100 K vec/s | — |
| Compression | 4.00× | ✅ |
| Bit-exact с upstream | 100% | ✅ |
| Latency p99 | < 1 ms на 1024 vec batch | TBD |
| Энергия per vec на ASIC 7nm | < 5 nJ | теоретически 5.1 nJ |

## 11. Ссылки

- Upstream NautilusQuant: https://github.com/ORTODOX1/NautilusQuant
- OCP MX standard: https://www.opencompute.org/documents/ocp-mx-spec
- TurboQuant: https://arxiv.org/abs/2504.19874
- KIVI: https://arxiv.org/abs/2402.02750
- QuIP#: https://arxiv.org/abs/2402.04396
