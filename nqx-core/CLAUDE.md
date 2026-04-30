# Project memory: NQX-Core

Этот файл — project-level instructions. Глобальные правила пользователя в `~/.claude/CLAUDE.md` остаются в силе, здесь только проектное.

## Что это за проект

`NQX-Core` — software-эмулятор специализированного процессора под pipeline
[NautilusQuant](https://github.com/ORTODOX1/NautilusQuant) (квантование KV-cache
LLM через золотое сечение). Полный PRD: `docs/PRD.md`.

## Прежде чем что-то делать

1. Прочитай `docs/PRD.md` — там roadmap, архитектура, acceptance criteria, scope.
2. Если задача — из списка задач, прочитай `audits/prompts/heavy.md` (большие
   тяжёлые) или `audits/prompts/routine.md` (рутина) и возьми **первый** не
   отмеченный таск (`[ ]`).

## Жёсткие правила

- **Не добавляй docstrings/комментарии** если не просят явно.
- **Не добавляй features** которых нет в текущем таске.
- **Не рефактори** вне scope.
- **pytest должен проходить целиком** после каждого изменения:
  `python -m pytest tests -q`
- Code, identifiers, commits — English. Общение со мной — Russian.
- Прежде чем писать новое — Glob/Grep по существующему коду. Не дублировать.

## Команды быстрой проверки

```bash
python -m pytest tests -q            # все 32+ тестa
python run.py verify --dim 128       # acceptance
python run.py bench --vectors 4096   # перформанс + энергия
bash deploy/test_api.sh http://localhost:8000   # if server running
```

## Когда добавляешь файлы

- Новый функциональный модуль → `nqx/<name>.py` + `tests/test_<name>.py`
- RTL → `rtl/<name>.sv` + Verilator testbench `rtl/tb_<name>.sv`
- Интеграция в LLM stack → `integrations/<framework>_adapter.py`
- Новый dataset для бенча → `bench/datasets/`

## Когда отчитываешься

- В чате — кратко: что изменено (файл:строка), что сломалось / что работает.
- Без trailing summaries и квот моих сообщений.
- Tables > paragraphs.
- Если что-то срезал/решил пропустить — назови это явно.
