# Audits — куда все CLI-агенты пишут свои отчёты

```
audits/
  prompts/      готовые промпты для аудита
  results/      ответы CLI (gitignored)
  logs/         stderr/exit codes (gitignored)
```

## Запуск

```bash
# один промпт во все CLI параллельно
nqx-audit architecture

# только в одну CLI
nqx-audit performance --only deepseek

# спалить максимум токенов: каждый промпт × каждая CLI
nqx-audit --all

# список доступных промптов
nqx-audit --list
```

Результаты складываются в `results/<cli>-<prompt>-<timestamp>.md`.

## Что в каждом промпте

| Промпт | Что просит | Для чего |
|---|---|---|
| `architecture.md` | Hardware-architect ревью | Найти архитектурные косяки в NQX-Core |
| `correctness.md` | Verification ревью | Поиск багов, edge cases, off-by-one |
| `performance.md` | Performance ревью | Узкие места эмулятора, оптимизации |
| `vs-reference.md` | Сравнение с upstream NautilusQuant | Bit-exact diff, что не реализовано |
| `roadmap.md` | Дорожная карта от Python → ASIC | E1-E6 этапы, цена, время, риски |
| `full.md` | Все 4 ревью в одном промпте | Один большой отчёт, максимум токенов |

## Своя промпт

```bash
echo "# Своя задача\n\nТы делаешь X..." > audits/prompts/myask.md
nqx-audit myask
```

Файл должен быть `<name>.md`, иначе `nqx-audit` его не подхватит.
