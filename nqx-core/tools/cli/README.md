# NQX-Core CLI launchers

Шорткаты для запуска AI CLI прямо в контексте NQX-Core проекта.

## Установка

```bash
bash tools/cli/install.sh
```

Создаёт symlinks в `~/.local/bin/`:

| Команда | Что делает |
|---|---|
| `nqx-claude` | Anthropic Claude CLI в `/home/user/nautilusquantcore` |
| `nqx-deepseek` | DeepSeek V4 Pro CLI там же |
| `nqx-flash` | DeepSeek V4 Flash (cheap) там же |
| `nqx-codex` | OpenAI Codex CLI там же |
| `nqx-trio` | tmux 3-pane: claude + deepseek + codex |
| `nqx-audit` | one-shot: один промпт → все CLI параллельно |

## Примеры

Интерактивно:
```bash
nqx-claude            # как обычный claude, но cwd = проект
nqx-deepseek          # то же для DeepSeek
DEEPSEEK_MODEL=deepseek-v4-flash nqx-deepseek    # форсировать flash
```

Параллельно (батч):
```bash
nqx-audit architecture                       # 4 CLI работают на 1 промпт
nqx-audit correctness --only deepseek        # только DeepSeek
nqx-audit --all                              # каждый промпт × каждая CLI
nqx-audit --list                             # показать доступные промпты
```

Все ответы — в `audits/results/<cli>-<prompt>-<timestamp>.md`.
Логи (stderr + exit codes) — в `audits/logs/`.

## Куда смотреть после прогона

```bash
ls -lt audits/results/ | head
cat audits/results/$(ls -t audits/results/ | head -1)
```
