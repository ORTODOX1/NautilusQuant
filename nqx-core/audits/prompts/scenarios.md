# Scenarios / Production-readiness — для DeepSeek-flash

Задачи которые превратят проект в **полностью deployment-ready**: когда зальёшь
на сервер — всё уже проверено через сценарии, мониторинг, chaos-тесты, фикстуры,
отладчики и smoke-тесты.

Бери первый `[ ]`, делай минимальный diff, `pytest tests -q` после каждой,
ставь `[x]`. Не лезь в `T*` (это Claude/heavy). Без docstrings.

## Перед стартом

- `python -m pytest tests -q` — должно зеленеть
- `routine.md` уже завершён или почти — ты сделал R1-R28

## Scenarios — реалистичные нагрузки и edge cases

- [x] **S1. Multi-turn chat session.** `tests/scenarios/test_chat_session.py` —
  эмуляция KV-cache растущего на 5 turns × 200 токенов. Encode на каждом turn,
  проверить что cumulative RMSE не drift'ует > 2× от single-turn.

- [x] **S2. Long-context boundary.** `tests/scenarios/test_long_context.py` —
  encode batch с seq_len ∈ {1K, 4K, 16K, 64K}, dim=128. Проверить cycles
  растут линейно, RMSE стабилен. Если 64K не лезет в память — пометить skip.

- [x] **S3. Variable batch sizes.** `tests/scenarios/test_batch_variability.py` —
  batch ∈ {1, 8, 64, 256, 1024, 4096}. Verify: (a) compression ratio всегда 4.00×,
  (b) RMSE не зависит от batch (с точностью ±10%).

- [x] **S4. Mixed-modality KV.** `tests/scenarios/test_multimodal.py` —
  смешанный батч: 50% текстовых KV (нормальное распр), 50% «визуальных»
  (с большими outlier dims). Verify что Sub-bit allocation адаптируется
  (используя `SubBitUnit`).

- [x] **S5. Edge inputs.** `tests/scenarios/test_edge_inputs.py` — encode на:
  empty `(0, 128)`, single vector `(1, 128)`, all-zeros, all-NaN (должно raise),
  all-Inf (raise), constant 1.0 (compression mse ≈ 0), max FP16 (~65k).

## Fixtures — реалистичные KV-данные

- [x] **S6. Realistic KV generator.** `tests/fixtures/realistic_kv.py` —
  генерит KV-cache, который выглядит как реальный Llama 3 (heavy-tailed
  distribution, ~6 outlier dimensions per head, bimodal channels). Без torch.
  Сохраняй сэмплы в `tests/fixtures/data/llama3_like_seq256.npy` и т.д.

- [x] **S7. Golden reference snapshots.** `tests/fixtures/golden/` — фиксированные
  входы (seed=42) с заранее посчитанными encode-выходами в `.npz`. Регрессионный
  тест `tests/test_golden_regression.py` сравнивает текущий encode с snapshot.
  Если хеши drift'ят — тест красный (bit-exact protection).

- [x] **S8. Adversarial KV.** `tests/fixtures/adversarial.py` — генерит KV который
  максимизирует quantization error: спайки в нескольких dim'ах одновременно,
  периодика на golden angle (попытка resonance с LUT). Тест: даже на adversarial
  RMSE остаётся ниже катастрофического threshold.

## Monitoring / observability — для production-deploy

- [x] **S9. Request logging middleware.** `server/middleware.py` — FastAPI middleware,
  логирует JSON-line: `{request_id, ts, route, latency_ms, status, payload_bytes}`.
  Файл `server/logs/access.jsonl` (rotation в 10 MB). Тест: 5 запросов → 5 lines.

- [x] **S10. Prometheus /metrics endpoint.** `server/metrics.py` — Counter/Histogram:
  `nqx_encode_total`, `nqx_decode_total`, `nqx_encode_latency_ms` (histogram),
  `nqx_errors_total{type=...}`. Зарегистрируй endpoint `/metrics` в `api.py`,
  Prometheus-text format (без библиотек, пиши руками).

- [x] **S11. Deep health check.** `server/health_deep.py` — endpoint `/health/deep`:
  - SHA-256 хеш ROM_LUT (для tamper-detection)
  - check `forward(eye)`.T @ `forward(eye)` ≈ I (T^T·T self-test, < 1e-5)
  - last 10 ошибок (if any)
  - uptime
  - backend type + device

- [x] **S12. Structured error responses.** `server/errors.py` — единый формат ошибок:
  `{"error_type": "BadShape|EncodeFailure|...", "detail": "...", "request_id": "..."}`.
  Apply ко всем `HTTPException` в `api.py`. Тесты: invalid shape → 400 + правильный
  JSON.

## Chaos / stress / fault injection

- [x] **S13. OOM behaviour.** `tests/chaos/test_oom.py` — encode с очень большим
  батчем (10K vectors dim=512). Должно либо успешно отработать, либо raise
  `MemoryError` с понятным message — не silent fail.

- [x] **S14. Corrupt payload.** `tests/chaos/test_corrupt_payload.py` — POST /decode с:
  invalid base64, wrong byte length для shape, garbage в `mins`/`maxs`,
  `bits=99`. Должны быть 4xx с структурированными ошибками (использует S12).

- [x] **S15. Concurrent requests.** `tests/chaos/test_concurrent.py` — 32 параллельных
  encode через `httpx.AsyncClient`. Verify: нет race conditions, нет утечек
  памяти, все 32 ответа правильные. (Запуск опциональный — если сервер не подняли,
  skip.)

- [x] **S16. Slow client.** `tests/chaos/test_slow_client.py` — отправь request с
  очень медленным upload (chunked, 10 ms между chunks). Сервер должен либо
  timeout с 408, либо обработать. Не зависнуть навсегда.

## Debugging tools

- [x] **S17. nqx-debug CLI.** `tools/cli/nqx-debug` (bash) +
  `tools/debug/inspect_encode.py` (Python). Принимает `.npy` файл с одним
  вектором, прогоняет через encode пошагово, печатает на каждом stage:
  - input vector (first 8 elems)
  - после L1, L2, L3 rotation (first 8 elems каждый раз)
  - polar (r, θ pairs)
  - quantized indices
  - sign bits
  - packed bytes (hex dump)
  Линкуй в `~/.local/bin/nqx-debug`.

- [x] **S18. Replay harness.** `tools/debug/replay.py` — берёт log из S9
  (`access.jsonl`), повторяет каждый encode локально (если payload сохранён),
  сравнивает с записанным результатом. Используется для регрессионного
  testing после релиза.

- [x] **S19. Failure snapshots.** При любой ошибке encode — автоматически дампить
  `audits/snapshots/error-<ts>.npz` с input + traceback + version + LUT hash.
  Реализуй в `nqx/cpu.py::NQXCore.encode` через try/except → snapshot → re-raise.
  Это для post-mortem отладки.

## Smoke / deploy validation

- [x] **S20. Post-deploy smoke test.** `deploy/smoke.sh` — bash скрипт:
  1. wait_for_health (poll `/health` до 200 OK или 30 sec timeout)
  2. POST /encode на known input (golden из S7)
  3. сравнить response с golden expected
  4. POST /benchmark — verify throughput > 1000 vec/s
  5. POST /verify — verify orthogonality < 1e-5
  Exit 0 если всё OK, 1 + diagnostic иначе. Используется в CI после `docker run`.

- [x] **S21. Load test.** `deploy/load_test.py` — `locust`-подобный (но без deps,
  pure asyncio + httpx если есть, иначе `urllib`). 100 одновременных клиентов,
  50% encode 50% decode, 60 сек. Замеряй p50/p95/p99 latency, throughput.
  Дамп `deploy/load_test_results.md`.

## Demo notebooks

- [x] **S22. Jupyter intro.** `demos/notebooks/01_intro.ipynb` — без torch:
  - что такое KV cache (картинка из документации)
  - запустить NQXCore.encode на 16 векторах
  - показать что compression == 4×, RMSE acceptable
  - визуализация (ASCII или matplotlib если есть)

- [x] **S23. Comparison notebook.** `demos/notebooks/02_compare.ipynb` — рядом
  TurboQuant и NautilusQuant (юзает `demos/turboquant_emul.py` от Claude из D1).
  Показать таблицу + bar chart (если matplotlib есть, иначе ASCII).

- [x] **S24. Attention demo notebook.** `demos/notebooks/03_attention.ipynb` —
  juzает `AttentionUnit.dot_polar`, показывает что компрессия не ломает
  attention pattern (heat map до/после).

## Acceptance

| Что | Где |
|---|---|
| `pytest tests -q` всё ещё проходит (без regressions) | sanity |
| `bash deploy/smoke.sh http://localhost:8000` exit 0 | S20 |
| `tools/cli/nqx-debug fixture.npy` печатает stage-by-stage | S17 |
| `/metrics` endpoint показывает `nqx_encode_total{...} N` | S10 |
| `/health/deep` показывает hash LUT | S11 |
| Notebooks открываются и выполняются без ошибок | S22-S24 |
