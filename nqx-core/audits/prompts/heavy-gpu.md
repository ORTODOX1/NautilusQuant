# Heavy GPU tasks — выполнять на vast.ai-инстансе

Эти задачи требуют CUDA + `torch` + `transformers` + `vllm` + `triton`.
Локально не делать — перенесено сюда из `heavy.md` для ясности.

## Перед стартом (на GPU-ноде)

```bash
pip install torch transformers vllm triton accelerate
python -m pytest tests -q   # должно зеленить (131+)
nvidia-smi                  # должна быть GPU
```

## Задачи

- [ ] **T10. HuggingFace KV-hook.** `integrations/hf_kv_hook.py` — кастомный `Cache` класс для `transformers`, который пакует KV через `NQXCore.encode` и распаковывает на каждом attention forward. Тест на Pythia-160M (или Llama-3.2-1B если влезает): perplexity на WikiText-103 не должна расти > 5% относительно FP16 baseline.

- [ ] **T11. vLLM adapter.** `integrations/vllm_kvquant.py` — KV-cache quantization plugin для vLLM. Регистрация в `kv_cache_quantization_methods`. Bench: tokens/s vs FP16 baseline на одном GPU, чтобы compression 4× давала меньше HBM трафика → больше TPS.

- [ ] **T12. Triton kernel в server/.** В `server/backends.py::GPUBackend` сейчас используется `NautilusQuantPyTorch` reference. Замени на `NautilusQuantTriton` (из `nautilus_triton.py`) когда `torch.cuda.is_available()`. Бенч encode 4096 vec dim=128: цель < 1 ms на RTX 5090, < 0.5 ms на B200.

## Workflow

1. Подними vast.ai инстанс по `deploy/vastai.md`
2. ssh в инстанс, `git clone` репо
3. Установи GPU-deps выше
4. Бери первый `[ ]`, делай, тест, отметь `[x]`
5. Когда закроешь все три — коммит и push, потом экспортируй results back через scp.

## Acceptance

- T10: `pytest integrations/test_hf_hook.py` проходит, perplexity дельта ≤ 5%
- T11: vLLM запускается с `--kv-cache-quant nqx`, throughput ≥ FP16 baseline
- T12: Triton kernel encode/decode bit-exact с PyTorch reference, ≥ 5× быстрее
