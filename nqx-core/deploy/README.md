# Deploying NQX-Server

A small HTTP API around the NautilusQuant pipeline:
- **CPU backend**: pure-numpy via the NQX emulator (correctness reference, slow)
- **GPU backend**: PyTorch + (optional) Triton, using the upstream `nautilus_triton.py`

## Run locally (CPU)

```bash
pip install -r requirements.txt
uvicorn server.api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
bash deploy/test_api.sh http://localhost:8000
```

## Docker

```bash
# CPU image (small, no GPU needed)
docker build -f Dockerfile.cpu -t nqx-server:cpu .
docker run --rm -p 8000:8000 nqx-server:cpu

# GPU image (uses NVIDIA Container Toolkit; needs CUDA-capable host)
docker build -t nqx-server:gpu .
docker run --rm --gpus all -p 8000:8000 nqx-server:gpu
```

### Multi-arch CPU image (amd64 + arm64)

`Dockerfile.cpu` is multi-arch ready (uses `--platform=$TARGETPLATFORM`).
Build with `docker buildx`:

```bash
# 1) one-time setup of a multi-arch builder
docker buildx create --name nqx-builder --use --bootstrap

# 2) build & push (Docker Hub / GHCR)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.cpu \
  -t ghcr.io/<you>/nqx-server:cpu \
  --push .

# 3) local single-arch (without push) — pick one platform:
docker buildx build --platform linux/arm64 -f Dockerfile.cpu \
  -t nqx-server:cpu-arm64 --load .
```

The image's `NQX_TARGETARCH` env var reflects the build arch (`amd64` / `arm64`)
so runtime code can branch if needed.

## docker-compose

```bash
docker compose --profile cpu up      # CPU only
docker compose --profile gpu up      # GPU (auto-detects CUDA)
```

## Environment variables

| Name | Default | Effect |
|---|---|---|
| `NQX_BACKEND` | `auto` | `auto` / `cpu` / `gpu` |
| `NQX_DIM` | `128` | Vector dimension |
| `NQX_BITS` | `3` | Quantization bits |

## Endpoints

- `GET /` — landing HTML
- `GET /docs` — Swagger UI (auto-generated)
- `GET /health`, `/info`
- `POST /encode`, `/decode`
- `POST /benchmark`, `/verify`

## Vast.ai

See [vastai.md](vastai.md).
