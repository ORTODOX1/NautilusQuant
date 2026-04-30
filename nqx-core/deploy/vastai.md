# Deploy NQX-Server on vast.ai

Steps (Linux/macOS terminal). End result: live HTTP API on a public URL,
validated NautilusQuant encode/decode running on real GPU.

## 0. One-time setup

```bash
# Sign up: https://vast.ai/console/billing/  (~ $5 starter credit, no card pre-charge)
pip install --upgrade vastai
vastai set api-key YOUR_API_KEY      # from https://vast.ai/console/cli/
```

## 1. Build & push the image

You need a Docker Hub (or GHCR) account.

```bash
cd /home/user/nautilusquantcore

# Test locally first (CPU image, fast):
docker build -f Dockerfile.cpu -t YOURUSER/nqx-server:cpu .
docker run --rm -p 8000:8000 YOURUSER/nqx-server:cpu &
bash deploy/test_api.sh http://localhost:8000
docker stop $(docker ps -q --filter ancestor=YOURUSER/nqx-server:cpu) || true

# Build the GPU image (large, ~6 GB; needs Docker buildx + emulation if no GPU on build host):
docker build -t YOURUSER/nqx-server:gpu .

# Push both:
docker login
docker push YOURUSER/nqx-server:cpu
docker push YOURUSER/nqx-server:gpu
```

## 2. Pick an instance on vast.ai

CLI search (find cheap RTX 4090 / 5090 / H100):

```bash
# RTX 4090, on-demand, NA region, > 24 GB RAM, > 50 GB disk:
vastai search offers \
  'gpu_name=RTX_4090 dph<0.6 inet_down>500 disk_space>50 cuda_vers>=12.0' \
  -o 'dph+'

# H100 80GB (best perf):
vastai search offers \
  'gpu_name=H100_SXM dph<2.5 inet_down>500 disk_space>80 cuda_vers>=12.4' \
  -o 'dph+'
```

Pick an offer ID from the first column.

## 3. Launch the container

```bash
OFFER_ID=12345678  # from step 2
IMAGE=YOURUSER/nqx-server:gpu

vastai create instance $OFFER_ID \
  --image $IMAGE \
  --disk 30 \
  --env '-p 8000:8000 -e NQX_BACKEND=auto -e NQX_DIM=128 -e NQX_BITS=3'
```

Or via the web UI:
1. https://vast.ai/console/create/
2. **Template**: `Custom` → fill in image: `YOURUSER/nqx-server:gpu`
3. **On-start script**: leave empty (CMD already in Dockerfile)
4. **Args / Docker Options**: `-p 8000:8000 -e NQX_BACKEND=auto`
5. Pick an offer with RTX 4090 or H100, click **Rent**

## 4. Get the public URL

```bash
vastai show instances
# look at "ssh_host" and the port mapping for 8000 (e.g. 1234)
# URL: http://<ssh_host>:<mapped_port>/
```

Or from the web UI: **Instances → click your instance → "Open Ports" → 8000**

## 5. Verify

```bash
URL=http://ssh4.vast.ai:12345     # replace with your real URL
bash deploy/test_api.sh $URL
```

Expected response from `/health`:
```json
{
  "status": "ok",
  "backend": "gpu-torch",
  "device": "cuda (NVIDIA GeForce RTX 4090)",
  "nqx_version": "1.0.0"
}
```

Expected from `/benchmark` on RTX 4090:
- `encode_ms`: ≈ 5–15 ms for 4096 vectors
- `throughput_vec_per_sec`: ~ 300K – 1M vec/s
- `compression_ratio`: ~ 4.0
- `roundtrip_rmse`: ≈ 0.1 – 0.5

## 6. Stop / teardown

```bash
vastai stop instance INSTANCE_ID    # pause (keeps disk, reduced cost)
vastai destroy instance INSTANCE_ID # remove completely
```

## Cost estimates (April 2026)

| GPU | $/hour spot | Encode 4096 vec/dim128 | Suitable for |
|---|---|---|---|
| RTX 4090 | $0.30–0.45 | ~10 ms | dev, demo |
| RTX 5090 | $0.50–0.80 | ~6 ms | demo, light prod |
| H100 80GB | $1.50–2.50 | ~3 ms | production batch |
| A100 40GB | $0.70–1.30 | ~5 ms | mid-tier |

For just spinning up and showing the demo: RTX 4090 spot is enough.
