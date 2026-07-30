ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r requirements-gpu.txt

RUN git clone --depth=1 https://github.com/hermandoronin/NautilusQuant.git /opt/nautilusquant
ENV PYTHONPATH=/app:/opt/nautilusquant

COPY nqx /app/nqx
COPY server /app/server
COPY programs /app/programs
COPY tests /app/tests
COPY run.py README.md ./

ENV NQX_BACKEND=auto NQX_DIM=128 NQX_BITS=3
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
