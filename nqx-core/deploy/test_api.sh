#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-http://localhost:8000}"
echo "Testing $HOST"

echo "--- /health ---"
curl -fsS "$HOST/health" | python3 -m json.tool

echo "--- /info ---"
curl -fsS "$HOST/info" | python3 -m json.tool

echo "--- /verify ---"
curl -fsS -X POST "$HOST/verify" | python3 -m json.tool

echo "--- /benchmark (4096 vectors, dim=128) ---"
curl -fsS -X POST "$HOST/benchmark" \
  -H 'content-type: application/json' \
  -d '{"n_vectors":4096,"dim":128,"bits":3}' | python3 -m json.tool

echo "--- /encode + /decode roundtrip (4 vectors, dim=128) ---"
python3 - "$HOST" <<'PY'
import json, sys, base64, urllib.request
import numpy as np
host = sys.argv[1]

x = np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32).tolist()
req = json.dumps({"vectors": x}).encode()
r = urllib.request.Request(host + "/encode", req, {"content-type": "application/json"})
enc = json.loads(urllib.request.urlopen(r).read())
print("encode_ms:", enc["stats"]["encode_ms"])
print("compression:", enc["stats"]["compression_ratio"])

dec_req = {
    "packed_b64": enc["packed_b64"], "sign_b64": enc["sign_b64"],
    "mins": enc["mins"], "maxs": enc["maxs"],
    "n": enc["n"], "dim": enc["dim"], "bits": enc["bits"],
}
r = urllib.request.Request(host + "/decode",
    json.dumps(dec_req).encode(), {"content-type": "application/json"})
dec = json.loads(urllib.request.urlopen(r).read())
y = np.asarray(dec["vectors"])
x_orig = np.asarray(x)
rmse = float(np.sqrt(((x_orig - y)**2).mean()))
print(f"decode_ms: {dec['decode_ms']:.2f}")
print(f"roundtrip RMSE: {rmse:.4f}")
PY

echo
echo "All tests OK"
