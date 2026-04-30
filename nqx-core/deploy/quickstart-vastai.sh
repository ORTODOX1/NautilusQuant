#!/usr/bin/env bash
# quickstart-vastai.sh — one-command deploy NQX-server to vast.ai with GPU.
#
# Prerequisites:
#   1. Account on vast.ai with credit (~$5 starter is enough)
#   2. `vastai` CLI installed: pip install --upgrade vastai
#   3. API key set: vastai set api-key YOUR_API_KEY
#   4. Docker Hub / GHCR account, image already pushed:
#        docker build -t YOURUSER/nqx-server:gpu .
#        docker push YOURUSER/nqx-server:gpu
#
# Usage:
#   IMAGE=YOURUSER/nqx-server:gpu bash deploy/quickstart-vastai.sh
#   IMAGE=YOURUSER/nqx-server:gpu GPU=RTX_4090 bash deploy/quickstart-vastai.sh
#   IMAGE=YOURUSER/nqx-server:gpu GPU=H100_SXM MAX_DPH=2.5 bash deploy/quickstart-vastai.sh

set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE=YOURUSER/nqx-server:gpu (built and pushed)}"
GPU="${GPU:-RTX_5090}"
MAX_DPH="${MAX_DPH:-1.0}"
DISK_GB="${DISK_GB:-30}"
PORT="${PORT:-8000}"

echo "═══════════════════════════════════════════════════════════"
echo " NQX-Server → vast.ai quickstart"
echo " Image:    $IMAGE"
echo " GPU:      $GPU (max \$${MAX_DPH}/hr)"
echo " Disk:     ${DISK_GB} GB"
echo " Port:     $PORT"
echo "═══════════════════════════════════════════════════════════"
echo

if ! command -v vastai >/dev/null 2>&1; then
    echo "ERROR: vastai CLI not installed."
    echo "  pip install --upgrade vastai && vastai set api-key YOUR_KEY"
    exit 1
fi

echo "[1/5] Searching for $GPU offers cheaper than \$${MAX_DPH}/hr..."
OFFERS=$(vastai search offers \
    "gpu_name=${GPU} dph<${MAX_DPH} cuda_vers>=12.0 inet_down>500 disk_space>${DISK_GB}" \
    -o 'dph+' \
    --raw 2>/dev/null | head -200 || true)

OFFER_ID=$(echo "$OFFERS" | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    if isinstance(data, list) and data:
        print(data[0]['id'])
except Exception:
    pass
" 2>/dev/null || true)

if [ -z "$OFFER_ID" ]; then
    echo "ERROR: no $GPU offers under \$${MAX_DPH}/hr right now. Try:"
    echo "  GPU=RTX_4090 MAX_DPH=0.6 bash $0"
    echo "  GPU=H100_SXM MAX_DPH=2.5 bash $0"
    echo "  vastai search offers 'gpu_name=$GPU' -o 'dph+'   # browse manually"
    exit 1
fi

echo "  → picked offer #$OFFER_ID"

echo "[2/5] Creating instance..."
INSTANCE_RAW=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --env "-p ${PORT}:${PORT} -e NQX_BACKEND=auto -e NQX_DIM=128 -e NQX_BITS=3" \
    --raw 2>/dev/null)

INSTANCE_ID=$(echo "$INSTANCE_RAW" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
print(data.get('new_contract', data.get('instance_id', '')))
" 2>/dev/null || true)

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: failed to create instance. Output:"
    echo "$INSTANCE_RAW"
    exit 1
fi

echo "  → instance #$INSTANCE_ID created"

echo "[3/5] Waiting for instance to be running (this can take 30-90 sec)..."
for i in $(seq 1 30); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d.get('actual_status',''))" \
        2>/dev/null || echo "")
    if [ "$STATUS" = "running" ]; then
        echo "  → running"
        break
    fi
    echo "  ... status=$STATUS (attempt $i/30)"
    sleep 5
done

echo "[4/5] Resolving public URL..."
INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
HOST=$(echo "$INFO" | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d.get('ssh_host',''))" 2>/dev/null)
PORT_MAP=$(echo "$INFO" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
ports = d.get('ports', {}) or {}
print(ports.get('${PORT}/tcp', [{}])[0].get('HostPort', ''))
" 2>/dev/null || echo "")

if [ -z "$HOST" ] || [ -z "$PORT_MAP" ]; then
    echo "  ⚠ Could not auto-resolve URL. Run: vastai show instances"
    URL=""
else
    URL="http://${HOST}:${PORT_MAP}"
    echo "  → URL: $URL"
fi

echo "[5/5] Smoke test..."
if [ -n "$URL" ]; then
    sleep 5
    if [ -f "$(dirname "$0")/smoke.sh" ]; then
        bash "$(dirname "$0")/smoke.sh" "$URL" || echo "  ⚠ smoke test reported issues; check $URL/health manually"
    else
        for i in $(seq 1 12); do
            if curl -fsS "$URL/health" >/dev/null 2>&1; then
                echo "  ✓ /health OK"
                curl -fsS "$URL/health" | python3 -m json.tool | head -10
                break
            fi
            echo "  ... waiting for /health (attempt $i/12)"
            sleep 5
        done
    fi
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo " Done. Instance #$INSTANCE_ID running."
echo " URL:  ${URL:-<run: vastai show instance $INSTANCE_ID>}"
echo
echo " Test it:"
echo "   curl ${URL:-http://HOST:PORT}/health"
echo "   curl -X POST ${URL:-http://HOST:PORT}/benchmark \\"
echo "        -H 'content-type: application/json' \\"
echo "        -d '{\"n_vectors\":4096,\"dim\":128,\"bits\":3}'"
echo
echo " Stop / destroy:"
echo "   vastai stop instance $INSTANCE_ID    # pause"
echo "   vastai destroy instance $INSTANCE_ID # remove"
echo "═══════════════════════════════════════════════════════════"
