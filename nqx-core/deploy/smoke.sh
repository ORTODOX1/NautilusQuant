#!/usr/bin/env bash
# S20: Post-deploy smoke test — run against a running NQX server.
# Usage: bash deploy/smoke.sh [url]
#   url defaults to http://localhost:8000
set -euo pipefail

URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

echo "=== NQX Smoke Test ==="
echo "Target: $URL"
echo ""

# 1. Health check with timeout
echo "--- 1. Health check ---"
for i in $(seq 1 30); do
    if curl -sf "$URL/health" > /dev/null 2>&1; then
        ok "/health → 200"
        break
    fi
    if [ "$i" -eq 30 ]; then
        fail "/health — timeout after 30s"
    fi
    sleep 1
done

# 2. POST /encode with known input
echo "--- 2. Encode golden input ---"
# Fixed seed input: 4 vectors dim=128
VECTORS='{"vectors":'
# Generate 4x128 matrix of seed=42 values (Python snippet embeds the literal)
# We use a curl-friendly small payload
VECTORS+='[[0.1257,-0.1321,0.6404,0.1049,-0.5357,0.3616,1.3040,0.9471,-0.7036,-0.6103,-0.4066,-0.5996,0.3588,-0.4642,-0.2926,2.1486,-1.7142,-0.7421,0.2054,-0.2023,-1.2869,-0.1833,0.4091,0.1174,0.7910,-1.5100,-0.3683,-0.1133,-0.4572,-0.8454,-0.1043,-0.6504,-0.4708,0.4081,-0.0078,-0.3474,-0.2312,0.8585,-0.6591,0.0912,0.6414,1.4138,0.3221,-0.0590,0.2064,-0.6242,-0.3269,0.3564,-1.3766,2.0759,-0.2867,-0.4203,1.0217,0.1270,1.0003,-0.9665,-0.3264,-0.6829,0.4282,0.0800,0.1107,0.3145,-1.0658,-0.3551,0.1415,0.0370,-0.9720,-0.0286,-0.7045,0.2477,0.2583,0.9781,-0.2019,-0.1064,-0.0344,0.2434,-0.2435,0.4227,-0.6396,-0.4775,0.6886,0.2374,-0.3676,0.0558,1.5331,0.4475,0.2058,-0.1170,1.5401,-0.6541,0.2984,-1.5281,0.2304,0.0438,0.7934,-0.0674,0.8772,-0.5136,0.0574,-1.7242,0.4409,-0.1273,-0.7534,-1.6182,-0.2653,-0.2294,0.0455,-0.1090,0.1456,1.0869,0.2662,0.4472,-0.2039,0.2999,0.7510,0.0301,-1.1600,-1.2235,-0.4997,0.0820,0.4486,0.3625,-1.0154,1.2316,1.4015,-0.1933,0.6961,-1.1879,2.0590,0.3786,0.7060,-0.9105,1.0491,-0.4919,0.1587,-0.9910,1.0344,0.2442,-0.9150,0.5787,-0.1560,-0.4560,0.7312,-0.7738,1.1792,-0.4973,0.3219,-0.4582,0.9704,-0.9140,1.0149,-2.1971,-0.3806,0.1685,0.1926,0.5960,0.2975,-0.5688,1.3275,0.8095,2.2190,-1.8087,-0.5240,0.1104,0.3270,-1.6737,-0.6330,1.9361,0.6201,0.4049,-0.2390,-1.3646,0.0567,0.5832,-0.0447,-0.5728,0.2827,1.8255]]'
VECTORS+=',"bits":3}'

ENC_RESP=$(curl -sf -X POST "$URL/encode" \
    -H "Content-Type: application/json" \
    -d "$VECTORS" 2>&1) && ok "/encode → 200" || fail "/encode — curl failed: $ENC_RESP"

if [ -n "$ENC_RESP" ]; then
    # Verify response has expected structure
    if echo "$ENC_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'packed_b64' in d, 'missing packed_b64'
assert 'stats' in d, 'missing stats'
assert d['stats']['compression_ratio'] > 3.0, f'low compression: {d[\"stats\"][\"compression_ratio\"]}'
print('OK')
" 2>&1; then
        ok "/encode response valid"
    else
        fail "/encode response invalid"
    fi
fi

# 3. POST /benchmark
echo "--- 3. Benchmark ---"
BENCH_RESP=$(curl -sf -X POST "$URL/benchmark" \
    -H "Content-Type: application/json" \
    -d '{"n_vectors":1024,"dim":128,"bits":3,"seed":0}' 2>&1) && ok "/benchmark → 200" || fail "/benchmark — curl failed"

if [ -n "$BENCH_RESP" ]; then
    THRPUT=$(echo "$BENCH_RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['throughput_vec_per_sec'])")
    if python3 -c "assert $THRPUT > 1000" 2>/dev/null; then
        ok "/benchmark throughput: $THRPUT vec/s (> 1000)"
    else
        fail "/benchmark throughput: $THRPUT vec/s (expected > 1000)"
    fi
fi

# 4. POST /verify
echo "--- 4. Verify ---"
VERIFY_RESP=$(curl -sf -X POST "$URL/verify" \
    -H "Content-Type: application/json" \
    -d '{}' 2>&1) && ok "/verify → 200" || fail "/verify — curl failed"

if [ -n "$VERIFY_RESP" ]; then
    ORTH_ERR=$(echo "$VERIFY_RESP" | python3 -c "import sys,json;print(json.load(sys.stdin)['orthogonality_err'])")
    if python3 -c "assert $ORTH_ERR < 1e-5" 2>/dev/null; then
        ok "/verify orthogonality: $ORTH_ERR (< 1e-5)"
    else
        fail "/verify orthogonality: $ORTH_ERR (expected < 1e-5)"
    fi
fi

echo ""
echo "=== Result: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then exit 1; fi
