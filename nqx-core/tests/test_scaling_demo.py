import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from demos.scaling_demo import kv_cache_bytes_fp16, kv_cache_bytes_nqx, run


def test_kv_compression_ratio_is_four():
    fp16 = kv_cache_bytes_fp16(80, 64, 128, 128 * 1024)
    nqx = kv_cache_bytes_nqx(80, 64, 128, 128 * 1024, bits=3)
    assert abs(fp16 / nqx - 4.0) < 1e-9


def test_run_returns_expected_keys():
    r = run("Llama-3-70B")
    for k in ("kv_bytes_fp16", "kv_bytes_nqx", "h100_count", "nqx_count", "savings_ratio"):
        assert k in r
    assert r["compression_ratio"] > 3.9
    assert r["savings_ratio"] > 1.0


def test_405b_kv_bigger_than_70b():
    r70 = run("Llama-3-70B")
    r405 = run("Llama-3-405B")
    assert r405["kv_bytes_fp16"] > r70["kv_bytes_fp16"]
