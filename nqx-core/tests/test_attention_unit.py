import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.functional_units import AttentionUnit


@pytest.mark.parametrize("dim", [64, 128])
def test_attn_dot_vs_full_decode(dim):
    cfg = NQXConfig(dim=dim, bits=3)
    core = NQXCore(cfg)
    attn = AttentionUnit(cfg)
    rng = np.random.default_rng(42)

    q = rng.standard_normal((5, dim)).astype(np.float32)
    k = rng.standard_normal((5, dim)).astype(np.float32)

    enc_q = core.encode(q)
    enc_k = core.encode(k)

    deq_q, _ = core.qu.dequantize(enc_q.quantized_indices, enc_q.mins, enc_q.maxs, cfg.bits)
    deq_k, _ = core.qu.dequantize(enc_k.quantized_indices, enc_k.mins, enc_k.maxs, cfg.bits)

    fast, _ = attn.dot_polar(deq_q, deq_k)

    cart_q, _ = core.pu.from_polar(deq_q)
    cart_k, _ = core.pu.from_polar(deq_k)
    cart_q, _ = core.gu.apply_layer(cart_q, 2, inverse=True)
    cart_q, _ = core.gu.apply_layer(cart_q, 1, inverse=True)
    cart_q, _ = core.gu.apply_layer(cart_q, 0, inverse=True)
    cart_k, _ = core.gu.apply_layer(cart_k, 2, inverse=True)
    cart_k, _ = core.gu.apply_layer(cart_k, 1, inverse=True)
    cart_k, _ = core.gu.apply_layer(cart_k, 0, inverse=True)

    ref = cart_q @ cart_k.T

    rmse = float(np.sqrt(((fast - ref) ** 2).mean()))
    assert rmse < 1e-3, f"dim={dim} rmse={rmse:.6f}"
