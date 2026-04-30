import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nqx.constants import NQXConfig
from demos.turboquant_emul import (
    encode,
    decode,
    encode_cycles,
    encode_energy_pj,
    random_orthogonal,
    rmse,
    state_size_bytes,
)


def test_random_orthogonal_is_orthogonal():
    rng = np.random.default_rng(0)
    T = random_orthogonal(rng, 32)
    err = float(np.abs(T.T @ T - np.eye(32)).max())
    assert err < 1e-4


def test_encode_decode_roundtrip_below_one():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((256, 64)).astype(np.float32)
    enc = encode(x, bits=3, seed=1)
    back = decode(enc)
    assert back.shape == x.shape
    assert rmse(x, back) < 1.0


def test_encode_includes_random_state():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    e1 = encode(x, seed=1)
    e2 = encode(x, seed=2)
    assert not np.array_equal(e1.rotation, e2.rotation)


def test_cycles_dominated_by_prng():
    cfg = NQXConfig(dim=128)
    cycles = encode_cycles(cfg, n_vec=64)
    prng = 4 * 128 * 128
    assert cycles > prng


def test_energy_grows_with_dim():
    cfg64 = NQXConfig(dim=64)
    cfg128 = NQXConfig(dim=128)
    e64 = encode_energy_pj(cfg64, 1024)
    e128 = encode_energy_pj(cfg128, 1024)
    assert e128["total_pj"] > e64["total_pj"]


def test_state_size_is_dim_squared_fp16():
    assert state_size_bytes(NQXConfig(dim=128)) == 128 * 128 * 2
