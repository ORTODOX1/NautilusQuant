import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def _make_data(n=200, d=128, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32) * 0.5
    for col in (0, 15, 31, 63, 95, 127):
        if col < d:
            mask = rng.random(n) < 0.75
            x[mask, col] = rng.standard_normal(int(mask.sum())).astype(np.float32) * 30.0
    return x


def test_encode_decode_returns_same_shape():
    core = NQXCore(NQXConfig(dim=128))
    x = _make_data(100, 128, seed=1)
    enc = core.encode(x)
    dec = core.decode(enc)
    assert dec.reconstructed.shape == x.shape


def test_encode_packed_size_is_4x_compression():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    x = _make_data(64, 128, seed=2)
    enc = core.encode(x)
    raw_bytes = x.size * 2
    packed_bytes = len(enc.packed_bytes)
    ratio = raw_bytes / packed_bytes
    assert ratio >= 3.9, f"compression too low: {ratio:.3f}x"
    assert ratio <= 4.1, f"compression too high (suspicious): {ratio:.3f}x"


def test_encode_polar_norm_preservation_against_reference_math():
    core = NQXCore(NQXConfig(dim=128))
    x = _make_data(32, 128, seed=3)
    rotated = core.forward_rotation(x)
    norm_in = np.linalg.norm(x, axis=-1)
    norm_rot = np.linalg.norm(rotated, axis=-1)
    err = np.abs(norm_in - norm_rot).max()
    assert err < 5e-3


def test_decode_reconstruction_mse_bounded():
    core = NQXCore(NQXConfig(dim=128, bits=3))
    x = _make_data(200, 128, seed=4)
    enc = core.encode(x)
    dec = core.decode(enc)
    mse = ((x - dec.reconstructed) ** 2).mean()
    assert mse < 50.0, f"MSE too high: {mse}"
    assert np.isfinite(mse)


def test_quantization_indices_in_range():
    core = NQXCore(NQXConfig(dim=128, bits=3))
    x = _make_data(50, 128, seed=5)
    enc = core.encode(x)
    assert enc.quantized_indices.min() >= 0
    assert enc.quantized_indices.max() <= (2**3) - 1


def test_qjl_sign_bit_is_binary():
    core = NQXCore(NQXConfig(dim=128, bits=3))
    x = _make_data(30, 128, seed=6)
    enc = core.encode(x)
    unique = set(np.unique(enc.sign_bits).tolist())
    assert unique <= {0, 1}, f"sign bits not binary: {unique}"
