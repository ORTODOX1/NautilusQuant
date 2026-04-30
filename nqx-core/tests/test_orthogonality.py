import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def test_rotation_matrix_orthogonal_dim128():
    core = NQXCore(NQXConfig(dim=128))
    T = core.rotation_matrix()
    err = np.abs(T.T @ T - np.eye(128)).max()
    assert err < 1e-5, f"orthogonality error: {err}"


def test_norm_preservation():
    rng = np.random.default_rng(0)
    core = NQXCore(NQXConfig(dim=128))
    x = rng.standard_normal((64, 128)).astype(np.float32)
    y = core.forward_rotation(x)
    diff = np.abs(np.linalg.norm(x, axis=-1) - np.linalg.norm(y, axis=-1)).max()
    assert diff < 1e-3, f"norm preservation error: {diff}"


def test_dot_product_preservation():
    rng = np.random.default_rng(1)
    core = NQXCore(NQXConfig(dim=128))
    q = rng.standard_normal((10, 128)).astype(np.float32)
    k = rng.standard_normal((10, 128)).astype(np.float32)
    qr = core.forward_rotation(q)
    kr = core.forward_rotation(k)
    orig = (q * k).sum(axis=-1)
    rotated = (qr * kr).sum(axis=-1)
    err = np.abs(orig - rotated).max()
    assert err < 5e-3, f"dot product preservation error: {err}"


def test_roundtrip_no_quantization():
    rng = np.random.default_rng(2)
    core = NQXCore(NQXConfig(dim=128))
    x = rng.standard_normal((20, 128)).astype(np.float32)
    y = core.forward_rotation(x)
    x_back = core.inverse_rotation(y)
    rmse = float(np.sqrt(((x - x_back) ** 2).mean()))
    assert rmse < 1e-5, f"roundtrip RMSE: {rmse}"


def test_orthogonality_smaller_dims():
    for d in (16, 32, 64, 256):
        core = NQXCore(NQXConfig(dim=d))
        T = core.rotation_matrix()
        err = np.abs(T.T @ T - np.eye(d)).max()
        assert err < 1e-4, f"dim={d}: orthogonality err {err}"
