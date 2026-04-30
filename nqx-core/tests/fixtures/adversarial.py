"""S8: Adversarial KV — inputs designed to maximize quantization error."""

from __future__ import annotations

import numpy as np


def generate_adversarial_spikes(n_vectors: int, dim: int, rng: np.random.Generator | None = None):
    """Vectors with spikes in multiple dims to expand quantization range."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    spike_dims = rng.integers(0, dim, size=max(1, dim // 8))
    x[:, spike_dims] *= 50.0  # 50x outliers explode the min/max range
    return x


def generate_adversarial_golden_angle(n_vectors: int, dim: int):
    """Periodic pattern at golden-angle frequency to try resonance with LUT."""
    phi = (1 + 5 ** 0.5) / 2
    t = np.arange(n_vectors)[:, None] * np.arange(dim)[None, :]
    x = np.sin(t * 2 * np.pi / phi ** 2).astype(np.float32)
    return x


def test_adversarial_rmse_bounded():
    """Test from this module: even adversarial inputs stay below threshold."""
    from nqx.constants import NQXConfig
    from nqx.cpu import NQXCore

    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)

    spikes = generate_adversarial_spikes(16, 128, rng)
    golden = generate_adversarial_golden_angle(16, 128)

    for name, x in [("spikes", spikes), ("golden_angle", golden)]:
        enc = core.encode(x)
        dec = core.decode(enc)
        rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
        assert rmse < 5.0, f"{name}: RMSE {rmse:.4f} exceeds catastrophic threshold"
