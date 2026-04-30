"""S1: Multi-turn chat session — growing KV-cache, check RMSE drift."""

from __future__ import annotations

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def test_cumulative_rmse_does_not_drift():
    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    rng = np.random.default_rng(42)
    tokens_per_turn = 200
    n_turns = 5
    rmse_list = []

    for turn in range(n_turns):
        new_vecs = rng.standard_normal((tokens_per_turn, 128)).astype(np.float32)
        if turn == 0:
            cache = new_vecs
        else:
            cache = np.concatenate([cache, new_vecs], axis=0)

        enc = core.encode(cache)
        dec = core.decode(enc)
        rmse = float(np.sqrt(np.mean((cache - dec.reconstructed) ** 2)))
        rmse_list.append(rmse)

    baseline = rmse_list[0]
    worst = max(rmse_list)
    assert worst < 2.0 * baseline, f"RMSE drifted: turn0={baseline:.4f} worst={worst:.4f}"
