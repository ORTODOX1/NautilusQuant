import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from bench.linear_quant import lloyd_max_quant, measure, uniform_quant


def test_uniform_and_lloyd_match_on_uniform_input():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(2048, 16)).astype(np.float32)
    uni = uniform_quant(x, bits=4)
    llm = lloyd_max_quant(x, bits=4)
    rmse_uni = float(np.sqrt(((x - uni) ** 2).mean()))
    rmse_llm = float(np.sqrt(((x - llm) ** 2).mean()))
    delta = (rmse_uni - rmse_llm) / rmse_llm
    assert delta < 0.10, f"On uniform inputs the gap should be small, got {delta:.3f}"


def test_lloyd_better_than_uniform_on_gaussian():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2048, 16)).astype(np.float32)
    uni = uniform_quant(x, bits=3)
    llm = lloyd_max_quant(x, bits=3)
    rmse_uni = float(np.sqrt(((x - uni) ** 2).mean()))
    rmse_llm = float(np.sqrt(((x - llm) ** 2).mean()))
    assert rmse_llm < rmse_uni


def test_measure_returns_expected_keys():
    row = measure("phi", dim=64, bits=3, n_vec=128, seed=0)
    for k in ("rotation", "dim", "bits", "rmse_uniform", "rmse_lloyd_max", "delta_pct"):
        assert k in row
    assert row["rmse_lloyd_max"] >= 0
    assert row["rmse_uniform"] >= 0
