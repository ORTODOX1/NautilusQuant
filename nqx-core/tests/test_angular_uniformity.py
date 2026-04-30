import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from bench.angular_uniformity import (
    fit_slope_log,
    kolmogorov_smirnov,
    measure,
    phi_angles,
)


def test_phi_discrepancy_below_random():
    n = 1024
    phi_ks = measure("phi", n)["ks"]
    rand_ks = np.mean([measure("random", n, seed=s)["ks"] for s in range(8)])
    assert phi_ks < rand_ks, f"phi {phi_ks} not lower than random {rand_ks}"


def test_phi_slope_close_to_minus_one():
    ns = [64, 256, 1024, 4096]
    phi_ks = [measure("phi", n)["ks"] for n in ns]
    slope = fit_slope_log(np.array(ns), np.array(phi_ks))
    assert slope < -0.8, f"phi slope {slope} should be close to -1"


def test_random_slope_close_to_minus_half():
    ns = [64, 256, 1024, 4096]
    rng = np.random.default_rng(0)
    rand_ks = []
    for n in ns:
        seeds = [int(rng.integers(0, 10**9)) for _ in range(8)]
        rand_ks.append(float(np.mean([measure("random", n, seed=s)["ks"] for s in seeds])))
    slope = fit_slope_log(np.array(ns), np.array(rand_ks))
    assert -0.7 < slope < -0.3, f"random slope {slope} should be near -0.5"


def test_phi_angles_in_range():
    a = phi_angles(64)
    assert np.all(a >= 0) and np.all(a < 2 * np.pi)
    assert kolmogorov_smirnov(a) < 0.05
