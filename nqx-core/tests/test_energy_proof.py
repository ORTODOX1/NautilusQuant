import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.energy_proof import measure_phi, measure_random
from nqx.constants import NQXConfig
from nqx.energy import random_rotation_energy_pj


def test_random_rotation_energy_grows_quadratic_in_dim():
    cfg64 = NQXConfig(dim=64)
    cfg128 = NQXConfig(dim=128)
    e64 = random_rotation_energy_pj(cfg64, n_vec=1024)
    e128 = random_rotation_energy_pj(cfg128, n_vec=1024)
    ratio = e128["total_pj"] / e64["total_pj"]
    assert 3.0 < ratio < 5.0, f"Expected ~4x for 2x dim (dim^2), got {ratio:.2f}"


def test_phi_total_below_random_total():
    phi = measure_phi(dim=128, n_vec=512)
    rnd = measure_random(dim=128, n_vec=512)
    assert phi["total_pj"] < rnd["total_pj"]
    assert rnd["total_pj"] / phi["total_pj"] > 5.0


def test_random_matrix_bytes_match_dim_squared():
    cfg = NQXConfig(dim=128)
    e = random_rotation_energy_pj(cfg, n_vec=64)
    assert e["matrix_bytes"] == 128 * 128 * 2
