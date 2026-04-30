import numpy as np
import pytest

from nqx.constants import NQXConfig
from nqx.mx_unit import MXQuantizer, MX_FORMATS


RMSE_BUDGET = {
    "MXFP4": 0.1,
    "MXFP6": 0.02,
    "MXFP8": 0.01,
    "MXINT8": 0.005,
}


@pytest.mark.parametrize("fmt", list(MX_FORMATS))
def test_mx_constant_input(fmt):
    mx = MXQuantizer(NQXConfig(dim=128), format_name=fmt)
    x = np.full((4, 128), 0.5, dtype=np.float32)

    dequant, _, _ = mx.quantize(x)

    rmse = float(np.sqrt(((dequant - 0.5) ** 2).mean()))
    assert rmse < RMSE_BUDGET[fmt], f"{fmt} constant rmse={rmse:.6f}"


@pytest.mark.parametrize("fmt", list(MX_FORMATS))
def test_mx_all_zeros(fmt):
    mx = MXQuantizer(NQXConfig(dim=128), format_name=fmt)
    x = np.zeros((4, 128), dtype=np.float32)

    dequant, _, _ = mx.quantize(x)

    assert np.all(dequant == 0.0), f"{fmt} zeros not exact"


@pytest.mark.parametrize("fmt", list(MX_FORMATS))
def test_mx_single_outlier_per_block(fmt):
    cfg = NQXConfig(dim=128)
    mx = MXQuantizer(cfg, format_name=fmt, block_size=32)
    n_blocks = (4 * 128) // 32
    x = np.zeros((4, 128), dtype=np.float32)
    offsets = [32 * b + (b % 32) for b in range(n_blocks)]
    for off in offsets:
        x.flat[off] = 100.0

    dequant, meta, _ = mx.quantize(x)

    for off in offsets:
        assert dequant.flat[off] > 10.0, f"{fmt} outlier not preserved at {off}"
    zero_mask = np.ones(x.size, dtype=bool)
    zero_mask[offsets] = False
    rmse_zero = float(np.sqrt((dequant.flat[zero_mask] ** 2).mean()))
    assert rmse_zero < RMSE_BUDGET[fmt] * 2, f"{fmt} outlier zero rmse={rmse_zero:.6f}"
