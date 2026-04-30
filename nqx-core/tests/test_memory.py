import numpy as np
import pytest

from nqx.memory import HBM, SRAM


def test_hbm_read_write_boundaries():
    mem = HBM(size_bytes=256)
    mem.write_bytes(0, b"\x01" * 64)
    assert mem.read_bytes(0, 64) == b"\x01" * 64

    mem.write_bytes(192, b"\x02" * 64)
    assert mem.read_bytes(192, 64) == b"\x02" * 64


def test_hbm_oob_read():
    mem = HBM(size_bytes=256)
    with pytest.raises(IndexError):
        mem.read_bytes(0, 257)
    with pytest.raises(IndexError):
        mem.read_bytes(250, 10)
    with pytest.raises(IndexError):
        mem.read_bytes(-1, 1)


def test_hbm_oob_write():
    mem = HBM(size_bytes=256)
    with pytest.raises(IndexError):
        mem.write_bytes(0, b"\x00" * 257)
    with pytest.raises(IndexError):
        mem.write_bytes(-1, b"\x00")


def test_hbm_lazy_page_crossing():
    page = 64 * 1024
    mem = HBM(size_bytes=page * 3)
    data = b"\xab" * 128
    mem.write_bytes(page - 64, data)
    mem.write_bytes(page * 2 - 64, data)

    assert mem.read_bytes(page - 64, 128) == data
    assert mem.read_bytes(page * 2 - 64, 128) == data
    assert mem._pages.get(0) is not None
    assert mem._pages.get(1) is not None
    assert mem._pages.get(2) is not None


def test_hbm_fp16_roundtrip():
    mem = HBM(size_bytes=4096)
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(8, 128)).astype(np.float32)

    mem.store_fp16_vectors(0, vecs)
    loaded = mem.load_fp16_vectors(0, 8, 128)

    assert loaded.shape == (8, 128)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, vecs, atol=1e-3)


def test_hbm_packed_roundtrip():
    mem = HBM(size_bytes=1024)
    data = bytes(range(256))

    mem.store_packed(0, data)
    loaded = mem.load_packed(0, 256)

    assert loaded == data


def test_sram_read_write():
    sram = SRAM(size_bytes=128)
    sram.write_bytes(0, b"\xca\xfe")
    assert sram.read_bytes(0, 2) == b"\xca\xfe"


def test_sram_oob():
    sram = SRAM(size_bytes=64)
    with pytest.raises(IndexError):
        sram.read_bytes(60, 10)
