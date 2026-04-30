"""Memory hierarchy: HBM, SRAM scratchpad, register files."""

from __future__ import annotations

from typing import Dict

import numpy as np

from nqx.constants import NQXConfig


class _MemRegion:
    def __init__(self, size_bytes: int, name: str, lazy: bool = False):
        self.size_bytes = size_bytes
        self.name = name
        self.lazy = lazy
        self.buf: bytearray | None = None if lazy else bytearray(size_bytes)
        self._pages: dict[int, bytearray] = {}
        self._page_size = 64 * 1024
        self.reads_bytes = 0
        self.writes_bytes = 0

    def _ensure_buf(self) -> bytearray:
        if self.buf is None:
            self.buf = bytearray(self.size_bytes)
        return self.buf

    def read_bytes(self, addr: int, n: int) -> bytes:
        if addr < 0 or addr + n > self.size_bytes:
            raise IndexError(f"{self.name}: oob read [{addr}, {addr + n})")
        self.reads_bytes += n
        if self.lazy:
            return self._lazy_read(addr, n)
        return bytes(self.buf[addr : addr + n])

    def write_bytes(self, addr: int, data: bytes) -> None:
        if addr < 0 or addr + len(data) > self.size_bytes:
            raise IndexError(f"{self.name}: oob write [{addr}, {addr + len(data)})")
        if self.lazy:
            self._lazy_write(addr, data)
        else:
            self.buf[addr : addr + len(data)] = data
        self.writes_bytes += len(data)

    def _lazy_read(self, addr: int, n: int) -> bytes:
        out = bytearray(n)
        end = addr + n
        cur = addr
        while cur < end:
            page_id = cur // self._page_size
            page_off = cur - page_id * self._page_size
            chunk = min(self._page_size - page_off, end - cur)
            page = self._pages.get(page_id)
            if page is not None:
                out[cur - addr : cur - addr + chunk] = page[page_off : page_off + chunk]
            cur += chunk
        return bytes(out)

    def _lazy_write(self, addr: int, data: bytes) -> None:
        end = addr + len(data)
        cur = addr
        while cur < end:
            page_id = cur // self._page_size
            page_off = cur - page_id * self._page_size
            chunk = min(self._page_size - page_off, end - cur)
            page = self._pages.get(page_id)
            if page is None:
                page = bytearray(self._page_size)
                self._pages[page_id] = page
            page[page_off : page_off + chunk] = data[cur - addr : cur - addr + chunk]
            cur += chunk


class HBM(_MemRegion):
    def __init__(self, size_bytes: int = 256 * 1024 * 1024):
        super().__init__(size_bytes, "HBM", lazy=True)

    def store_fp16_vectors(self, addr: int, vectors: np.ndarray) -> None:
        assert vectors.dtype in (np.float16, np.float32)
        data = vectors.astype(np.float16).tobytes()
        self.write_bytes(addr, data)

    def load_fp16_vectors(self, addr: int, count: int, dim: int) -> np.ndarray:
        n = count * dim * 2
        raw = self.read_bytes(addr, n)
        arr = np.frombuffer(raw, dtype=np.float16).reshape(count, dim).copy()
        return arr.astype(np.float32)

    def store_packed(self, addr: int, packed: bytes) -> None:
        self.write_bytes(addr, packed)

    def load_packed(self, addr: int, n_bytes: int) -> bytes:
        return self.read_bytes(addr, n_bytes)


class SRAM(_MemRegion):
    def __init__(self, size_bytes: int, name: str = "SRAM"):
        super().__init__(size_bytes, name)


class VectorRegisterFile:
    def __init__(self, config: NQXConfig):
        self.config = config
        self.regs: Dict[int, np.ndarray] = {
            i: np.zeros(config.dim, dtype=np.float32) for i in range(config.n_vector_regs)
        }
        self.batch: Dict[int, np.ndarray] = {
            i: np.zeros((0, config.dim), dtype=np.float32) for i in range(config.n_vector_regs)
        }

    def read(self, idx: int) -> np.ndarray:
        self._check(idx)
        return self.batch[idx]

    def write(self, idx: int, value: np.ndarray) -> None:
        self._check(idx)
        if value.ndim == 1:
            assert value.shape == (self.config.dim,)
            self.batch[idx] = value.reshape(1, self.config.dim).astype(np.float32, copy=False)
        else:
            assert (
                value.shape[-1] == self.config.dim
            ), f"V{idx}: expected dim={self.config.dim}, got {value.shape}"
            self.batch[idx] = value.astype(np.float32, copy=False)

    def _check(self, idx: int) -> None:
        if not (0 <= idx < self.config.n_vector_regs):
            raise IndexError(f"V{idx} out of range (have {self.config.n_vector_regs})")


class ScalarRegisterFile:
    def __init__(self, config: NQXConfig):
        self.config = config
        self.regs = np.zeros((config.n_scalar_regs, config.dim), dtype=np.float32)

    def read(self, idx: int) -> np.ndarray:
        self._check(idx)
        return self.regs[idx]

    def write(self, idx: int, value: np.ndarray) -> None:
        self._check(idx)
        if np.isscalar(value):
            self.regs[idx, :] = value
        else:
            v = np.asarray(value, dtype=np.float32).reshape(-1)
            if v.size == 1:
                self.regs[idx, :] = v.item()
            else:
                self.regs[idx, : v.size] = v

    def _check(self, idx: int) -> None:
        if not (0 <= idx < self.config.n_scalar_regs):
            raise IndexError(f"S{idx} out of range (have {self.config.n_scalar_regs})")
