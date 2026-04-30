"""NQXCore: top-level orchestrator. Loads ROM-LUT, executes program, exposes traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from nqx.constants import NQXConfig
from nqx.counters import PerfCounters
from nqx.energy import EnergyModel
from nqx.functional_units import (
    AttentionUnit,
    GivensUnit,
    PackUnit,
    PolarUnit,
    QJLUnit,
    QuantUnit,
)
from nqx.isa import Instruction, Opcode
from nqx.lut import GoldenAngleLUT
from nqx.memory import HBM, SRAM, ScalarRegisterFile, VectorRegisterFile
from nqx.mx_unit import MX_FORMAT_BY_INDEX, MXQuantizer
from nqx.pipeline import CycleCounter, Pipeline
from nqx.subbit_unit import SubBitUnit


@dataclass
class EncodeResult:
    quantized_indices: np.ndarray
    sign_bits: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    packed_bytes: bytes
    polar: np.ndarray
    cycles: int
    energy_nj: float


@dataclass
class DecodeResult:
    reconstructed: np.ndarray
    cycles: int
    energy_nj: float


class NQXCore:
    def __init__(self, config: Optional[NQXConfig] = None):
        self.config = config or NQXConfig()
        self.lut = GoldenAngleLUT(self.config)

        self.hbm = HBM(self.config.hbm_bytes)
        self.sram_in = SRAM(self.config.sram_in_bytes, "SRAM_in")
        self.sram_out = SRAM(self.config.sram_out_bytes, "SRAM_out")

        self.vrf = VectorRegisterFile(self.config)
        self.srf = ScalarRegisterFile(self.config)

        self.gu = GivensUnit(self.config, self.lut)
        self.pu = PolarUnit(self.config)
        self.qu = QuantUnit(self.config)
        self.qjl = QJLUnit(self.config)
        self.pk = PackUnit(self.config)
        self.mx = {name: MXQuantizer(self.config, format_name=name) for name in MX_FORMAT_BY_INDEX}
        self.sb = SubBitUnit(self.config)
        self.attn = AttentionUnit(self.config)
        self.last_attn_dot: Optional[np.ndarray] = None
        self.dma_finish_at: int = 0

        self.cycles = CycleCounter()
        self.pipeline = Pipeline(cycles=self.cycles)
        self.energy = EnergyModel()
        self.perf = PerfCounters()

        self.last_pack_meta: Dict[int, Dict[str, Any]] = {}
        self.trace_log: List[str] = []

    def reset_counters(self) -> None:
        self.cycles = CycleCounter()
        self.pipeline = Pipeline(cycles=self.cycles)
        self.energy = EnergyModel()
        self.perf.reset()
        self.trace_log.clear()
        self.dma_finish_at = 0

    def load_vectors_to_hbm(self, addr: int, vectors: np.ndarray) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        assert (
            vectors.shape[-1] == self.config.dim
        ), f"expected dim={self.config.dim}, got {vectors.shape}"
        self.hbm.store_fp16_vectors(addr, vectors)

    def execute_program(self, program: List[Instruction]) -> Dict[str, Any]:
        pc = 0
        ldv_count: Dict[int, int] = {}
        results: Dict[str, Any] = {"halted": False, "outputs": {}}
        while pc < len(program):
            ins = program[pc]
            self.trace_log.append(f"PC={pc:04d}  {ins}")
            if ins.opcode == Opcode.HALT:
                results["halted"] = True
                break
            self._execute(ins, ldv_count, results)
            pc += 1
        self.perf.counts["cycle_count"] = self.cycles.total
        self.perf.counts["prng_cycles_baseline"] = 4 * self.config.dim * self.config.dim
        return results

    def _execute(
        self, ins: Instruction, ldv_count: Dict[int, int], results: Dict[str, Any]
    ) -> None:
        op = ins.opcode

        if op == Opcode.NOP:
            self.cycles.tick(1, "ctrl")
            return

        if op == Opcode.BARRIER:
            wait = max(0, self.dma_finish_at - self.cycles.total)
            if wait > 0:
                self.cycles.tick(wait, "DMA_wait")
                self.perf.add("stall_cycles", wait)
            self.dma_finish_at = self.cycles.total
            self.cycles.tick(1, "ctrl")
            return

        if op == Opcode.LDV:
            self._do_ldv(ins, ldv_count)
            return

        if op == Opcode.LDV_ASYNC:
            self._do_ldv_async(ins, ldv_count)
            return

        if op == Opcode.STV:
            self._do_stv(ins, results)
            return

        if op == Opcode.MOV:
            self.vrf.write(ins.rd, self.vrf.read(ins.rs1).copy())
            self.cycles.tick(1, "MOV")
            return

        if op in (Opcode.GVNS, Opcode.GVNS_INV):
            inverse = op == Opcode.GVNS_INV
            x = self.vrf.read(ins.rd)
            out, fu = self.gu.apply_layer(x, ins.rs1, inverse=inverse)
            self.vrf.write(ins.rd, out)
            self.cycles.tick(fu.cycles, f"GVNS_L{ins.rs1}")
            self.energy.add_unit("GivensUnit", fu.energy_pj)
            self.perf.add("gu_busy_cycles", fu.cycles)
            return

        if op == Opcode.POLAR:
            x = self.vrf.read(ins.rd)
            out, fu = self.pu.to_polar(x)
            self.vrf.write(ins.rd, out)
            self.cycles.tick(fu.cycles, "POLAR")
            self.energy.add_unit("PolarUnit", fu.energy_pj)
            self.perf.add("pu_busy_cycles", fu.cycles)
            return

        if op == Opcode.IPOLAR:
            x = self.vrf.read(ins.rd)
            out, fu = self.pu.from_polar(x)
            self.vrf.write(ins.rd, out)
            self.cycles.tick(fu.cycles, "IPOLAR")
            self.energy.add_unit("PolarUnit", fu.energy_pj)
            self.perf.add("pu_busy_cycles", fu.cycles)
            return

        if op == Opcode.QUANT:
            x = self.vrf.read(ins.rd)
            dequant, q_idx, mins, maxs, fu = self.qu.quantize(x, ins.rs1)
            self.vrf.write(ins.rd, dequant)
            self.last_pack_meta[ins.rd] = {
                "q": q_idx,
                "mins": mins,
                "maxs": maxs,
                "bits": ins.rs1,
            }
            self.cycles.tick(fu.cycles, "QUANT")
            self.energy.add_unit("QuantUnit", fu.energy_pj)
            self.perf.add("qu_busy_cycles", fu.cycles)
            return

        if op == Opcode.DEQUANT:
            meta = self.last_pack_meta.get(ins.rd)
            if meta is None:
                raise RuntimeError(f"DEQUANT V{ins.rd}: no metadata (run QUANT/UNPACK3 first)")
            dequant, fu = self.qu.dequantize(meta["q"], meta["mins"], meta["maxs"], ins.rs1)
            self.vrf.write(ins.rd, dequant)
            self.cycles.tick(fu.cycles, "DEQUANT")
            self.energy.add_unit("QuantUnit", fu.energy_pj)
            self.perf.add("qu_busy_cycles", fu.cycles)
            return

        if op == Opcode.QJL:
            original = self.vrf.read(ins.rd)
            quantized = self.vrf.read(ins.rs1)
            alpha = ins.rs2 / 256.0 if ins.rs2 else self.config.qjl_alpha
            corrected, sign_bits, fu = self.qjl.apply(original, quantized, alpha)
            self.vrf.write(ins.rd, corrected)
            self.last_pack_meta.setdefault(ins.rd, {})["sign"] = sign_bits
            if ins.rs1 in self.last_pack_meta:
                self.last_pack_meta[ins.rd].update(self.last_pack_meta[ins.rs1])
                self.last_pack_meta[ins.rd]["sign"] = sign_bits
            self.cycles.tick(fu.cycles, "QJL")
            self.energy.add_unit("QJLUnit", fu.energy_pj)
            return

        if op == Opcode.UNQJL:
            self.cycles.tick(1, "UNQJL")
            return

        if op == Opcode.PACK3:
            meta = self.last_pack_meta.get(ins.rd)
            if meta is None or "q" not in meta:
                raise RuntimeError(f"PACK3 V{ins.rd}: no quant metadata")
            sign = meta.get("sign")
            if sign is None:
                sign = np.zeros_like(meta["q"])
            packed, fu = self.pk.pack3plus1(meta["q"], sign)
            base = 0
            self.sram_out.write_bytes(base, packed)
            meta["packed"] = packed
            self.cycles.tick(fu.cycles, "PACK3")
            self.energy.add_unit("PackUnit", fu.energy_pj)
            return

        if op == Opcode.UNPACK3:
            n = ldv_count.get(ins.rd, 1)
            base = 0
            blob_len = (n * self.config.dim * (self.config.bits + 1) + 7) // 8
            data = self.sram_in.read_bytes(base, blob_len)
            q, sign, fu = self.pk.unpack3plus1(data, n)
            self.last_pack_meta[ins.rd] = {
                "q": q,
                "sign": sign,
                "bits": self.config.bits,
            }
            self.cycles.tick(fu.cycles, "UNPACK3")
            self.energy.add_unit("PackUnit", fu.energy_pj)
            return

        if op == Opcode.MXPACK:
            fmt_idx = ins.rs1
            if not (0 <= fmt_idx < len(MX_FORMAT_BY_INDEX)):
                raise RuntimeError(f"MXPACK: fmt index {fmt_idx} out of range")
            fmt_name = MX_FORMAT_BY_INDEX[fmt_idx]
            mx = self.mx[fmt_name]
            x = self.vrf.read(ins.rd)
            dequant, meta, fu = mx.quantize(x)
            self.vrf.write(ins.rd, dequant)
            packed = mx.serialize(meta)
            meta["packed"] = packed
            self.last_pack_meta[ins.rd] = {"mx": meta, "fmt": fmt_name, "packed": packed}
            self.sram_out.write_bytes(0, packed)
            self.cycles.tick(fu.cycles, "MXPACK")
            self.energy.add_unit("MXUnit", fu.energy_pj)
            return

        if op == Opcode.MXUNPACK:
            fmt_idx = ins.rs1
            if not (0 <= fmt_idx < len(MX_FORMAT_BY_INDEX)):
                raise RuntimeError(f"MXUNPACK: fmt index {fmt_idx} out of range")
            fmt_name = MX_FORMAT_BY_INDEX[fmt_idx]
            mx = self.mx[fmt_name]
            entry = self.last_pack_meta.get(ins.rd)
            if entry is None or "mx" not in entry:
                raise RuntimeError(f"MXUNPACK V{ins.rd}: no MX metadata (run MXPACK first)")
            meta = entry["mx"]
            dequant, fu = mx.dequantize(meta)
            self.vrf.write(ins.rd, dequant)
            self.cycles.tick(fu.cycles, "MXUNPACK")
            self.energy.add_unit("MXUnit", fu.energy_pj)
            return

        if op == Opcode.SUBBIT_ENC:
            r_bits = ins.rs1
            a_bits = ins.rs2
            polar = self.vrf.read(ins.rd)
            dequant, meta, fu = self.sb.encode(polar, r_bits, a_bits)
            self.vrf.write(ins.rd, dequant)
            self.last_pack_meta[ins.rd] = {"subbit": meta}
            self.cycles.tick(fu.cycles, "SUBBIT_ENC")
            self.energy.add_unit("SubBitUnit", fu.energy_pj)
            return

        if op == Opcode.SUBBIT_DEC:
            entry = self.last_pack_meta.get(ins.rd)
            if entry is None or "subbit" not in entry:
                raise RuntimeError(
                    f"SUBBIT_DEC V{ins.rd}: no subbit metadata (run SUBBIT_ENC first)"
                )
            dequant, fu = self.sb.decode(entry["subbit"])
            self.vrf.write(ins.rd, dequant)
            self.cycles.tick(fu.cycles, "SUBBIT_DEC")
            self.energy.add_unit("SubBitUnit", fu.energy_pj)
            return

        if op == Opcode.ATTN_DOT:
            q = self.vrf.read(ins.rs1)
            k = self.vrf.read(ins.rs2)
            result, fu = self.attn.dot_polar(q, k)
            self.last_attn_dot = result
            if result.size == 1:
                self.srf.write(0, float(result.flat[0]))
            else:
                self.srf.write(0, result.flat[: self.config.dim])
            self.cycles.tick(fu.cycles, "ATTN_DOT")
            self.energy.add_unit("AttentionUnit", fu.energy_pj)
            return

        if op == Opcode.ENC:
            self._do_enc(ins, results)
            return

        if op == Opcode.DEC:
            self._do_dec(ins, results)
            return

        raise NotImplementedError(f"opcode {op.name}")

    def _do_ldv(self, ins: Instruction, ldv_count: Dict[int, int]) -> None:
        addr = ins.extra.get("addr", ins.imm)
        n = ins.extra.get("count", 1)
        x = self.hbm.load_fp16_vectors(addr, n, self.config.dim)
        self.vrf.write(ins.rd, x)
        ldv_count[ins.rd] = n
        bytes_moved = n * self.config.dim * 2
        cycles = max(1, int(self.config.cycles_dma_per_byte * bytes_moved))
        self.cycles.tick(cycles, "LDV")
        self.dma_finish_at = self.cycles.total
        self.energy.add_memory("HBM", bytes_moved * self.config.pj_hbm_byte)
        self.energy.add_memory("SRAM", bytes_moved * self.config.pj_sram_byte)
        self.perf.add("dma_in_bytes", bytes_moved)

    def _do_ldv_async(self, ins: Instruction, ldv_count: Dict[int, int]) -> None:
        addr = ins.extra.get("addr", ins.imm)
        n = ins.extra.get("count", 1)
        x = self.hbm.load_fp16_vectors(addr, n, self.config.dim)
        self.vrf.write(ins.rd, x)
        ldv_count[ins.rd] = n
        bytes_moved = n * self.config.dim * 2
        cycles = max(1, int(self.config.cycles_dma_per_byte * bytes_moved))
        finish = self.cycles.total + cycles
        if finish > self.dma_finish_at:
            self.dma_finish_at = finish
        self.cycles.tick(1, "LDV_ASYNC_kick")
        self.energy.add_memory("HBM", bytes_moved * self.config.pj_hbm_byte)
        self.energy.add_memory("SRAM", bytes_moved * self.config.pj_sram_byte)
        self.perf.add("dma_in_bytes", bytes_moved)

    def _do_stv(self, ins: Instruction, results: Dict[str, Any]) -> None:
        addr = ins.extra.get("addr", ins.imm)
        meta = self.last_pack_meta.get(ins.rd, {})
        packed = meta.get("packed")
        if packed is None:
            data = self.vrf.read(ins.rd).astype(np.float16).tobytes()
            self.hbm.store_packed(addr, data)
            bytes_moved = len(data)
        else:
            self.hbm.store_packed(addr, packed)
            bytes_moved = len(packed)
        results.setdefault("outputs", {})[addr] = meta or {}
        self.perf.add("dma_out_bytes", bytes_moved)
        cycles = max(1, int(self.config.cycles_dma_per_byte * bytes_moved))
        self.cycles.tick(cycles, "STV")
        self.energy.add_memory("HBM", bytes_moved * self.config.pj_hbm_byte)

    def _do_enc(self, ins: Instruction, results: Dict[str, Any]) -> None:
        src = ins.extra["src"]
        dst = ins.extra["dst"]
        cnt = ins.extra["cnt"]
        x = self.hbm.load_fp16_vectors(src, cnt, self.config.dim)

        bytes_in = cnt * self.config.dim * 2
        self.energy.add_memory("HBM", bytes_in * self.config.pj_hbm_byte)
        self.energy.add_memory("SRAM", bytes_in * self.config.pj_sram_byte)

        rotated, fu = self.gu.apply_layer(x, 0)
        self.energy.add_unit("GivensUnit", fu.energy_pj)
        rotated, fu = self.gu.apply_layer(rotated, 1)
        self.energy.add_unit("GivensUnit", fu.energy_pj)
        rotated, fu = self.gu.apply_layer(rotated, 2)
        self.energy.add_unit("GivensUnit", fu.energy_pj)
        polar, fu = self.pu.to_polar(rotated)
        self.energy.add_unit("PolarUnit", fu.energy_pj)
        dequant, q_idx, mins, maxs, fu = self.qu.quantize(polar, self.config.bits)
        self.energy.add_unit("QuantUnit", fu.energy_pj)
        corrected, sign_bits, fu = self.qjl.apply(polar, dequant)
        self.energy.add_unit("QJLUnit", fu.energy_pj)
        packed, fu = self.pk.pack3plus1(q_idx, sign_bits)
        self.energy.add_unit("PackUnit", fu.energy_pj)

        self.hbm.store_packed(dst, packed)
        bytes_out = len(packed)
        self.energy.add_memory("HBM", bytes_out * self.config.pj_hbm_byte)

        depth = (
            3 * self.config.cycles_givens_layer
            + self.config.cycles_polar
            + self.config.cycles_quant_minmax
            + self.config.cycles_quant_round
            + self.config.cycles_qjl
            + self.config.cycles_pack
        )
        steady = cnt
        cycles = depth + steady - 1
        self.cycles.tick(cycles, "ENC_macro")

        out = EncodeResult(
            quantized_indices=q_idx,
            sign_bits=sign_bits,
            mins=mins,
            maxs=maxs,
            packed_bytes=packed,
            polar=polar,
            cycles=cycles,
            energy_nj=self.energy.total_nj(),
        )
        results.setdefault("outputs", {})[dst] = out

    def _do_dec(self, ins: Instruction, results: Dict[str, Any]) -> None:
        dst = ins.extra["dst"]
        cnt = ins.extra["cnt"]

        latest_enc: Optional[EncodeResult] = None
        for v in (results.get("outputs") or {}).values():
            if isinstance(v, EncodeResult):
                latest_enc = v
        if latest_enc is None:
            raise RuntimeError("DEC: no preceding ENC result available")

        bytes_in = len(latest_enc.packed_bytes)
        self.energy.add_memory("HBM", bytes_in * self.config.pj_hbm_byte)

        q = latest_enc.quantized_indices
        mins = latest_enc.mins
        maxs = latest_enc.maxs
        bits = self.config.bits

        dequant, fu = self.qu.dequantize(q, mins, maxs, bits)
        self.energy.add_unit("QuantUnit", fu.energy_pj)
        cart, fu = self.pu.from_polar(dequant)
        self.energy.add_unit("PolarUnit", fu.energy_pj)
        out, fu = self.gu.apply_layer(cart, 2, inverse=True)
        self.energy.add_unit("GivensUnit", fu.energy_pj)
        out, fu = self.gu.apply_layer(out, 1, inverse=True)
        self.energy.add_unit("GivensUnit", fu.energy_pj)
        out, fu = self.gu.apply_layer(out, 0, inverse=True)
        self.energy.add_unit("GivensUnit", fu.energy_pj)

        self.hbm.store_fp16_vectors(dst, out)
        bytes_out = cnt * self.config.dim * 2
        self.energy.add_memory("HBM", bytes_out * self.config.pj_hbm_byte)

        depth = 1 + 1 + 3 * self.config.cycles_givens_layer + self.config.cycles_polar
        cycles = depth + cnt - 1
        self.cycles.tick(cycles, "DEC_macro")

        dec = DecodeResult(reconstructed=out, cycles=cycles, energy_nj=self.energy.total_nj())
        results.setdefault("outputs", {})[dst] = dec

    def encode(self, vectors: np.ndarray) -> EncodeResult:
        try:
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if vectors.shape[0] == 0:
                raise ValueError("empty batch: vectors must have at least one row")
            if not np.all(np.isfinite(vectors)):
                raise ValueError("vectors contain NaN or Inf")
            return self._encode_impl(vectors)
        except Exception:
            self._dump_failure_snapshot(vectors)
            raise

    def _encode_impl(self, vectors: np.ndarray) -> EncodeResult:
        cnt = vectors.shape[0]
        src = 0
        dst = 0
        self.load_vectors_to_hbm(src, vectors)
        ins = Instruction(opcode=Opcode.ENC, imm=cnt, extra={"src": src, "dst": dst, "cnt": cnt})
        results: Dict[str, Any] = {"outputs": {}}
        self._do_enc(ins, results)
        return results["outputs"][dst]

    def _dump_failure_snapshot(self, vectors: np.ndarray) -> None:
        import hashlib
        import os
        import traceback
        from datetime import datetime, timezone

        from nqx import __version__

        snap_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audits", "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        try:
            # LUT hash
            buf = bytearray()
            for name in ("L1", "L2", "L3"):
                layer = self.lut.layers[name]
                buf.extend(layer.cos_arr.tobytes())
                buf.extend(layer.sin_arr.tobytes())
            lut_hash = hashlib.sha256(buf).hexdigest()
        except Exception:
            lut_hash = "unknown"

        tb = traceback.format_exc()
        path = os.path.join(snap_dir, f"error-{ts}.npz")
        np.savez_compressed(
            path,
            input=vectors,
            traceback=tb,
            version=__version__,
            lut_sha256=lut_hash,
        )
        print(f"[NQX] Failure snapshot saved: {path}")

    def decode(self, encoded: EncodeResult) -> DecodeResult:
        results: Dict[str, Any] = {"outputs": {0: encoded}}
        ins = Instruction(
            opcode=Opcode.DEC,
            imm=encoded.quantized_indices.shape[0],
            extra={"src": 0, "dst": 0, "cnt": encoded.quantized_indices.shape[0]},
        )
        self._do_dec(ins, results)
        return results["outputs"][0]

    def rotation_matrix(self) -> np.ndarray:
        d = self.config.dim
        eye = np.eye(d, dtype=np.float32)
        out, _ = self.gu.apply_layer(eye, 0)
        out, _ = self.gu.apply_layer(out, 1)
        out, _ = self.gu.apply_layer(out, 2)
        return out.T.astype(np.float64)

    def forward_rotation(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out, _ = self.gu.apply_layer(x.astype(np.float32), 0)
        out, _ = self.gu.apply_layer(out, 1)
        out, _ = self.gu.apply_layer(out, 2)
        return out

    def inverse_rotation(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out, _ = self.gu.apply_layer(x.astype(np.float32), 2, inverse=True)
        out, _ = self.gu.apply_layer(out, 1, inverse=True)
        out, _ = self.gu.apply_layer(out, 0, inverse=True)
        return out
