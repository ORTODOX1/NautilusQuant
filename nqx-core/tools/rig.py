#!/usr/bin/env python3
"""Random Instruction Generator for NQ-ASM. Generates valid programs and
runs them through assembler + emulator, checking invariants."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from nqx.assembler import assemble
from nqx.constants import NQXConfig
from nqx.coverage import Coverage, trace_program, write_report
from nqx.cpu import NQXCore
from nqx.isa import Opcode
from nqx.mx_unit import MX_FORMAT_BY_INDEX

N_VECTOR_REGS = 16
LDV_BATCH = 4


@dataclass
class GenState:
    rng: np.random.Generator
    initialised_regs: set = field(default_factory=set)
    polarised_regs: set = field(default_factory=set)
    quant_meta_regs: set = field(default_factory=set)
    sign_meta_regs: set = field(default_factory=set)
    mx_meta_regs: set = field(default_factory=set)
    subbit_meta_regs: set = field(default_factory=set)
    n_dma: int = 0


def _pick_reg(state: GenState, must_be_in: set | None = None) -> int:
    pool = list(must_be_in if must_be_in is not None else state.initialised_regs)
    if not pool:
        return -1
    return int(state.rng.choice(pool))


def _gen_ldv(state: GenState) -> str:
    rd = int(state.rng.integers(0, N_VECTOR_REGS))
    addr = state.n_dma * 0x100
    state.n_dma += 1
    state.initialised_regs.add(rd)
    state.polarised_regs.discard(rd)
    state.quant_meta_regs.discard(rd)
    state.sign_meta_regs.discard(rd)
    state.mx_meta_regs.discard(rd)
    state.subbit_meta_regs.discard(rd)
    return f"LDV V{rd}, [0x{addr:x}]"


def _gen_ldv_async(state: GenState) -> str:
    line = _gen_ldv(state)
    return line.replace("LDV ", "LDV_ASYNC ")


def _gen_gvns(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    layer = int(state.rng.integers(0, 3))
    return f"GVNS V{rd}, {layer}"


def _gen_gvns_inv(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    layer = int(state.rng.integers(0, 3))
    return f"GVNS_INV V{rd}, {layer}"


def _gen_polar(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    state.polarised_regs.add(rd)
    return f"POLAR V{rd}"


def _gen_ipolar(state: GenState) -> str | None:
    rd = _pick_reg(state, state.polarised_regs)
    if rd < 0:
        return None
    state.polarised_regs.discard(rd)
    return f"IPOLAR V{rd}"


def _drop_meta(state: GenState, rd: int) -> None:
    state.quant_meta_regs.discard(rd)
    state.sign_meta_regs.discard(rd)
    state.mx_meta_regs.discard(rd)
    state.subbit_meta_regs.discard(rd)


def _gen_quant(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    bits = int(state.rng.choice([2, 3, 4]))
    _drop_meta(state, rd)
    state.quant_meta_regs.add(rd)
    return f"QUANT V{rd}, {bits}"


def _gen_dequant(state: GenState) -> str | None:
    rd = _pick_reg(state, state.quant_meta_regs)
    if rd < 0:
        return None
    return f"DEQUANT V{rd}, 3"


def _gen_qjl(state: GenState) -> str | None:
    if not state.initialised_regs or len(state.initialised_regs) < 2:
        return None
    rd = _pick_reg(state, state.quant_meta_regs)
    if rd < 0:
        return None
    rs1_pool = list(state.initialised_regs - {rd})
    if not rs1_pool:
        return None
    rs1 = int(state.rng.choice(rs1_pool))
    state.sign_meta_regs.add(rd)
    return f"QJL V{rd}, V{rs1}, 0x80"


def _gen_pack3(state: GenState) -> str | None:
    rd = _pick_reg(state, state.quant_meta_regs)
    if rd < 0:
        return None
    sign_pool = list(state.initialised_regs)
    if not sign_pool:
        return None
    sign_reg = int(state.rng.choice(sign_pool))
    return f"PACK3 V{rd}, V{sign_reg}"


def _gen_mxpack(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    fmt = state.rng.choice(MX_FORMAT_BY_INDEX)
    _drop_meta(state, rd)
    state.mx_meta_regs.add(rd)
    return f"MXPACK V{rd}, {fmt}"


def _gen_mxunpack(state: GenState) -> str | None:
    rd = _pick_reg(state, state.mx_meta_regs)
    if rd < 0:
        return None
    fmt = state.rng.choice(MX_FORMAT_BY_INDEX)
    return f"MXUNPACK V{rd}, {fmt}"


def _gen_subbit_enc(state: GenState) -> str | None:
    rd = _pick_reg(state, state.polarised_regs)
    if rd < 0:
        return None
    r = int(state.rng.integers(1, 5))
    a = int(state.rng.integers(1, 5))
    _drop_meta(state, rd)
    state.subbit_meta_regs.add(rd)
    return f"SUBBIT_ENC V{rd}, {r}, {a}"


def _gen_subbit_dec(state: GenState) -> str | None:
    rd = _pick_reg(state, state.subbit_meta_regs)
    if rd < 0:
        return None
    return f"SUBBIT_DEC V{rd}"


def _gen_attn_dot(state: GenState) -> str | None:
    if len(state.polarised_regs) < 2:
        return None
    pool = list(state.polarised_regs)
    qi = int(state.rng.choice(pool))
    pool.remove(qi)
    ki = int(state.rng.choice(pool))
    return f"ATTN_DOT V{qi}, V{ki}"


def _gen_mov(state: GenState) -> str | None:
    if not state.initialised_regs:
        return None
    rs = _pick_reg(state)
    rd = int(state.rng.integers(0, N_VECTOR_REGS))
    state.initialised_regs.add(rd)
    return f"MOV V{rd}, V{rs}"


def _gen_barrier(state: GenState) -> str:
    return "BARRIER"


def _gen_nop(state: GenState) -> str:
    return "NOP"


def _gen_stv(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    addr = 0x10000000 + state.n_dma * 0x100
    state.n_dma += 1
    return f"STV V{rd}, [0x{addr:x}]"


def _gen_unqjl(state: GenState) -> str | None:
    if not state.initialised_regs or len(state.initialised_regs) < 2:
        return None
    rd = _pick_reg(state)
    rs1 = _pick_reg(state)
    return f"UNQJL V{rd}, V{rs1}"


def _gen_enc(state: GenState) -> str:
    src = state.n_dma * 0x100
    dst = 0x10000000 + state.n_dma * 0x100
    state.n_dma += 1
    return f"ENC [0x{src:x}], [0x{dst:x}], {LDV_BATCH}"


def _gen_dec(state: GenState) -> str | None:
    src = state.n_dma * 0x100
    dst = 0x10000000 + state.n_dma * 0x100
    state.n_dma += 1
    return f"DEC [0x{src:x}], [0x{dst:x}], {LDV_BATCH}"


def _gen_unpack3(state: GenState) -> str | None:
    rd = _pick_reg(state)
    if rd < 0:
        return None
    _drop_meta(state, rd)
    return f"UNPACK3 V{rd}"


GENERATORS = [
    ("LDV", _gen_ldv),
    ("LDV_ASYNC", _gen_ldv_async),
    ("MOV", _gen_mov),
    ("GVNS", _gen_gvns),
    ("GVNS_INV", _gen_gvns_inv),
    ("POLAR", _gen_polar),
    ("IPOLAR", _gen_ipolar),
    ("QUANT", _gen_quant),
    ("DEQUANT", _gen_dequant),
    ("QJL", _gen_qjl),
    ("PACK3", _gen_pack3),
    ("MXPACK", _gen_mxpack),
    ("MXUNPACK", _gen_mxunpack),
    ("SUBBIT_ENC", _gen_subbit_enc),
    ("SUBBIT_DEC", _gen_subbit_dec),
    ("ATTN_DOT", _gen_attn_dot),
    ("BARRIER", _gen_barrier),
    ("NOP", _gen_nop),
    ("STV", _gen_stv),
    ("UNQJL", _gen_unqjl),
    ("UNPACK3", _gen_unpack3),
]


_TAIL_GENERATORS = [_gen_enc, _gen_dec]


def generate_program(rng: np.random.Generator, length: int) -> tuple[str, dict]:
    state = GenState(rng=rng)
    state.initialised_regs.add(0)
    lines = ["LDV V0, [0x0]"]
    state.n_dma += 1
    if rng.random() < 0.5:
        lines.append(_gen_enc(state))
        lines.append(_gen_dec(state))
    while len(lines) < length:
        name, gen = GENERATORS[int(rng.integers(0, len(GENERATORS)))]
        line = gen(state)
        if line is not None:
            lines.append(line)
    lines.append("HALT")
    opcode_hist = {}
    for line in lines:
        op = line.split()[0]
        opcode_hist[op] = opcode_hist.get(op, 0) + 1
    return "\n".join(lines), opcode_hist


def execute_with_invariants(src: str, dim: int, n_dma: int) -> dict:
    cfg = NQXConfig(dim=dim)
    core = NQXCore(cfg)
    rng = np.random.default_rng(0)
    for i in range(max(1, n_dma)):
        vec = rng.standard_normal((LDV_BATCH, dim)).astype(np.float32)
        core.load_vectors_to_hbm(i * 0x100, vec)
    program = assemble(src)
    for ins in program:
        if ins.opcode in (Opcode.LDV, Opcode.LDV_ASYNC):
            ins.extra["count"] = LDV_BATCH

    core.execute_program(program)

    for i in range(N_VECTOR_REGS):
        v = core.vrf.read(i)
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"V{i} contains non-finite values after run")
    return {
        "n_instructions": len(program),
        "cycles": core.cycles.total,
        "energy_nj": core.energy.total_nj(),
    }


def run_iterations(n_iters: int, length_min: int, length_max: int, dim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    coverage = Coverage()
    crashes: List[dict] = []
    cycles_total = 0
    n_instr_total = 0
    for i in range(n_iters):
        length = int(rng.integers(length_min, length_max + 1))
        src, _ = generate_program(np.random.default_rng(int(rng.integers(0, 10**9))), length)
        n_dma = sum(1 for line in src.split("\n") if line.startswith("LDV"))
        try:
            program = assemble(src)
            coverage.merge(trace_program(program))
            for ins in program:
                if ins.opcode in (Opcode.LDV, Opcode.LDV_ASYNC):
                    ins.extra["count"] = LDV_BATCH
            result = execute_with_invariants(src, dim=dim, n_dma=n_dma)
            cycles_total += result["cycles"]
            n_instr_total += result["n_instructions"]
        except Exception as exc:
            crashes.append(
                {"iter": i, "src": src, "error": repr(exc), "trace": traceback.format_exc(limit=4)}
            )
    return {
        "iterations": n_iters,
        "instructions_total": n_instr_total,
        "cycles_total": cycles_total,
        "coverage": coverage,
        "crashes": crashes,
        "crash_count": len(crashes),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Random Instruction Generator for NQ-ASM")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("audits/results"))
    ap.add_argument("--show-program", action="store_true")
    args = ap.parse_args(argv)

    if args.show_program:
        rng = np.random.default_rng(args.seed)
        src, _ = generate_program(rng, length=20)
        print(src)
        return 0

    report = run_iterations(args.iters, args.min_len, args.max_len, args.dim, args.seed)
    cov = report["coverage"]
    print(
        f"RIG: {report['iterations']} programs, "
        f"{report['instructions_total']} instructions, "
        f"{report['crash_count']} crashes, "
        f"{report['cycles_total']:,} simulated cycles, "
        f"opcode coverage {cov.opcode_coverage_fraction() * 100:.1f}%"
    )
    if cov.missing_opcodes():
        print(f"  uncovered opcodes: {', '.join(cov.missing_opcodes())}")
    if report["crashes"]:
        print("FIRST CRASH:")
        print(report["crashes"][0]["error"])
        print(report["crashes"][0]["src"])
    md = write_report(cov, args.out_dir)
    print(f"  coverage report: {md}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "iterations": report["iterations"],
                    "instructions_total": report["instructions_total"],
                    "cycles_total": report["cycles_total"],
                    "crash_count": report["crash_count"],
                    "opcode_coverage": cov.opcode_coverage_fraction(),
                    "missing_opcodes": cov.missing_opcodes(),
                    "opcode_counts": cov.opcode_counts,
                },
                indent=2,
            )
        )
    return 0 if report["crash_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
