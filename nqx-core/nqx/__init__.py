"""NQX-Core: NautilusQuant hardware accelerator emulator."""

from nqx.constants import PHI, GOLDEN_ANGLE, NQXConfig
from nqx.lut import GoldenAngleLUT
from nqx.memory import HBM, SRAM, VectorRegisterFile, ScalarRegisterFile
from nqx.functional_units import (
    GivensUnit,
    PolarUnit,
    QuantUnit,
    QJLUnit,
    PackUnit,
)
from nqx.pipeline import Pipeline, CycleCounter
from nqx.energy import EnergyModel
from nqx.isa import Opcode, Instruction, encode_instruction, decode_instruction
from nqx.assembler import assemble, AssemblyError
from nqx.cpu import NQXCore

__version__ = "1.0.0"
__all__ = [
    "PHI",
    "GOLDEN_ANGLE",
    "NQXConfig",
    "GoldenAngleLUT",
    "HBM",
    "SRAM",
    "VectorRegisterFile",
    "ScalarRegisterFile",
    "GivensUnit",
    "PolarUnit",
    "QuantUnit",
    "QJLUnit",
    "PackUnit",
    "Pipeline",
    "CycleCounter",
    "EnergyModel",
    "Opcode",
    "Instruction",
    "encode_instruction",
    "decode_instruction",
    "assemble",
    "AssemblyError",
    "NQXCore",
]
