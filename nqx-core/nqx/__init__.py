"""NQX-Core: NautilusQuant hardware accelerator emulator."""

from nqx.assembler import AssemblyError, assemble
from nqx.constants import GOLDEN_ANGLE, PHI, NQXConfig
from nqx.cpu import NQXCore
from nqx.energy import EnergyModel
from nqx.functional_units import (
    GivensUnit,
    PackUnit,
    PolarUnit,
    QJLUnit,
    QuantUnit,
)
from nqx.isa import Instruction, Opcode, decode_instruction, encode_instruction
from nqx.lut import GoldenAngleLUT
from nqx.memory import HBM, SRAM, ScalarRegisterFile, VectorRegisterFile
from nqx.pipeline import CycleCounter, Pipeline

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
