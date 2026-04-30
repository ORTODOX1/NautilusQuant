import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from nqx.constants import NQXConfig
from nqx.lut import GoldenAngleLUT

REPO = Path(__file__).resolve().parent.parent


def test_gen_rom_matches_lut(tmp_path):
    out = tmp_path / "golden_rom.mem"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "gen_rom.py"),
            "--dim",
            "128",
            "--out",
            str(out),
            "--verify",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ROM verified" in result.stdout
    cfg = NQXConfig(dim=128)
    lut = GoldenAngleLUT(cfg)
    text = out.read_text().splitlines()
    words = [
        int(line.split()[0], 16) for line in text if line.strip() and not line.startswith("//")
    ]
    expected_words = sum(len(lut.layers[n]) for n in ("L1", "L2", "L3")) * 4
    assert len(words) == expected_words

    idx = 0
    for layer_name in ("L1", "L2", "L3"):
        layer = lut.layers[layer_name]
        for k in range(len(layer)):
            assert words[idx] == layer.pairs[k][0]
            idx += 1
            assert words[idx] == layer.pairs[k][1]
            idx += 1
            cos_val = struct.unpack("<f", struct.pack("<I", words[idx]))[0]
            assert np.float32(cos_val) == np.float32(layer.cos[k])
            idx += 1
            sin_val = struct.unpack("<f", struct.pack("<I", words[idx]))[0]
            assert np.float32(sin_val) == np.float32(layer.sin[k])
            idx += 1
