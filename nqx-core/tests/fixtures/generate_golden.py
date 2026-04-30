"""Generate golden reference snapshots for regression test (S7)."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore


def main():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((16, 128)).astype(np.float32)

    cfg = NQXConfig(dim=128, bits=3)
    core = NQXCore(cfg)
    enc = core.encode(x)
    dec = core.decode(enc)

    # Hash the full roundtrip for bit-exact regression
    digest = hashlib.sha256(dec.reconstructed.tobytes()).hexdigest()

    path = os.path.join(os.path.dirname(__file__), "golden", "seed42_dim128_bits3.npz")
    np.savez_compressed(
        path,
        input=x,
        quantized_indices=enc.quantized_indices,
        sign_bits=enc.sign_bits,
        mins=enc.mins,
        maxs=enc.maxs,
        packed_bytes=enc.packed_bytes,
        reconstructed=dec.reconstructed,
        sha256=digest,
    )
    print(f"Saved: {path}  sha256={digest}")


if __name__ == "__main__":
    main()
