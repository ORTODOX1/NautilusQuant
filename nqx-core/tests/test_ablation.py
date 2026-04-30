import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_ablation_runs(tmp_path):
    out = tmp_path / "ablation.md"
    js = tmp_path / "ablation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "bench" / "ablation.py"),
            "--dims",
            "64",
            "128",
            "--bits",
            "3",
            "4",
            "--vectors",
            "256",
            "--out",
            str(out),
            "--json",
            str(js),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote" in result.stdout
    text = out.read_text()
    assert "phi" in text and "hadamard" in text and "random" in text and "none" in text
    assert "dim = 64" in text and "dim = 128" in text
    import json

    rows = json.loads(js.read_text())
    assert len(rows) == 4 * 2 * 2
    for r in rows:
        assert r["rmse"] >= 0


def test_phi_rotation_beats_no_rotation():
    sys.path.insert(0, str(REPO))
    from bench.ablation import run

    rows = run(dims=[128], bits_list=[3], n_vec=512, seed=42)
    by_rot = {r["rotation"]: r["rmse"] for r in rows}
    assert (
        by_rot["phi"] < by_rot["none"]
    ), f"phi-rotation should improve RMSE over no rotation: {by_rot}"
