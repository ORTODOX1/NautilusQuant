import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_run_demo_finishes_under_30_seconds():
    t0 = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "demos" / "run_demo.py"),
            "--vectors",
            "256",
            "--runs",
            "10",
            "--seq",
            "256",
            "--n-heads",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, result.stderr
    assert elapsed < 30, f"took {elapsed:.2f}s"
    assert "Side-by-side" in result.stdout
    assert "Scaling projection" in result.stdout


def test_side_by_side_table_has_all_metrics():
    text = (
        (REPO / "demos" / "side_by_side.md").read_text()
        if (REPO / "demos" / "side_by_side.md").exists()
        else ""
    )
    if not text:
        subprocess.run(
            [
                sys.executable,
                str(REPO / "demos" / "side_by_side.py"),
                "--vectors",
                "256",
                "--runs",
                "5",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        text = (REPO / "demos" / "side_by_side.md").read_text()
    for needle in ("RMSE roundtrip", "Cycles", "Energy", "LUT", "Determinism", "Compression"):
        assert needle in text


def test_pitch_md_has_ten_slides():
    text = (REPO / "demos" / "pitch.md").read_text()
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert len(headings) >= 10
