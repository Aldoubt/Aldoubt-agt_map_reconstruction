from pathlib import Path
import subprocess
import sys


def test_d31_v2_cli_help_exposes_ridge_and_fit_quality_controls():
    script = Path(__file__).resolve().parents[1] / "tools" / "build_structural_endpoint_d31_v2.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert "--min-support-fraction" in completed.stdout
    assert "--min-persistence-m" in completed.stdout
    assert "--max-side-endpoint-disagreement-m" in completed.stdout
    assert "--max-fit-rmse-m" in completed.stdout
