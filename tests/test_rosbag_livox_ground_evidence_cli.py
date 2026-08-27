from pathlib import Path
import subprocess
import sys


def test_streaming_cli_help_does_not_require_ros_runtime():
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "stream_livox_ground_aware_evidence.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert "--benchmark-run" in completed.stdout
    assert "--ground-reference" in completed.stdout
    assert "--batch-ray-limit" in completed.stdout
    assert "--export-point-stride" in completed.stdout
