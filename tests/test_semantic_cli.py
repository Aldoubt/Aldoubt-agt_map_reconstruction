from pathlib import Path
import subprocess
import sys


def test_semantic_navigation_cli_help_does_not_require_open3d():
    script = Path(__file__).resolve().parents[1] / "tools" / "build_semantic_navigation_assets.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--pcd" in completed.stdout
    assert "--row-direction" in completed.stdout
    assert "--confirmed-free-only" in completed.stdout
