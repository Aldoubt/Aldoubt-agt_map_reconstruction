from pathlib import Path
import subprocess
import sys


def _help_output():
    script = Path(__file__).resolve().parents[1] / "tools" / "build_semantic_navigation_assets.py"
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
    )


def test_semantic_navigation_cli_help_does_not_require_open3d():
    completed = _help_output()

    assert completed.returncode == 0, completed.stderr
    assert "--pcd" in completed.stdout
    assert "--row-direction" in completed.stdout
    assert "--confirmed-free-only" in completed.stdout


def test_semantic_navigation_cli_does_not_expose_unused_offline_costmap_knobs():
    completed = _help_output()

    assert completed.returncode == 0, completed.stderr
    assert "--obstacle-inflation-radius-m" not in completed.stdout
    assert "--interpolated-ground-cost" not in completed.stdout
