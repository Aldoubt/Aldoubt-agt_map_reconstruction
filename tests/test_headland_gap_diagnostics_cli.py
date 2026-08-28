import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def _aisle(aisle_id, y):
    return {
        "aisle_id": aisle_id,
        "label": f"A{aisle_id:02d}",
        "polygon_xy": [[20.0, y - 4.0], [80.0, y - 4.0], [80.0, y + 4.0], [20.0, y + 4.0]],
        "centerline_xy": [[20.0, float(y)], [80.0, float(y)]],
        "width_m": 1.0,
        "length_m": 6.0,
        "heading_rad": 0.0,
    }


def test_cli_reports_clarified_gap_metrics(tmp_path):
    shape = (60, 100)
    baseline = np.full(shape, 205, dtype=np.uint8)
    baseline[17:24, 20:81] = 254
    baseline[27:34, 20:81] = 254
    conservative = baseline.copy()
    conservative[17:25, 8:23] = 254

    baseline_pgm = tmp_path / "baseline.pgm"
    conservative_pgm = tmp_path / "conservative.pgm"
    cv2.imwrite(str(baseline_pgm), np.flipud(baseline))
    cv2.imwrite(str(conservative_pgm), np.flipud(conservative))

    aisle_path = tmp_path / "aisles.json"
    aisle_path.write_text(json.dumps({
        "grid": {
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
            "width": 100,
            "height": 60,
            "frame_id": "map",
        },
        "rectangles": [_aisle(1, 20), _aisle(2, 30)],
    }))

    masks = {
        "entry_depth_0_0p5": np.zeros(shape, dtype=np.uint8),
        "entry_boundary_uncertainty": np.zeros(shape, dtype=np.uint8),
        "exit_depth_0_0p5": np.zeros(shape, dtype=np.uint8),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=np.uint8),
        "structurally_unresolved_cross": np.zeros(shape, dtype=np.uint8),
    }
    masks["entry_depth_0_0p5"][:, 4:18] = 1
    masks["entry_boundary_uncertainty"][:, 18:22] = 1
    masks["exit_boundary_uncertainty"][:, 78:82] = 1
    masks["exit_depth_0_0p5"][:, 82:96] = 1
    mask_files = {}
    for key, value in masks.items():
        filename = f"{key}.npy"
        np.save(tmp_path / filename, value)
        mask_files[key] = filename

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "grid_shape_yx": list(shape),
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "entry": {
            "bands": [{"mask_key": "entry_depth_0_0p5"}],
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
        },
        "exit": {
            "bands": [{"mask_key": "exit_depth_0_0p5"}],
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
        },
        "mask_files": mask_files,
    }))

    connectivity_path = tmp_path / "connectivity.json"
    connectivity_path.write_text(json.dumps({
        "radius_m": 0.2,
        "pairs": [{
            "pair_id": "A01-A02",
            "first_aisle": "A01",
            "second_aisle": "A02",
            "side": "entry",
            "evaluation_status": "evaluated",
            "first_anchor_grid_xy": [30.0, 20.0],
            "second_anchor_grid_xy": [30.0, 30.0],
            "baseline_connected": False,
            "conservative_connected": False,
        }],
    }))

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "evaluate_headland_gap_diagnostics.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--baseline-map", str(baseline_pgm),
            "--conservative-map", str(conservative_pgm),
            "--aisles", str(aisle_path),
            "--depth-profile", str(profile_path),
            "--connectivity", str(connectivity_path),
            "--radius", "0.20",
            "--output", str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "safe_overlay_not_bridging: 1" in completed.stdout
    assert "mixed_bridge: 1" in completed.stdout
    assert "promoted_safe=" in completed.stdout
    assert "baseline_new_safe=" in completed.stdout
    assert (output / "headland_gap_diagnostics.json").exists()
    assert (output / "headland_gap_diagnostics.csv").exists()
    assert (output / "headland_gap_diagnostics.png").exists()
