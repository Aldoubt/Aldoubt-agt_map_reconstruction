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


def _handoff(label, aisle_id, y, entry_x):
    return {
        "aisle_id": aisle_id,
        "label": label,
        "status": "ok",
        "width_clearance_eligible": True,
        "entry_handoff": {
            "grid_xy": [float(entry_x), float(y)],
            "map_xy_m": [0.1 * float(entry_x), 0.1 * float(y)],
            "heading_rad": 0.0,
            "clearance_m": 0.3,
        },
        "exit_handoff": {
            "grid_xy": [70.0, float(y)],
            "map_xy_m": [7.0, 0.1 * float(y)],
            "heading_rad": 0.0,
            "clearance_m": 0.3,
        },
    }


def test_cli_writes_headland_connectivity_bundle(tmp_path):
    shape = (60, 100)
    unknown, free = 205, 254
    baseline = np.full(shape, unknown, dtype=np.uint8)
    baseline[17:24, 20:81] = free
    baseline[27:34, 20:81] = free
    conservative = baseline.copy()
    conservative[17:34, 8:23] = free

    baseline_pgm = tmp_path / "baseline.pgm"
    conservative_pgm = tmp_path / "conservative.pgm"
    cv2.imwrite(str(baseline_pgm), np.flipud(baseline))
    cv2.imwrite(str(conservative_pgm), np.flipud(conservative))

    aisles = [_aisle(1, 20), _aisle(2, 30)]
    aisle_path = tmp_path / "aisles.json"
    aisle_path.write_text(json.dumps({
        "grid": {
            "resolution": 0.1,
            "origin": [0.0, 0.0, 0.0],
            "width": 100,
            "height": 60,
            "frame_id": "map",
        },
        "rectangles": aisles,
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
            "bands": [{"mask_key": "entry_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}],
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
        },
        "exit": {
            "bands": [{"mask_key": "exit_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}],
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
        },
        "mask_files": mask_files,
    }))

    baseline_handoffs = tmp_path / "baseline_handoffs.json"
    conservative_handoffs = tmp_path / "conservative_handoffs.json"
    baseline_handoffs.write_text(json.dumps({"handoffs": [
        _handoff("A01", 1, 20, 30),
        _handoff("A02", 2, 30, 30),
    ]}))
    conservative_handoffs.write_text(json.dumps({"handoffs": [
        _handoff("A01", 1, 20, 24),
        _handoff("A02", 2, 30, 24),
    ]}))

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "evaluate_headland_handoff_connectivity.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--baseline-map", str(baseline_pgm),
            "--conservative-map", str(conservative_pgm),
            "--aisles", str(aisle_path),
            "--depth-profile", str(profile_path),
            "--baseline-handoffs", str(baseline_handoffs),
            "--conservative-handoffs", str(conservative_handoffs),
            "--radius", "0.20",
            "--output", str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "gained_by_trusted_overlay: 1" in completed.stdout
    assert (output / "headland_connectivity.json").exists()
    assert (output / "headland_gates.geojson").exists()
    assert (output / "planner_pairs.yaml").exists()
    assert (output / "headland_connectivity.png").exists()
