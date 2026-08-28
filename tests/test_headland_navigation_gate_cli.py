import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_headland_navigation_gate_cli_writes_traceable_masks(tmp_path):
    shape = (5, 8)
    base = np.full(shape, 205, dtype=np.uint8)
    map_path = tmp_path / "base.npy"
    np.save(map_path, base)

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    masks = {
        "entry_depth_0_0p5": np.zeros(shape, dtype=bool),
        "exit_depth_0_0p5": np.zeros(shape, dtype=bool),
        "entry_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_depth_0_0p5"][2, 1:4] = True
    mask_files = {}
    for key, mask in masks.items():
        filename = f"{key}.npy"
        np.save(profile_dir / filename, mask)
        mask_files[key] = filename
    profile = {
        "grid_shape_yx": list(shape),
        "mask_files": mask_files,
        "entry": {
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
            "bands": [{"mask_key": "entry_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}],
        },
        "exit": {
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
            "bands": [{"mask_key": "exit_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}],
        },
    }
    profile_path = profile_dir / "headland_depth_profile.json"
    profile_path.write_text(json.dumps(profile))

    ref_a = tmp_path / "ref_a"
    ref_b = tmp_path / "ref_b"
    ref_a.mkdir()
    ref_b.mkdir()
    distance = np.full(shape, np.nan)
    distance[2, 1:4] = 0.2
    ground_a = np.zeros(shape)
    ground_b = np.zeros(shape)
    ground_b[2, 3] = 0.3
    for directory, ground in ((ref_a, ground_a), (ref_b, ground_b)):
        np.save(directory / "ground_reference.npy", ground)
        np.save(directory / "ground_reference_nearest_support_distance.npy", distance)

    scans = np.zeros(shape, dtype=np.uint16)
    scans[2, 1:4] = [1, 1, 1]
    scans_path = tmp_path / "scans.npy"
    np.save(scans_path, scans)

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_headland_navigation_gate.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--map-npy", str(map_path),
            "--depth-profile", str(profile_path),
            "--reference-a", str(ref_a),
            "--reference-b", str(ref_b),
            "--scan-support-count", str(scans_path),
            "--entry-max-depth-m", "0.5",
            "--exit-max-depth-m", "0.0",
            "--max-support-distance-m", "0.5",
            "--max-model-disagreement-m", "0.1",
            "--min-scan-support", "1",
            "--output", str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    trusted = np.load(output / "trusted_headland_free_mask.npy")
    assert trusted[2, 1] == 1
    assert trusted[2, 2] == 1
    assert trusted[2, 3] == 0
    assert (output / "headland_navigation_gate.json").exists()
    assert (output / "headland_navigation_uncertainty_mask.npy").exists()
    assert "automatic_threshold_selection: false" in completed.stdout
