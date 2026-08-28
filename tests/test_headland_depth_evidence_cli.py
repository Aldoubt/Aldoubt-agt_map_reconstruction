import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE


def test_headland_depth_evidence_cli_reuses_frozen_arrays(tmp_path):
    shape = (4, 8)
    map_path = tmp_path / "map.pgm"
    cv2.imwrite(str(map_path), np.flipud(np.full(shape, UNKNOWN_VALUE, dtype=np.uint8)))

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    masks = {
        "entry_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "entry_depth_0_0p5": np.zeros(shape, dtype=bool),
        "entry_depth_0p5_1": np.zeros(shape, dtype=bool),
        "exit_depth_0_0p5": np.zeros(shape, dtype=bool),
        "exit_depth_0p5_1": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_depth_0_0p5"][:, 0] = True
    masks["entry_depth_0p5_1"][:, 1] = True
    masks["entry_boundary_uncertainty"][:, 2] = True
    masks["structurally_unresolved_cross"][:, 3:5] = True
    masks["exit_boundary_uncertainty"][:, 5] = True
    masks["exit_depth_0p5_1"][:, 6] = True
    masks["exit_depth_0_0p5"][:, 7] = True

    mask_files = {}
    for key, mask in masks.items():
        filename = f"{key}_mask.npy"
        np.save(profile_dir / filename, mask)
        mask_files[key] = filename

    payload = {
        "method": "finite_structural_headland_depth_profile",
        "grid_shape_yx": list(shape),
        "depth_edges_m": [0.0, 0.5, 1.0],
        "entry": {
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
            "bands": [
                {"mask_key": "entry_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5},
                {"mask_key": "entry_depth_0p5_1", "depth_min_m": 0.5, "depth_max_m": 1.0},
            ],
        },
        "exit": {
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
            "bands": [
                {"mask_key": "exit_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5},
                {"mask_key": "exit_depth_0p5_1", "depth_min_m": 0.5, "depth_max_m": 1.0},
            ],
        },
        "mask_files": mask_files,
        "sources": {"map": str(map_path.resolve())},
    }
    profile_path = profile_dir / "headland_depth_profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    ground = np.zeros(shape, dtype=float)
    scan = np.zeros(shape, dtype=np.uint32)
    ray = np.zeros(shape, dtype=np.uint32)
    scan[:, 0] = 2
    ray[:, 0] = 1
    ground_path = tmp_path / "ground.npy"
    scan_path = tmp_path / "scan.npy"
    ray_path = tmp_path / "ray.npy"
    np.save(ground_path, ground)
    np.save(scan_path, scan)
    np.save(ray_path, ray)

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "evaluate_headland_depth_evidence.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--depth-profile",
            str(profile_path),
            "--ground-reference",
            str(ground_path),
            "--scan-support-count",
            str(scan_path),
            "--ray-support-count",
            str(ray_path),
            "--min-repeated-scans",
            "2",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "headland_depth_evidence.json").read_text(encoding="utf-8"))
    assert result["method"] == "finite_headland_depth_observation_sufficiency"
    assert result["entry"]["bands"][0]["unknown_repeated_scan_support_cell_count"] == 4
    assert result["entry"]["bands"][0]["depth_midpoint_m"] == 0.25
    assert result["policy"]["frozen_evidence_reused"] is True
    assert result["policy"]["rosbag_replay_performed"] is False
    assert result["policy"]["ray_evidence_regenerated"] is False
    assert result["sources"]["depth_profile"] == str(profile_path.resolve())
    assert (output / "headland_depth_evidence.png").exists()
    assert "frozen_evidence_reused: true" in completed.stdout
    assert "rosbag_replay_performed: false" in completed.stdout
