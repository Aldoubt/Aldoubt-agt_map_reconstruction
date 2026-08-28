import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE


def test_headland_depth_ground_gate_sweep_cli_writes_depth_sensitivity(tmp_path):
    shape = (3, 6)
    map_path = tmp_path / "map.pgm"
    cv2.imwrite(str(map_path), np.flipud(np.full(shape, UNKNOWN_VALUE, dtype=np.uint8)))

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    masks = {
        "entry_depth_0_0p5": np.zeros(shape, dtype=bool),
        "entry_depth_0p5_1": np.zeros(shape, dtype=bool),
        "exit_depth_0_0p5": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_depth_0_0p5"][:, 0] = True
    masks["entry_depth_0p5_1"][:, 1] = True
    masks["structurally_unresolved_cross"][:, 3] = True
    masks["exit_depth_0_0p5"][:, 5] = True
    mask_files = {}
    for key, mask in masks.items():
        filename = f"{key}_mask.npy"
        np.save(profile_dir / filename, mask)
        mask_files[key] = filename

    profile = {
        "grid_shape_yx": list(shape),
        "entry": {
            "bands": [
                {"mask_key": "entry_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5},
                {"mask_key": "entry_depth_0p5_1", "depth_min_m": 0.5, "depth_max_m": 1.0},
            ]
        },
        "exit": {
            "bands": [
                {"mask_key": "exit_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5}
            ]
        },
        "mask_files": mask_files,
        "sources": {"map": str(map_path.resolve())},
    }
    profile_path = profile_dir / "headland_depth_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    ref_a = tmp_path / "ref_a"
    ref_b = tmp_path / "ref_b"
    ref_a.mkdir()
    ref_b.mkdir()
    a = np.zeros(shape, dtype=float)
    b = np.full(shape, 0.05, dtype=float)
    distance = np.full(shape, 0.4, dtype=float)
    np.save(ref_a / "ground_reference.npy", a)
    np.save(ref_b / "ground_reference.npy", b)
    np.save(ref_a / "ground_reference_nearest_support_distance.npy", distance)
    np.save(ref_b / "ground_reference_nearest_support_distance.npy", distance)

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "sweep_headland_depth_ground_gate.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--map",
            str(map_path),
            "--depth-profile",
            str(profile_path),
            "--reference-a",
            str(ref_a),
            "--reference-b",
            str(ref_b),
            "--max-support-distance-m",
            "0.5",
            "1.0",
            "--max-model-disagreement-m",
            "0.1",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        (output / "headland_depth_ground_gate_sweep.json").read_text(encoding="utf-8")
    )
    assert result["method"] == "finite_headland_depth_ground_reference_gate_sweep"
    assert result["entry"]["bands"][0]["grid"][0]["accepted_unknown_fraction"] == 1.0
    assert result["automatic_threshold_selection"] is False
    assert result["physical_site_boundary_required"] is False
    assert result["sources"]["depth_profile"] == str(profile_path.resolve())
    assert (output / "headland_depth_ground_gate_sweep.png").exists()
    assert "automatic_threshold_selection: false" in completed.stdout
    assert "physical_site_boundary_required: false" in completed.stdout
