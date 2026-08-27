import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import UNKNOWN_VALUE


def test_uncertainty_roi_ground_gate_sweep_cli_reuses_frozen_masks(tmp_path):
    shape = (4, 6)
    map_path = tmp_path / "map.pgm"
    cv2.imwrite(str(map_path), np.full(shape, UNKNOWN_VALUE, dtype=np.uint8))

    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    masks = {
        "entry_conservative_outward": np.zeros(shape, dtype=bool),
        "entry_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "exit_conservative_outward": np.zeros(shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_conservative_outward"][:, 0] = True
    masks["entry_boundary_uncertainty"][:, 1] = True
    masks["exit_boundary_uncertainty"][:, 4] = True
    masks["exit_conservative_outward"][:, 5] = True
    masks["structurally_unresolved_cross"][:, 2:4] = True

    mask_files = {}
    for name, mask in masks.items():
        filename = f"{name}.npy"
        np.save(roi_dir / filename, mask)
        mask_files[name] = filename
    roi_path = roi_dir / "structural_endpoint_uncertainty_roi.json"
    roi_path.write_text(
        json.dumps({"mask_files": mask_files}),
        encoding="utf-8",
    )

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
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "sweep_structural_endpoint_uncertainty_ground_gate.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--map",
            str(map_path),
            "--roi",
            str(roi_path),
            "--reference-a",
            str(ref_a),
            "--reference-b",
            str(ref_b),
            "--output",
            str(output),
            "--max-support-distance-m",
            "0.5",
            "1.0",
            "--max-model-disagreement-m",
            "0.1",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        (output / "structural_endpoint_uncertainty_ground_gate_sweep.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["method"] == "fused_structural_roi_ground_reference_gate_sweep"
    assert result["regions"]["entry_conservative_outward"]["unknown_cell_count"] == 4
    assert result["regions"]["entry_conservative_outward"]["grid"][0][
        "accepted_unknown_fraction"
    ] == 1.0
    assert result["automatic_threshold_selection"] is False
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False
    assert "automatic_threshold_selection: false" in completed.stdout
