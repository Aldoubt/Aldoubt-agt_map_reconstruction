import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_local_ground_reference_cli_writes_reference_distance_and_manifest(tmp_path):
    yy, xx = np.indices((14, 18), dtype=float)
    ground = 0.03 * xx + 0.01 * yy + 0.02 * np.sin(xx * 0.4)
    ground[:, 7:10] = np.nan
    ground_path = tmp_path / "ground_surface.npy"
    np.save(ground_path, ground)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "grid": {
            "frame_id": "map",
            "resolution": 0.10,
            "origin": [0.0, 0.0, 0.0],
            "width": 18,
            "height": 14,
        }
    }), encoding="utf-8")

    output = tmp_path / "local_ground_reference"
    script = Path(__file__).resolve().parents[1] / "tools" / "fit_local_ground_reference.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--ground-surface", str(ground_path),
        "--grid-manifest", str(manifest_path),
        "--output", str(output),
        "--neighbors", "12",
        "--chunk-size", "64",
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    reference = np.load(output / "ground_reference.npy")
    distance = np.load(output / "ground_reference_nearest_support_distance.npy")
    valid = np.load(output / "ground_reference_valid_mask.npy")
    manifest = json.loads(
        (output / "ground_reference_manifest.json").read_text(encoding="utf-8")
    )

    assert reference.shape == ground.shape
    assert distance.shape == ground.shape
    assert valid.shape == ground.shape
    assert np.all(np.isfinite(reference[:, 7:10]))
    assert np.all(distance[:, 7:10] > 0.0)
    assert manifest["model"]["model_type"] == "knn_local_affine"
    assert manifest["model"]["neighbor_count"] == 12
    assert manifest["model"]["semantic_promotion"] is False
    assert "local_cv_rmse_m:" in completed.stdout
    assert "global_affine_rmse_m:" in completed.stdout
