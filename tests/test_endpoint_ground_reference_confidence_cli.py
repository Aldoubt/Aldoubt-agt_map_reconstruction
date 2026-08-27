import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from agt_map_reconstruction.maps.navigation_export import FREE_VALUE, UNKNOWN_VALUE


def _write_reference(directory, neighbors, reference, nearest, valid):
    directory.mkdir(parents=True)
    np.save(directory / "ground_reference.npy", reference)
    np.save(directory / "ground_reference_nearest_support_distance.npy", nearest)
    np.save(directory / "ground_reference_valid_mask.npy", valid.astype(np.uint8))
    (directory / "ground_reference_manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "grid": {
                "frame_id": "map",
                "resolution": 1.0,
                "origin": [0.0, 0.0, 0.0],
                "width": int(reference.shape[1]),
                "height": int(reference.shape[0]),
            },
            "model": {
                "model_type": "knn_local_affine",
                "neighbor_count": neighbors,
                "cv_residual_rmse_m": 0.01 * neighbors,
                "cv_residual_p95_abs_m": 0.02 * neighbors,
                "semantic_promotion": False,
            },
            "policy": {"semantic_promotion": False},
        }),
        encoding="utf-8",
    )


def test_endpoint_confidence_cli_writes_audit(tmp_path):
    base = np.full((6, 10), FREE_VALUE, dtype=np.uint8)
    base[1:5, :3] = UNKNOWN_VALUE
    base[1:5, 8:] = UNKNOWN_VALUE
    pgm_path = tmp_path / "map.pgm"
    cv2.imwrite(str(pgm_path), np.flipud(base))

    endpoint = {
        "schema_version": 1,
        "radius_m": 0.20,
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "row_cross_span": [1.0, 4.0],
        "sides": {
            "entry": {"endpoint_fit": {"slope_du_dv": 0.0, "intercept_u": 3.0}},
            "exit": {"endpoint_fit": {"slope_du_dv": 0.0, "intercept_u": 7.0}},
        },
    }
    endpoint_path = tmp_path / "headland_endpoint_envelope.json"
    endpoint_path.write_text(json.dumps(endpoint), encoding="utf-8")

    nearest = np.ones(base.shape, dtype=np.float32)
    ref8 = np.zeros(base.shape, dtype=np.float32)
    ref16 = np.full(base.shape, 0.02, dtype=np.float32)
    valid = np.ones(base.shape, dtype=bool)
    k8_dir = tmp_path / "k8"
    k16_dir = tmp_path / "k16"
    _write_reference(k8_dir, 8, ref8, nearest, valid)
    _write_reference(k16_dir, 16, ref16, nearest, valid)

    output = tmp_path / "audit"
    script = Path(__file__).resolve().parents[1] / "tools" / "audit_endpoint_ground_reference_confidence.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--map", str(pgm_path),
        "--endpoint-envelope", str(endpoint_path),
        "--reference-dir", str(k8_dir),
        "--reference-dir", str(k16_dir),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(
        (output / "endpoint_ground_reference_confidence.json").read_text(encoding="utf-8")
    )
    assert payload["model_names"] == ["k8", "k16"]
    assert payload["sides"]["entry"]["unknown_cell_count"] == 12
    assert payload["sides"]["exit"]["unknown_cell_count"] == 8
    assert payload["policy"]["semantic_promotion"] is False
    assert "k8__k16" in completed.stdout
