import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_ground_reference_plane_cli_writes_reference_and_residuals(tmp_path):
    resolution = 0.5
    height, width = 5, 6
    yy, xx = np.indices((height, width))
    world_x = (xx + 0.5) * resolution
    world_y = (yy + 0.5) * resolution
    truth = 0.02 * world_x - 0.01 * world_y + 0.30
    measured = truth.copy()
    measured[:, 3:] = np.nan

    ground_path = tmp_path / "ground_surface.npy"
    np.save(ground_path, measured)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "grid": {
            "frame_id": "map",
            "resolution": resolution,
            "origin": [0.0, 0.0, 0.0],
            "width": width,
            "height": height,
        }
    }), encoding="utf-8")

    output = tmp_path / "ground_reference"
    script = Path(__file__).resolve().parents[1] / "tools" / "fit_ground_reference_plane.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--ground-surface", str(ground_path),
        "--grid-manifest", str(manifest_path),
        "--output", str(output),
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    reference = np.load(output / "ground_reference.npy")
    source_mask = np.load(output / "ground_reference_source_mask.npy")
    payload = json.loads(
        (output / "ground_reference_manifest.json").read_text(encoding="utf-8")
    )
    np.testing.assert_allclose(reference, truth, atol=1e-6)
    assert int(source_mask.sum()) == int(np.isfinite(measured).sum())
    assert payload["residual_rmse_m"] < 1e-10
    assert payload["semantic_promotion"] is False
    assert "semantic_promotion: false" in completed.stdout
