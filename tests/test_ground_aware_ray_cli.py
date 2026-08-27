import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from agt_map_reconstruction.maps.observation_ray_bundle import (
    validate_observation_ray_bundle,
    write_observation_ray_bundle,
)


def test_ground_aware_ray_cli_writes_diagnostic_assets(tmp_path):
    rays = validate_observation_ray_bundle(
        [[0.5, 1.5, 0.20], [0.5, 1.5, 0.20]],
        [[5.5, 1.5, 0.20], [5.5, 1.5, 0.20]],
    )
    ray_path = tmp_path / "observation_rays.npz"
    write_observation_ray_bundle(ray_path, rays)

    ground_path = tmp_path / "ground_reference.npy"
    np.save(ground_path, np.zeros((3, 8), dtype=float))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "grid": {
            "frame_id": "map",
            "resolution": 1.0,
            "origin": [0.0, 0.0, 0.0],
            "width": 8,
            "height": 3,
        }
    }), encoding="utf-8")

    output = tmp_path / "ray_evidence"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_ground_aware_ray_evidence.py"
    completed = subprocess.run([
        sys.executable,
        str(script),
        "--rays", str(ray_path),
        "--ground-reference", str(ground_path),
        "--grid-manifest", str(manifest_path),
        "--output", str(output),
        "--min-ground-relative-height-m", "0.10",
        "--max-ground-relative-height-m", "0.40",
        "--min-support-rays", "2",
    ], text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    support = np.load(output / "ray_free_support_count.npy")
    mask = np.load(output / "ray_free_support_mask.npy")
    manifest = json.loads(
        (output / "observation_evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert support[1, 3] == 2
    assert mask[1, 3] == 1
    assert mask[1, 5] == 0
    assert manifest["ray_policy"]["hit_cell_is_free"] is False
    assert manifest["ray_policy"]["ground_reference_is_semantic_evidence"] is False
    assert manifest["ray_policy"]["semantic_promotion"] is False
    assert manifest["summary"]["supported_cell_count"] == 5
    assert "semantic_promotion: false" in completed.stdout
