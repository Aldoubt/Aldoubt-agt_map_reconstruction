import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def _rect():
    return {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[1, 1], [8, 1], [8, 4], [1, 4]],
        "width_m": 1.0,
        "length_m": 4.0,
    }


def test_build_navigation_map_cli_accepts_conservative_masks(tmp_path):
    semantic = np.zeros((6, 10), dtype=np.uint8)
    semantic[2, 7] = 2
    semantic_path = tmp_path / "semantic.npy"
    np.save(semantic_path, semantic)

    aisles_path = tmp_path / "aisles.json"
    aisles_path.write_text(json.dumps({"rectangles": [_rect()]}))

    trusted = np.zeros_like(semantic, dtype=np.uint8)
    trusted[2, 5] = 1
    trusted_path = tmp_path / "trusted.npy"
    np.save(trusted_path, trusted)

    uncertainty = np.zeros_like(semantic, dtype=np.uint8)
    uncertainty[2, 4] = 1
    uncertainty_path = tmp_path / "uncertainty.npy"
    np.save(uncertainty_path, uncertainty)

    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "tools" / "build_navigation_map.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--semantic-labels", str(semantic_path),
            "--aisles", str(aisles_path),
            "--trusted-free-mask", str(trusted_path),
            "--uncertainty-mask", str(uncertainty_path),
            "--no-promote-aisle-prior",
            "--output", str(output),
            "--resolution", "0.05",
            "--clearance-radii", "0.10",
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "validation.json").read_text())
    assert payload["aisle_prior_promotion_enabled"] is False
    assert payload["trusted_free_cell_count"] == 1
    assert payload["uncertainty_exported_as_free_cell_count"] == 0
    assert (output / "trusted_free_mask.npy").exists()
    assert (output / "uncertainty_mask.npy").exists()
    assert "conservative_uncertainty_semantics_valid: True" in completed.stdout
