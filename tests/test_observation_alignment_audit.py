import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from agt_map_reconstruction.maps.observation_alignment_audit import (
    audit_observation_ray_alignment,
    load_navigation_grid,
)
from agt_map_reconstruction.maps.observation_ray_bundle import (
    validate_observation_ray_bundle,
    write_observation_ray_bundle,
)


def _write_map(tmp_path):
    # Repository convention: row 0 is the lower map edge.
    grid = np.asarray(
        [
            [254, 205],
            [0, 254],
        ],
        dtype=np.uint8,
    )
    image = np.flipud(grid)
    image_path = tmp_path / "navigation_base_map.pgm"
    assert cv2.imwrite(str(image_path), image)
    yaml_path = tmp_path / "navigation_base_map.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "image": image_path.name,
                "mode": "trinary",
                "resolution": 1.0,
                "origin": [0.0, 0.0, 0.0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.196,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


def _bundle():
    origins = np.asarray(
        [
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0],
        ]
    )
    endpoints = np.asarray(
        [
            [0.5, 0.5, 0.0],   # free
            [1.5, 0.5, 0.0],   # unknown
            [0.5, 1.5, 0.0],   # occupied
            [2.5, 0.5, 0.0],   # outside map
        ]
    )
    return validate_observation_ray_bundle(origins, endpoints, frame_id="map")


def test_alignment_audit_respects_lower_left_map_orientation(tmp_path):
    navigation = load_navigation_grid(_write_map(tmp_path))
    result = audit_observation_ray_alignment(_bundle(), navigation)

    assert result["ray_count"] == 4
    assert result["origins"]["in_bounds_fraction"] == pytest.approx(1.0)
    assert result["endpoints"]["in_bounds_fraction"] == pytest.approx(0.75)
    assert result["endpoints"]["classes"]["free"]["count"] == 1
    assert result["endpoints"]["classes"]["unknown"]["count"] == 1
    assert result["endpoints"]["classes"]["occupied"]["count"] == 1
    assert result["endpoints"]["classes"]["free"]["fraction_of_in_bounds"] == pytest.approx(1.0 / 3.0)
    assert result["automatic_alignment_acceptance"] is False
    assert result["semantic_promotion"] is False


def test_alignment_audit_cli_writes_json(tmp_path):
    map_yaml = _write_map(tmp_path)
    rays = tmp_path / "observation_rays.npz"
    write_observation_ray_bundle(rays, _bundle())
    output = tmp_path / "alignment.json"
    script = Path(__file__).resolve().parents[1] / "tools" / "audit_observation_ray_alignment.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rays",
            str(rays),
            "--map-yaml",
            str(map_yaml),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ray_count"] == 4
    assert payload["endpoints"]["in_bounds_fraction"] == pytest.approx(0.75)
    assert "automatic_alignment_acceptance: false" in completed.stdout
