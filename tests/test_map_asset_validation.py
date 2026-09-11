import json

import numpy as np

from agt_map_reconstruction.map_asset_validation import validate_navigation_bundle
from agt_map_reconstruction.maps.navigation_export import write_navigation_bundle


def _build_bundle(tmp_path):
    semantic = np.ones((20, 20), dtype=np.uint8)
    semantic[10, 10] = 6  # pillar: hard static obstacle

    aisles = [
        {
            "aisle_id": 1,
            "label": "A01",
            "polygon_xy": [[1, 1], [18, 1], [18, 18], [1, 18]],
            "width_m": 1.0,
            "length_m": 1.0,
        }
    ]

    bundle = tmp_path / "bundle"
    write_navigation_bundle(
        semantic,
        aisles,
        bundle,
        resolution=0.05,
        clearance_radii_m=(0.05,),
    )
    return bundle


def test_valid_navigation_bundle_passes_contract(tmp_path):
    bundle = _build_bundle(tmp_path)
    report = validate_navigation_bundle(bundle)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["metrics"]["static_obstacle_cells"] == 1
    assert report["metrics"]["static_obstacle_as_free_cells"] == 0
    assert report["metrics"]["observed_gray_values"] == [0, 254]


def test_static_obstacle_exported_as_free_is_rejected(tmp_path):
    bundle = _build_bundle(tmp_path)

    static = np.load(bundle / "static_obstacle_mask.npy")
    static[5, 5] = 1
    np.save(bundle / "static_obstacle_mask.npy", static)

    report = validate_navigation_bundle(bundle)
    assert report["valid"] is False
    assert any("static obstacle cells are exported as free" in item for item in report["errors"])


def test_validator_rejects_missing_required_asset(tmp_path):
    bundle = _build_bundle(tmp_path)
    (bundle / "candidate_mask.npy").unlink()

    report = validate_navigation_bundle(bundle)
    assert report["valid"] is False
    assert "missing required asset: candidate_mask.npy" in report["errors"]
