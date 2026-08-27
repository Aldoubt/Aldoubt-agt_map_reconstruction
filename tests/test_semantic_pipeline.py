from types import SimpleNamespace

import numpy as np
import pytest

from agt_map_reconstruction.maps.semantic_pipeline import (
    build_semantic_assets_from_points,
    infer_row_direction_from_evidence,
    metadata_from_statistics,
)
from agt_map_reconstruction.maps.ground_evidence import (
    EvidenceClass,
    GroundEvidenceConfig,
)


def test_infer_row_direction_from_horizontal_evidence_bands():
    evidence = np.zeros((20, 80), dtype=np.uint8)
    evidence[4:7, 5:75] = EvidenceClass.FREE_CONFIRMED
    evidence[13:16, 5:75] = EvidenceClass.FREE_CONFIRMED

    direction = infer_row_direction_from_evidence(evidence)

    assert abs(direction[0]) > 0.99
    assert abs(direction[1]) < 0.05


def test_infer_row_direction_rejects_insufficient_support():
    evidence = np.zeros((5, 5), dtype=np.uint8)
    evidence[2, 2] = EvidenceClass.FREE_CONFIRMED

    with pytest.raises(ValueError, match="row direction"):
        infer_row_direction_from_evidence(evidence)


def test_metadata_from_statistics_preserves_origin_resolution_and_shape():
    statistics = SimpleNamespace(
        low_height=np.zeros((12, 34), dtype=float),
        origin_xy=np.array([-2.5, 7.25]),
        resolution=0.05,
    )

    metadata = metadata_from_statistics(statistics)

    assert metadata.origin_x == -2.5
    assert metadata.origin_y == 7.25
    assert metadata.resolution == 0.05
    assert metadata.width == 34
    assert metadata.height == 12
    assert metadata.frame_id == "map"


def _patch_pipeline(monkeypatch):
    import agt_map_reconstruction.maps.elevation_statistics as elevation_statistics
    import agt_map_reconstruction.maps.ground_evidence as ground_evidence
    import agt_map_reconstruction.maps.semantic_pipeline as semantic_pipeline

    statistics = SimpleNamespace(
        low_height=np.zeros((3, 4), dtype=float),
        q90_height=np.full((3, 4), 1.0, dtype=float),
        point_count=np.full((3, 4), 3, dtype=np.int64),
        origin_xy=np.array([1.0, 2.0], dtype=float),
        resolution=0.05,
    )
    seen = {}

    monkeypatch.setattr(
        elevation_statistics,
        "points_to_elevation_statistics",
        lambda *args, **kwargs: statistics,
    )

    def fake_build_ground_evidence_details(low_height, point_count, config, q90_height=None):
        seen["q90_height"] = q90_height
        return SimpleNamespace(
            ground_surface=np.zeros_like(low_height),
            clearance=np.zeros_like(low_height),
            evidence=np.full(low_height.shape, EvidenceClass.FREE_CONFIRMED, dtype=np.uint8),
        )

    monkeypatch.setattr(
        ground_evidence,
        "build_ground_evidence_details",
        fake_build_ground_evidence_details,
    )
    monkeypatch.setattr(
        semantic_pipeline,
        "write_semantic_navigation_assets",
        lambda **kwargs: {"manifest": {"row_direction": [1.0, 0.0]}},
    )
    return statistics, seen


def test_semantic_pipeline_does_not_use_q90_for_obstacles_by_default(tmp_path, monkeypatch):
    _, seen = _patch_pipeline(monkeypatch)

    build_semantic_assets_from_points(
        points=np.zeros((1, 3), dtype=float),
        output_dir=tmp_path,
        ground_config=GroundEvidenceConfig(resolution=0.05),
        row_direction=[1.0, 0.0],
    )

    assert seen["q90_height"] is None


def test_semantic_pipeline_uses_q90_only_when_explicitly_enabled(tmp_path, monkeypatch):
    statistics, seen = _patch_pipeline(monkeypatch)

    build_semantic_assets_from_points(
        points=np.zeros((1, 3), dtype=float),
        output_dir=tmp_path,
        ground_config=GroundEvidenceConfig(resolution=0.05),
        row_direction=[1.0, 0.0],
        use_q90_for_obstacles=True,
    )

    np.testing.assert_array_equal(seen["q90_height"], statistics.q90_height)
