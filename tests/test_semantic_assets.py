import json

import numpy as np
import pytest

from agt_map_reconstruction.maps.aisle_reconstruction import (
    recover_aisle_rectangles,
    write_aisle_bundle,
)
from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_evidence import EvidenceClass
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    build_navigation_layers,
)
from agt_map_reconstruction.maps.semantic_assets import (
    write_semantic_navigation_assets,
)
from agt_map_reconstruction.maps.semantic_reconstruction import (
    LABEL_AISLE,
    LABEL_OBSTACLE_CANDIDATE,
    LABEL_OCCUPIED_CONFIRMED,
    LABEL_UNKNOWN,
    build_basic_semantic_labels,
    corridor_seed_from_evidence,
    semantic_labels_from_evidence,
)


def test_evidence_mapping_keeps_confirmed_obstacles_hard_and_interpolation_unknown():
    evidence = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    labels = semantic_labels_from_evidence(evidence)
    assert labels.tolist() == [[
        LABEL_UNKNOWN,
        LABEL_AISLE,
        LABEL_OCCUPIED_CONFIRMED,
        LABEL_UNKNOWN,
    ]]


def test_corridor_seed_uses_bounded_interpolation_only_as_geometry_support():
    evidence = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    assert corridor_seed_from_evidence(evidence, True).tolist() == [
        [False, True, False, True]
    ]
    assert corridor_seed_from_evidence(evidence, False).tolist() == [
        [False, True, False, False]
    ]


def test_confirmed_occupied_label_overrides_aisle_prior():
    semantic = np.zeros((7, 12), dtype=np.uint8)
    semantic[2:5, 1:11] = LABEL_AISLE
    semantic[3, 6] = LABEL_OCCUPIED_CONFIRMED
    aisle = {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[1, 2], [10, 2], [10, 4], [1, 4]],
        "width_m": 0.15,
        "length_m": 0.45,
    }

    layers = build_navigation_layers(semantic, [aisle])

    assert layers.base_map[3, 6] == OCCUPIED_VALUE
    assert bool(layers.hard_obstacle_mask[3, 6]) is True


def _three_horizontal_aisles_with_headland_bridge():
    corridor = np.zeros((30, 80), dtype=bool)
    corridor[4:8, 5:75] = True
    corridor[13:17, 5:75] = True
    corridor[22:26, 5:75] = True
    corridor[4:26, 72:75] = True
    return corridor


def test_row_projection_recovers_three_aisles_despite_headland_bridge():
    aisles = recover_aisle_rectangles(
        _three_horizontal_aisles_with_headland_bridge(),
        row_direction=np.array([1.0, 0.0]),
        resolution=0.10,
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=5.0,
    )

    assert [item["label"] for item in aisles] == ["A01", "A02", "A03"]
    for aisle in aisles:
        assert 0.35 <= aisle["width_m"] <= 0.50
        assert aisle["length_m"] >= 6.5
        assert np.asarray(aisle["polygon_xy"]).shape == (4, 2)


def test_aisle_bundle_contains_legacy_grid_and_explicit_map_geometry(tmp_path):
    corridor = _three_horizontal_aisles_with_headland_bridge()
    metadata = GridMetadata(
        resolution=0.10,
        origin_x=-2.0,
        origin_y=3.0,
        width=corridor.shape[1],
        height=corridor.shape[0],
    )
    aisles = recover_aisle_rectangles(
        corridor,
        np.array([1.0, 0.0]),
        metadata.resolution,
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=5.0,
    )

    path = tmp_path / "aisle_rectangles.json"
    payload = write_aisle_bundle(aisles, metadata, path)

    assert json.loads(path.read_text()) == payload
    assert payload["schema_version"] == 1
    assert payload["grid"] == metadata.to_dict()
    first = payload["rectangles"][0]
    assert np.asarray(first["polygon_xy"]).shape == (4, 2)
    assert np.asarray(first["polygon_map_xy_m"]).shape == (4, 2)
    assert np.asarray(first["centerline_map_xy_m"]).shape == (2, 2)


def test_semantic_navigation_bundle_writes_current_exp003_inputs(tmp_path):
    evidence = np.zeros((20, 60), dtype=np.uint8)
    evidence[3:7, 5:55] = EvidenceClass.FREE_CONFIRMED
    evidence[12:16, 5:55] = EvidenceClass.FREE_CONFIRMED
    evidence[3:16, 52:54] = EvidenceClass.GROUND_INTERPOLATED
    evidence[4, 30] = EvidenceClass.OCCUPIED_CONFIRMED
    metadata = GridMetadata(
        resolution=0.10,
        origin_x=-1.0,
        origin_y=2.0,
        width=60,
        height=20,
    )

    result = write_semantic_navigation_assets(
        evidence=evidence,
        metadata=metadata,
        row_direction=np.array([1.0, 0.0]),
        output_dir=tmp_path,
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=4.0,
        navigation_clearance_radii_m=(0.10,),
    )

    expected = {
        "evidence.npy",
        "semantic_labels.npy",
        "corridor_seed.npy",
        "aisle_rectangles.json",
        "semantic_manifest.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert (tmp_path / "navigation" / "navigation_base_map.pgm").exists()
    assert (tmp_path / "navigation" / "navigation_base_map.yaml").exists()
    assert (tmp_path / "navigation" / "validation.json").exists()

    labels = np.load(tmp_path / "semantic_labels.npy")
    assert labels[4, 30] == LABEL_OBSTACLE_CANDIDATE
    assert result["navigation"]["layers"].base_map[4, 30] == FREE_VALUE
    assert bool(result["navigation"]["layers"].candidate_mask[4, 30]) is True
    assert result["manifest"]["aisle_count"] == 2
    assert result["manifest"]["evidence_counts"]["occupied_confirmed"] == 1
    assert result["manifest"]["aisle_conflict_candidate_count"] == 1


def test_legacy_exp002_fallback_stays_advisory_not_hard():
    traversability = np.array([[0, 1, 1, 2]], dtype=np.uint8)
    corridor = np.array([[0, 1, 0, 0]], dtype=bool)

    labels = build_basic_semantic_labels(traversability, corridor)

    assert labels.tolist() == [[
        LABEL_UNKNOWN,
        LABEL_AISLE,
        LABEL_UNKNOWN,
        LABEL_OBSTACLE_CANDIDATE,
    ]]


def test_semantic_bundle_rejects_grid_shape_mismatch(tmp_path):
    metadata = GridMetadata(0.1, 0.0, 0.0, width=4, height=3)
    with pytest.raises(ValueError, match="metadata shape"):
        write_semantic_navigation_assets(
            np.zeros((2, 4), np.uint8),
            metadata,
            [1.0, 0.0],
            tmp_path,
        )
