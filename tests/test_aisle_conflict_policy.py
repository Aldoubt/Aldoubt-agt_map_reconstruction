import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata
from agt_map_reconstruction.maps.ground_evidence import EvidenceClass
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    build_navigation_layers,
)
from agt_map_reconstruction.maps.semantic_assets import write_semantic_navigation_assets
from agt_map_reconstruction.maps.semantic_reconstruction import (
    LABEL_AISLE,
    LABEL_OBSTACLE_CANDIDATE,
    LABEL_OCCUPIED_CONFIRMED,
    LABEL_UNKNOWN,
    refine_occupied_evidence_with_aisle_prior,
)


def _aisle():
    return {
        "aisle_id": 1,
        "label": "A01",
        "polygon_xy": [[0, 0], [3, 0], [3, 0], [0, 0]],
        "width_m": 0.10,
        "length_m": 0.30,
    }


def test_aisle_semantic_refinement_demotes_only_confirmed_occupied_conflicts():
    labels = np.array([[
        LABEL_UNKNOWN,
        LABEL_OCCUPIED_CONFIRMED,
        LABEL_OCCUPIED_CONFIRMED,
        LABEL_AISLE,
    ]], dtype=np.uint8)
    aisle_prior = np.array([[False, True, False, True]], dtype=bool)

    refined = refine_occupied_evidence_with_aisle_prior(labels, aisle_prior)

    assert refined.tolist() == [[
        LABEL_UNKNOWN,
        LABEL_OBSTACLE_CANDIDATE,
        LABEL_OCCUPIED_CONFIRMED,
        LABEL_AISLE,
    ]]


def test_navigation_can_promote_only_candidates_inside_aisles():
    semantic = np.array([[
        LABEL_UNKNOWN,
        LABEL_OBSTACLE_CANDIDATE,
        LABEL_OCCUPIED_CONFIRMED,
        LABEL_AISLE,
    ]], dtype=np.uint8)

    layers = build_navigation_layers(
        semantic,
        [_aisle()],
        promote_aisle_prior=False,
        promote_candidates_in_aisles=True,
    )

    assert layers.base_map[0, 0] == UNKNOWN_VALUE
    assert layers.base_map[0, 1] == FREE_VALUE
    assert layers.base_map[0, 2] == OCCUPIED_VALUE
    assert layers.base_map[0, 3] == FREE_VALUE
    assert bool(layers.candidate_mask[0, 1]) is True
    assert bool(layers.hard_obstacle_mask[0, 2]) is True


def test_bundle_requires_explicit_candidate_policy_for_aisle_conflicts(tmp_path):
    evidence = np.zeros((12, 40), dtype=np.uint8)
    evidence[3:7, 3:37] = EvidenceClass.FREE_CONFIRMED
    evidence[4, 20] = EvidenceClass.OCCUPIED_CONFIRMED
    metadata = GridMetadata(0.10, 0.0, 0.0, width=40, height=12)

    hard = write_semantic_navigation_assets(
        evidence,
        metadata,
        [1.0, 0.0],
        tmp_path / "hard",
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=2.0,
    )
    candidate = write_semantic_navigation_assets(
        evidence,
        metadata,
        [1.0, 0.0],
        tmp_path / "candidate",
        min_longitudinal_support_ratio=0.50,
        min_width_m=0.30,
        min_length_m=2.0,
        occupied_aisle_conflict_policy="candidate",
    )

    assert hard["semantic_labels"][4, 20] == LABEL_OCCUPIED_CONFIRMED
    assert hard["navigation"]["layers"].base_map[4, 20] == OCCUPIED_VALUE
    assert candidate["semantic_labels"][4, 20] == LABEL_OBSTACLE_CANDIDATE
    assert candidate["navigation"]["layers"].base_map[4, 20] == FREE_VALUE
    assert bool(candidate["navigation"]["layers"].candidate_mask[4, 20]) is True
    assert candidate["manifest"]["geometry_policy"]["occupied_aisle_conflict_policy"] == "candidate"
