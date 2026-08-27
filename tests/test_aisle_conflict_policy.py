import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    build_navigation_layers,
)
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
