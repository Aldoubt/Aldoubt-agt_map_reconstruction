import numpy as np

from agt_map_reconstruction.maps.headland_navigation_gate import (
    build_headland_navigation_gate,
)
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    build_navigation_layers,
)


def _profile(shape):
    return {
        "grid_shape_yx": list(shape),
        "entry": {
            "bands": [
                {
                    "mask_key": "entry_depth_0_0p5",
                    "depth_min_m": 0.0,
                    "depth_max_m": 0.5,
                }
            ],
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
        },
        "exit": {
            "bands": [
                {
                    "mask_key": "exit_depth_0_0p5",
                    "depth_min_m": 0.0,
                    "depth_max_m": 0.5,
                }
            ],
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
        },
    }


def test_headland_gate_promotes_only_explicitly_supported_unknown_cells():
    shape = (5, 8)
    base = np.full(shape, UNKNOWN_VALUE, dtype=np.uint8)
    masks = {key: np.zeros(shape, dtype=bool) for key in (
        "entry_depth_0_0p5",
        "exit_depth_0_0p5",
        "entry_boundary_uncertainty",
        "exit_boundary_uncertainty",
        "structurally_unresolved_cross",
    )}
    masks["entry_depth_0_0p5"][2, 1:5] = True
    masks["exit_depth_0_0p5"][2, 6] = True
    masks["entry_boundary_uncertainty"][1, 1] = True
    masks["structurally_unresolved_cross"][2, 4] = True

    distance = np.full(shape, np.nan, dtype=float)
    disagreement = np.full(shape, np.nan, dtype=float)
    scans = np.zeros(shape, dtype=np.uint16)

    distance[2, 1:5] = [0.20, 0.20, 0.80, 0.20]
    disagreement[2, 1:5] = [0.05, 0.05, 0.05, 0.05]
    scans[2, 1:5] = [1, 0, 1, 1]
    distance[2, 6] = 0.20
    disagreement[2, 6] = 0.05
    scans[2, 6] = 2

    result, trusted, uncertainty = build_headland_navigation_gate(
        base,
        _profile(shape),
        masks,
        distance,
        disagreement,
        scans,
        entry_max_depth_m=0.5,
        exit_max_depth_m=0.0,
        max_support_distance_m=0.5,
        max_model_disagreement_m=0.1,
        min_scan_support=1,
    )

    assert trusted[2, 1]
    assert not trusted[2, 2]  # no scan support
    assert not trusted[2, 3]  # support too far away
    assert not trusted[2, 4]  # structurally unresolved
    assert not trusted[2, 6]  # exit promotion disabled explicitly
    assert uncertainty[1, 1]
    assert uncertainty[2, 4]
    assert result["trusted_free_cell_count"] == 1
    assert result["automatic_threshold_selection"] is False
    assert result["navigation_map_modified"] is False


def _rect(x0, y0, x1, y1):
    return {
        "aisle_id": 1,
        "polygon_xy": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def test_conservative_navigation_layers_preserve_baseline_free_and_veto_uncertain_new_promotion():
    semantic = np.zeros((6, 10), dtype=np.uint8)
    semantic[2, 2] = 1
    semantic[2, 7] = 2
    trusted = np.zeros_like(semantic, dtype=bool)
    trusted[2, 5] = True
    uncertainty = np.zeros_like(semantic, dtype=bool)
    uncertainty[2, 2] = True
    uncertainty[2, 4] = True

    layers = build_navigation_layers(
        semantic,
        [_rect(1, 1, 8, 4)],
        promote_aisle_prior=False,
        trusted_free_mask=trusted,
        uncertainty_mask=uncertainty,
    )

    assert layers.base_map[2, 5] == FREE_VALUE
    assert layers.base_map[2, 4] == UNKNOWN_VALUE  # uncertain non-baseline stays unknown
    assert layers.base_map[2, 2] == FREE_VALUE  # baseline free is never demoted by uncertainty
    assert layers.base_map[2, 7] == OCCUPIED_VALUE
    assert layers.base_map[3, 3] == UNKNOWN_VALUE
