import numpy as np

from agt_map_reconstruction.maps.headland_depth_evidence import (
    evaluate_headland_depth_evidence,
)
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.structural_endpoint_uncertainty_evidence import (
    summarize_observation_sufficiency_roi,
)


def test_public_single_roi_summary_preserves_existing_metrics():
    base = np.full((4, 5), FREE_VALUE, dtype=np.uint8)
    base[:, :4] = UNKNOWN_VALUE
    ground = np.full(base.shape, np.nan, dtype=float)
    ground[:, 1:4] = 0.0
    scan = np.zeros(base.shape, dtype=np.uint32)
    scan[0, 1] = 1
    scan[0, 2] = 2
    scan[1, 2] = 3
    ray = np.zeros(base.shape, dtype=np.uint32)
    ray[0, 1] = 1
    ray[0, 2] = 5
    roi = np.ones(base.shape, dtype=bool)

    result = summarize_observation_sufficiency_roi(
        base,
        ground,
        scan,
        roi,
        min_repeated_scans=2,
        ray=ray,
    )

    assert result["unknown_cell_count"] == 16
    assert result["trusted_ground_unknown_cell_count"] == 12
    assert np.isclose(result["ground_reference_ceiling_fraction_of_unknown"], 12 / 16)
    assert result["scan_observed_unknown_cell_count"] == 3
    assert np.isclose(result["scan_observed_fraction_of_unknown"], 3 / 16)
    assert result["unknown_repeated_scan_support_cell_count"] == 2
    assert np.isclose(result["repeated_scan_fraction_of_unknown"], 2 / 16)
    assert result["ray_supported_unknown_cell_count"] == 2
    assert np.isclose(result["ray_supported_fraction_of_unknown"], 2 / 16)


def _depth_payload_and_masks(shape=(5, 8)):
    entry_a = np.zeros(shape, dtype=bool)
    entry_b = np.zeros(shape, dtype=bool)
    exit_a = np.zeros(shape, dtype=bool)
    exit_b = np.zeros(shape, dtype=bool)
    entry_boundary = np.zeros(shape, dtype=bool)
    exit_boundary = np.zeros(shape, dtype=bool)
    unresolved = np.zeros(shape, dtype=bool)

    entry_a[:, 0] = True
    entry_b[:, 1] = True
    entry_boundary[:, 2] = True
    unresolved[2, 3:5] = True
    exit_boundary[:, 5] = True
    exit_b[:, 6] = True
    exit_a[:, 7] = True

    payload = {
        "method": "finite_structural_headland_depth_profile",
        "grid_shape_yx": list(shape),
        "depth_edges_m": [0.0, 0.5, 1.0],
        "entry": {
            "boundary_uncertainty_mask_key": "entry_boundary_uncertainty",
            "bands": [
                {
                    "mask_key": "entry_depth_0_0p5",
                    "depth_min_m": 0.0,
                    "depth_max_m": 0.5,
                },
                {
                    "mask_key": "entry_depth_0p5_1",
                    "depth_min_m": 0.5,
                    "depth_max_m": 1.0,
                },
            ],
        },
        "exit": {
            "boundary_uncertainty_mask_key": "exit_boundary_uncertainty",
            "bands": [
                {
                    "mask_key": "exit_depth_0_0p5",
                    "depth_min_m": 0.0,
                    "depth_max_m": 0.5,
                },
                {
                    "mask_key": "exit_depth_0p5_1",
                    "depth_min_m": 0.5,
                    "depth_max_m": 1.0,
                },
            ],
        },
    }
    masks = {
        "entry_depth_0_0p5": entry_a,
        "entry_depth_0p5_1": entry_b,
        "entry_boundary_uncertainty": entry_boundary,
        "exit_depth_0_0p5": exit_a,
        "exit_depth_0p5_1": exit_b,
        "exit_boundary_uncertainty": exit_boundary,
        "structurally_unresolved_cross": unresolved,
    }
    return payload, masks


def test_depth_evaluator_reuses_frozen_arrays_and_keeps_numeric_depth_metadata():
    shape = (5, 8)
    base = np.full(shape, UNKNOWN_VALUE, dtype=np.uint8)
    ground = np.zeros(shape, dtype=float)
    scan = np.zeros(shape, dtype=np.uint32)
    ray = np.zeros(shape, dtype=np.uint32)

    # Entry 0-0.5 m: 5 UNKNOWN cells, all ground eligible, 3 observed, 2 repeated.
    scan[0:3, 0] = [1, 2, 4]
    ray[0:3, 0] = 1
    # Entry 0.5-1 m: deliberately no observation.
    # Exit 0-0.5 m: all 5 repeatedly observed.
    scan[:, 7] = 3
    ray[:, 7] = 2

    payload, masks = _depth_payload_and_masks(shape)
    result = evaluate_headland_depth_evidence(
        base,
        ground,
        scan,
        payload,
        masks,
        min_repeated_scans=2,
        ray_support_count=ray,
    )

    assert result["method"] == "finite_headland_depth_observation_sufficiency"
    assert result["entry"]["bands"][0]["depth_min_m"] == 0.0
    assert result["entry"]["bands"][0]["depth_max_m"] == 0.5
    assert result["entry"]["bands"][0]["scan_observed_unknown_cell_count"] == 3
    assert result["entry"]["bands"][0]["unknown_repeated_scan_support_cell_count"] == 2
    assert result["entry"]["bands"][1]["scan_observed_unknown_cell_count"] == 0
    assert result["exit"]["bands"][0]["unknown_repeated_scan_support_cell_count"] == 5
    assert result["boundary_uncertainty"]["entry"]["roi_cell_count"] == 5
    assert result["boundary_uncertainty"]["exit"]["roi_cell_count"] == 5
    assert result["structurally_unresolved_cross"]["roi_cell_count"] == 2
    assert result["policy"]["frozen_evidence_reused"] is True
    assert result["policy"]["rosbag_replay_performed"] is False
    assert result["policy"]["ray_evidence_regenerated"] is False
    assert result["policy"]["navigation_map_modified"] is False
    assert result["policy"]["semantic_promotion"] is False
