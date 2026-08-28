import numpy as np
import pytest

from agt_map_reconstruction.maps.headland_depth_ground_gate_sweep import (
    sweep_headland_depth_ground_gate,
)


def _payload_and_masks():
    shape = (3, 6)
    entry_near = np.zeros(shape, dtype=bool)
    entry_far = np.zeros(shape, dtype=bool)
    exit_near = np.zeros(shape, dtype=bool)
    unresolved = np.zeros(shape, dtype=bool)
    entry_near[:, 0] = True
    entry_far[:, 1] = True
    exit_near[:, 5] = True
    unresolved[:, 3] = True
    payload = {
        "grid_shape_yx": list(shape),
        "entry": {
            "bands": [
                {"mask_key": "entry_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5},
                {"mask_key": "entry_depth_0p5_1", "depth_min_m": 0.5, "depth_max_m": 1.0},
            ]
        },
        "exit": {
            "bands": [
                {"mask_key": "exit_depth_0_0p5", "depth_min_m": 0.0, "depth_max_m": 0.5},
            ]
        },
    }
    masks = {
        "entry_depth_0_0p5": entry_near,
        "entry_depth_0p5_1": entry_far,
        "exit_depth_0_0p5": exit_near,
        "structurally_unresolved_cross": unresolved,
    }
    return payload, masks


def test_ground_gate_sweep_reports_exact_gate_grid_per_finite_depth_band():
    payload, masks = _payload_and_masks()
    shape = (3, 6)
    unknown = np.ones(shape, dtype=bool)
    distance = np.full(shape, 2.0, dtype=float)
    disagreement = np.full(shape, 0.5, dtype=float)

    # Entry near: two cells pass d<=0.5/a<=0.1, all three pass d<=1.0/a<=0.2.
    distance[:, 0] = [0.2, 0.4, 0.8]
    disagreement[:, 0] = [0.05, 0.08, 0.15]
    # Entry far: only one cell passes the tighter gate.
    distance[:, 1] = [0.3, 0.7, 1.2]
    disagreement[:, 1] = [0.05, 0.05, 0.05]

    result = sweep_headland_depth_ground_gate(
        unknown,
        payload,
        masks,
        distance,
        disagreement,
        max_support_distances_m=[0.5, 1.0],
        max_model_disagreements_m=[0.1, 0.2],
    )

    near = result["entry"]["bands"][0]
    assert near["unknown_cell_count"] == 3
    tight = next(
        item
        for item in near["grid"]
        if item["max_support_distance_m"] == 0.5
        and item["max_model_disagreement_m"] == 0.1
    )
    assert tight["accepted_unknown_cell_count"] == 2
    assert np.isclose(tight["accepted_unknown_fraction"], 2 / 3)

    relaxed = next(
        item
        for item in near["grid"]
        if item["max_support_distance_m"] == 1.0
        and item["max_model_disagreement_m"] == 0.2
    )
    assert relaxed["accepted_unknown_cell_count"] == 3
    assert np.isclose(relaxed["accepted_unknown_fraction"], 1.0)

    far = result["entry"]["bands"][1]
    assert far["depth_min_m"] == 0.5
    assert far["depth_max_m"] == 1.0
    assert result["automatic_threshold_selection"] is False
    assert result["physical_site_boundary_required"] is False
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False


def test_ground_gate_sweep_rejects_overlapping_resolved_depth_masks():
    payload, masks = _payload_and_masks()
    masks["entry_depth_0p5_1"][0, 0] = True
    shape = (3, 6)

    with pytest.raises(ValueError, match="overlap"):
        sweep_headland_depth_ground_gate(
            np.ones(shape, dtype=bool),
            payload,
            masks,
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=float),
            max_support_distances_m=[0.5],
            max_model_disagreements_m=[0.1],
        )


def test_ground_gate_sweep_rejects_depth_overlap_with_unresolved_strip():
    payload, masks = _payload_and_masks()
    masks["structurally_unresolved_cross"][0, 0] = True
    shape = (3, 6)

    with pytest.raises(ValueError, match="unresolved"):
        sweep_headland_depth_ground_gate(
            np.ones(shape, dtype=bool),
            payload,
            masks,
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=float),
            max_support_distances_m=[0.5],
            max_model_disagreements_m=[0.1],
        )
