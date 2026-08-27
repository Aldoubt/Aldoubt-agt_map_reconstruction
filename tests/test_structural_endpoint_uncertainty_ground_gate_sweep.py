import numpy as np

from agt_map_reconstruction.maps.structural_endpoint_uncertainty_ground_gate_sweep import (
    sweep_structural_endpoint_uncertainty_ground_gate,
)


def test_uncertainty_roi_ground_gate_sweep_reports_each_frozen_region_separately():
    unknown = np.ones((4, 6), dtype=bool)
    distance = np.array(
        [
            [0.1, 0.4, 0.8, 1.2, 0.2, 0.9],
            [0.2, 0.5, 0.9, 1.4, 0.3, 1.0],
            [0.3, 0.6, 1.0, 1.6, 0.4, 1.1],
            [0.4, 0.7, 1.1, 1.8, 0.5, 1.2],
        ],
        dtype=float,
    )
    disagreement = np.full(unknown.shape, 0.05, dtype=float)
    disagreement[0, 1] = 0.20
    disagreement[2, 4] = 0.20

    entry = np.zeros(unknown.shape, dtype=bool)
    entry[:, :2] = True
    entry_band = np.zeros(unknown.shape, dtype=bool)
    entry_band[:, 2] = True
    exit_band = np.zeros(unknown.shape, dtype=bool)
    exit_band[:, 3] = True
    exit_ = np.zeros(unknown.shape, dtype=bool)
    exit_[:, 4:] = True
    unresolved = np.zeros(unknown.shape, dtype=bool)
    unresolved[2, :] = True

    # Make the partitions disjoint the same way the production ROI builder does.
    for mask in (entry, entry_band, exit_band, exit_):
        mask[2, :] = False

    masks = {
        "entry_conservative_outward": entry,
        "entry_boundary_uncertainty": entry_band,
        "exit_conservative_outward": exit_,
        "exit_boundary_uncertainty": exit_band,
        "structurally_unresolved_cross": unresolved,
    }

    result = sweep_structural_endpoint_uncertainty_ground_gate(
        unknown,
        masks,
        distance,
        disagreement,
        max_support_distances_m=[0.5, 1.0],
        max_model_disagreements_m=[0.10],
    )

    entry_result = result["regions"]["entry_conservative_outward"]
    assert entry_result["unknown_cell_count"] == 6
    assert entry_result["grid"][0]["max_support_distance_m"] == 0.5
    assert entry_result["grid"][0]["max_model_disagreement_m"] == 0.1
    assert entry_result["grid"][0]["accepted_unknown_cell_count"] == 4
    assert np.isclose(entry_result["grid"][0]["accepted_unknown_fraction"], 4 / 6)

    unresolved_result = result["regions"]["structurally_unresolved_cross"]
    assert unresolved_result["unknown_cell_count"] == 6
    assert result["automatic_threshold_selection"] is False
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False


def test_uncertainty_roi_ground_gate_sweep_rejects_overlapping_frozen_regions():
    unknown = np.ones((2, 2), dtype=bool)
    distance = np.zeros((2, 2), dtype=float)
    disagreement = np.zeros((2, 2), dtype=float)
    overlap = np.ones((2, 2), dtype=bool)
    empty = np.zeros((2, 2), dtype=bool)
    masks = {
        "entry_conservative_outward": overlap,
        "entry_boundary_uncertainty": overlap,
        "exit_conservative_outward": empty,
        "exit_boundary_uncertainty": empty,
        "structurally_unresolved_cross": empty,
    }

    try:
        sweep_structural_endpoint_uncertainty_ground_gate(
            unknown,
            masks,
            distance,
            disagreement,
            max_support_distances_m=[0.5],
            max_model_disagreements_m=[0.1],
        )
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping frozen ROI masks must be rejected")
