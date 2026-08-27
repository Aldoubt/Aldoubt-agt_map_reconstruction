import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.structural_endpoint_uncertainty_evidence import (
    evaluate_uncertainty_roi_observation_sufficiency,
)


def test_uncertainty_roi_evidence_partitions_unknown_without_mutating_inputs():
    base = np.full((6, 8), FREE_VALUE, dtype=np.uint8)
    base[1:5, 0:4] = UNKNOWN_VALUE
    base[2, 2] = OCCUPIED_VALUE
    ground = np.full(base.shape, np.nan, dtype=float)
    ground[1:5, 1:4] = 0.0
    scan = np.zeros(base.shape, dtype=np.uint32)
    scan[1, 1] = 1
    scan[1, 2] = 2
    scan[3, 3] = 5
    ray = np.zeros(base.shape, dtype=np.uint32)
    ray[1, 2] = 7
    ray[4, 2] = 1

    entry = np.zeros(base.shape, dtype=bool)
    entry[1:5, 0:4] = True
    exit_ = np.zeros(base.shape, dtype=bool)
    exit_[:, 6:] = True
    entry_band = np.zeros(base.shape, dtype=bool)
    entry_band[:, 4] = True
    exit_band = np.zeros(base.shape, dtype=bool)
    exit_band[:, 5] = True
    unresolved = np.zeros(base.shape, dtype=bool)
    unresolved[4, :] = True
    masks = {
        "entry_conservative_outward": entry,
        "exit_conservative_outward": exit_,
        "entry_boundary_uncertainty": entry_band,
        "exit_boundary_uncertainty": exit_band,
        "structurally_unresolved_cross": unresolved,
    }

    before = base.copy()
    result = evaluate_uncertainty_roi_observation_sufficiency(
        base,
        ground,
        scan,
        masks,
        min_repeated_scans=2,
        ray_support_count=ray,
    )

    entry_stats = result["entry"]["conservative_outward"]
    assert entry_stats["unknown_cell_count"] > 0
    assert entry_stats["unknown_no_ground_reference_cell_count"] > 0
    assert entry_stats["unknown_single_scan_support_cell_count"] == 1
    assert entry_stats["unknown_repeated_scan_support_cell_count"] == 2
    assert entry_stats["ray_supported_unknown_cell_count"] == 2
    assert entry_stats["unknown_partition_cell_count"] == entry_stats["unknown_cell_count"]

    trusted_ground = (
        entry_stats["unknown_cell_count"]
        - entry_stats["unknown_no_ground_reference_cell_count"]
    )
    assert entry_stats["trusted_ground_unknown_cell_count"] == trusted_ground
    assert np.isclose(
        entry_stats["ground_reference_ceiling_fraction_of_unknown"],
        trusted_ground / entry_stats["unknown_cell_count"],
    )
    assert entry_stats["scan_observed_unknown_cell_count"] == 3
    assert np.isclose(
        entry_stats["scan_observed_fraction_of_trusted_ground_unknown"],
        3 / trusted_ground,
    )
    assert np.isclose(
        entry_stats["scan_observed_fraction_of_unknown"],
        3 / entry_stats["unknown_cell_count"],
    )
    assert np.isclose(
        entry_stats["repeated_scan_fraction_of_trusted_ground_unknown"],
        2 / trusted_ground,
    )
    assert np.isclose(
        entry_stats["repeated_scan_fraction_of_unknown"],
        2 / entry_stats["unknown_cell_count"],
    )
    assert np.isclose(
        entry_stats["ray_supported_fraction_of_trusted_ground_unknown"],
        2 / trusted_ground,
    )
    assert np.isclose(
        entry_stats["ray_supported_fraction_of_unknown"],
        2 / entry_stats["unknown_cell_count"],
    )

    assert result["policy"]["frozen_evidence_reused"] is True
    assert result["policy"]["ground_reference_ceiling_is_semantic_free"] is False
    assert result["policy"]["evaluation_overlay_only"] is True
    assert result["policy"]["navigation_map_modified"] is False
    assert np.array_equal(base, before)


def test_unresolved_cross_strip_is_reported_separately_not_as_conservative_roi():
    base = np.full((5, 5), UNKNOWN_VALUE, dtype=np.uint8)
    ground = np.zeros(base.shape, dtype=float)
    scan = np.full(base.shape, 3, dtype=np.uint32)
    unresolved = np.zeros(base.shape, dtype=bool)
    unresolved[2, :] = True
    masks = {
        "entry_conservative_outward": np.zeros(base.shape, dtype=bool),
        "exit_conservative_outward": np.zeros(base.shape, dtype=bool),
        "entry_boundary_uncertainty": np.zeros(base.shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(base.shape, dtype=bool),
        "structurally_unresolved_cross": unresolved,
    }

    result = evaluate_uncertainty_roi_observation_sufficiency(
        base,
        ground,
        scan,
        masks,
        min_repeated_scans=2,
    )

    assert result["structurally_unresolved_cross"]["roi_cell_count"] == 5
    assert result["structurally_unresolved_cross"]["unknown_repeated_scan_support_cell_count"] == 5
    assert result["entry"]["conservative_outward"]["roi_cell_count"] == 0
    assert result["entry"]["conservative_outward"]["ground_reference_ceiling_fraction_of_unknown"] is None
    assert result["entry"]["conservative_outward"]["scan_observed_fraction_of_trusted_ground_unknown"] is None
    assert result["entry"]["conservative_outward"]["scan_observed_fraction_of_unknown"] is None
    assert result["policy"]["unresolved_cross_strip_promoted_to_resolved"] is False
