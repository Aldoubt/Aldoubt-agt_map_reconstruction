import numpy as np

from agt_map_reconstruction.maps.structural_endpoint_detection import (
    detect_structural_endpoints,
)


def _profile(left, right, bin_size_m=0.50):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    assert left.shape == right.shape
    n = left.size
    edges = np.arange(n + 1, dtype=float) * 5.0  # 5 grid cells/bin at 0.1 m
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "schema_version": 1,
        "label": "A01",
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "cross_row_span_cells": [25.0, 35.0],
        "resolution_m": 0.10,
        "bin_size_m": bin_size_m,
        "bin_edges_u_cells": edges.tolist(),
        "bin_center_u_cells": centers.tolist(),
        "left_hard_support_fraction": left.tolist(),
        "right_hard_support_fraction": right.tolist(),
        "left_unknown_fraction": np.zeros(n).tolist(),
        "right_unknown_fraction": np.zeros(n).tolist(),
    }


def test_bilateral_persistent_run_stops_before_geometric_aisle_end():
    left = np.zeros(20)
    right = np.zeros(20)
    left[2:15] = 1.0
    right[2:15] = 1.0
    # One short interior gap must be tolerated rather than split the row.
    left[8] = 0.0
    right[8] = 0.0

    result = detect_structural_endpoints(
        _profile(left, right),
        min_support_fraction=0.50,
        min_persistence_m=2.0,
        max_internal_gap_m=0.50,
        max_side_endpoint_disagreement_m=0.75,
    )

    assert result["entry"]["status"] == "ok_bilateral"
    assert result["exit"]["status"] == "ok_bilateral"
    assert np.isclose(result["entry"]["structural_u_cells"], 10.0)
    assert np.isclose(result["exit"]["structural_u_cells"], 75.0)
    assert np.allclose(result["entry"]["structural_grid_xy"], [10.0, 30.0])
    assert np.allclose(result["exit"]["structural_grid_xy"], [75.0, 30.0])
    assert result["policy"]["raw_endpoint_fallback"] is False
    assert result["policy"]["handoff_fallback"] is False


def test_missing_one_structural_side_is_ambiguous_not_fabricated():
    left = np.zeros(20)
    right = np.zeros(20)
    left[3:16] = 1.0

    result = detect_structural_endpoints(
        _profile(left, right),
        min_support_fraction=0.50,
        min_persistence_m=2.0,
        max_internal_gap_m=0.50,
        max_side_endpoint_disagreement_m=0.75,
    )

    assert result["entry"]["status"] == "ambiguous_single_side"
    assert result["exit"]["status"] == "ambiguous_single_side"
    assert result["entry"]["structural_grid_xy"] is None
    assert result["exit"]["structural_grid_xy"] is None
    assert result["entry"]["candidate_source"] == "left_only"
    assert result["exit"]["candidate_source"] == "left_only"


def test_no_persistent_structure_reports_insufficient_support():
    left = np.zeros(20)
    right = np.zeros(20)
    left[4] = 1.0
    right[4] = 1.0

    result = detect_structural_endpoints(
        _profile(left, right),
        min_support_fraction=0.50,
        min_persistence_m=2.0,
        max_internal_gap_m=0.50,
        max_side_endpoint_disagreement_m=0.75,
    )

    assert result["entry"]["status"] == "insufficient_structural_support"
    assert result["exit"]["status"] == "insufficient_structural_support"


def test_entry_and_exit_disagreement_are_classified_independently():
    left = np.zeros(20)
    right = np.zeros(20)
    left[2:16] = 1.0
    right[2:11] = 1.0

    result = detect_structural_endpoints(
        _profile(left, right),
        min_support_fraction=0.50,
        min_persistence_m=2.0,
        max_internal_gap_m=0.50,
        max_side_endpoint_disagreement_m=0.75,
    )

    assert result["entry"]["status"] == "ok_bilateral"
    assert np.isclose(result["entry"]["side_disagreement_m"], 0.0)
    assert result["exit"]["status"] == "ambiguous_single_side"
    assert result["exit"]["candidate_source"] == "side_disagreement"
    assert result["exit"]["side_disagreement_m"] > 0.75
