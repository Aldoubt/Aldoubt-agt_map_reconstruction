import numpy as np

from agt_map_reconstruction.maps.structural_endpoint_uncertainty import (
    build_structural_endpoint_uncertainty_envelope,
)


def _ridge(index, v, entry_u, exit_u, *, status="ok"):
    if status != "ok":
        return {
            "ridge_id": f"R{index}",
            "status": status,
            "entry_u_cells": None,
            "exit_u_cells": None,
            "entry_grid_xy": None,
            "exit_grid_xy": None,
            "resolution_m": 0.10,
        }
    return {
        "ridge_id": f"R{index}",
        "status": "ok",
        "entry_u_cells": float(entry_u),
        "exit_u_cells": float(exit_u),
        "entry_grid_xy": [float(entry_u), float(v)],
        "exit_grid_xy": [float(exit_u), float(v)],
        "resolution_m": 0.10,
    }


def _profile(index, v):
    return {
        "ridge_id": f"R{index}",
        "ridge_cross_span_cells": [float(v - 1.0), float(v + 1.0)],
    }


def test_uncertainty_envelope_preserves_all_supported_ridge_terminations():
    bundle = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.10,
        "ridge_profiles": [
            _profile(1, 0.0),
            _profile(2, 10.0),
            _profile(3, 20.0),
            _profile(4, 30.0),
            _profile(5, 40.0),
        ],
        "ridge_terminations": [
            _ridge(1, 0.0, 10.0, 70.0),
            _ridge(2, 10.0, 11.0, 69.0),
            _ridge(3, 20.0, 12.0, 68.0),
            _ridge(4, 30.0, 30.0, 50.0),
            _ridge(5, 40.0, 0.0, 0.0, status="insufficient_structural_support"),
        ],
        "paired_endpoints": [],
    }

    result = build_structural_endpoint_uncertainty_envelope(bundle)

    assert result["ridge_count"] == 5
    assert result["supported_ridge_count"] == 4
    assert result["unsupported_ridge_count"] == 1
    assert result["entry"]["supported_count"] == 4
    assert result["exit"]["supported_count"] == 4
    assert len(result["entry"]["ridge_points"]) == 4
    assert len(result["exit"]["ridge_points"]) == 4
    assert result["entry"]["trend"]["method"] == "median_pairwise_slope_plus_median_intercept"
    assert result["entry"]["abs_residual_m"]["max"] > result["entry"]["abs_residual_m"]["p50"]
    assert np.isclose(result["entry"]["cross_row_span_fraction"], 0.75)
    assert np.isclose(result["exit"]["cross_row_span_fraction"], 0.75)
    assert result["policy"]["ridge_outliers_deleted"] is False
    assert result["policy"]["bilateral_agreement_required_for_envelope"] is False
    assert result["policy"]["semantic_promotion"] is False


def test_uncertainty_envelope_keeps_bilateral_disagreement_as_uncertainty_metadata():
    bundle = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.10,
        "ridge_profiles": [_profile(1, 0.0), _profile(2, 10.0)],
        "ridge_terminations": [
            _ridge(1, 0.0, 10.0, 70.0),
            _ridge(2, 10.0, 11.0, 69.0),
        ],
        "paired_endpoints": [
            {
                "label": "L02",
                "left_ridge_id": "R1",
                "right_ridge_id": "R2",
                "entry": {
                    "status": "ambiguous_single_side",
                    "candidate_u_cells": 10.5,
                    "candidate_grid_xy": [10.5, 5.0],
                    "side_disagreement_m": 1.20,
                },
                "exit": {
                    "status": "ok_bilateral",
                    "candidate_u_cells": 69.5,
                    "candidate_grid_xy": [69.5, 5.0],
                    "side_disagreement_m": 0.20,
                },
            }
        ],
    }

    result = build_structural_endpoint_uncertainty_envelope(bundle)
    aisle = result["aisle_endpoint_uncertainty"][0]

    assert aisle["label"] == "L02"
    assert aisle["entry"]["evidence_class"] == "bilateral_disagree"
    assert aisle["entry"]["side_disagreement_m"] == 1.20
    assert aisle["exit"]["evidence_class"] == "bilateral_agree"
    assert aisle["exit"]["side_disagreement_m"] == 0.20
