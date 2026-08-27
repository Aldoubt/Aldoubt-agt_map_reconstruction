import numpy as np

from agt_map_reconstruction.maps.structural_endpoint_uncertainty_roi import (
    build_structural_endpoint_uncertainty_roi,
)


def _fused_bundle():
    return {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 1.0,
        "lattice_rows": [
            {"label": "L1", "polygon_xy": [[0, 1], [9, 1], [9, 2], [0, 2]]},
            {"label": "L2", "polygon_xy": [[0, 4], [9, 4], [9, 5], [0, 5]]},
            {"label": "L3", "polygon_xy": [[0, 7], [9, 7], [9, 8], [0, 8]]},
        ],
        "ridge_profiles": [
            {"ridge_id": "R12", "ridge_cross_span_cells": [2.0, 4.0]},
            {"ridge_id": "R23", "ridge_cross_span_cells": [5.0, 7.0]},
        ],
        "ridge_terminations": [
            {"ridge_id": "R12", "status": "ok", "evidence_source": "pgm_hard"},
            {
                "ridge_id": "R23",
                "status": "insufficient_structural_support",
                "evidence_source": "unresolved",
                "local_3d_structure_observed": True,
            },
        ],
    }


def _uncertainty():
    def side(intercept):
        return {
            "trend_status": "ok",
            "trend": {
                "slope_du_dv": 0.0,
                "intercept_u": float(intercept),
                "center_trend_only": True,
            },
            "abs_residual_m": {
                "p50": 0.5,
                "p90": 0.8,
                "p95": 1.0,
                "max": 1.5,
            },
        }

    return {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 1.0,
        "entry": side(3.0),
        "exit": side(7.0),
    }


def test_roi_uses_p95_band_and_only_clear_outward_cells_are_conservative():
    result, masks = build_structural_endpoint_uncertainty_roi(
        _fused_bundle(), _uncertainty(), grid_shape_yx=(10, 10)
    )

    # entry center=3, p95=1: x=2..4 is uncertainty; x<2 is conservative outward.
    assert masks["entry_conservative_outward"][3, 1]
    assert not masks["entry_conservative_outward"][3, 2]
    assert masks["entry_boundary_uncertainty"][3, 2]
    assert masks["entry_boundary_uncertainty"][3, 3]
    assert masks["entry_boundary_uncertainty"][3, 4]

    # exit center=7, p95=1: x=6..8 is uncertainty; x>8 is conservative outward.
    assert masks["exit_conservative_outward"][3, 9]
    assert not masks["exit_conservative_outward"][3, 8]
    assert masks["exit_boundary_uncertainty"][3, 8]

    assert result["uncertainty_quantile"] == "p95"
    assert result["entry"]["uncertainty_half_width_m"] == 1.0
    assert result["exit"]["uncertainty_half_width_m"] == 1.0
    assert result["policy"]["center_trend_promoted_to_semantic_boundary"] is False
    assert result["policy"]["navigation_map_modified"] is False


def test_roi_excludes_structurally_unresolved_ridge_cross_strip():
    result, masks = build_structural_endpoint_uncertainty_roi(
        _fused_bundle(), _uncertainty(), grid_shape_yx=(10, 10)
    )

    # R23 occupies 5<=cross-v<=7. It remains structurally unresolved even though
    # the global supported cross span may reach both sides of it.
    assert masks["structurally_unresolved_cross"][6, 1]
    assert not masks["entry_conservative_outward"][6, 1]
    assert not masks["exit_conservative_outward"][6, 9]

    # A resolved cross strip remains available to conservative ROI evaluation.
    assert masks["entry_conservative_outward"][3, 1]
    assert result["unresolved_ridge_ids"] == ["R23"]
    assert result["policy"]["unresolved_cross_strip_excluded"] is True
    assert result["policy"]["geometry_only_lattice_supplies_structural_evidence"] is False
    assert result["policy"]["semantic_promotion"] is False
