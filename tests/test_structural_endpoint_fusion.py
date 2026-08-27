from agt_map_reconstruction.maps.structural_endpoint_fusion import (
    fuse_structural_endpoint_evidence,
)


def _row(label, y):
    return {
        "label": label,
        "region_class": "row_aisle",
        "polygon_xy": [[0.0, y - 0.4], [10.0, y - 0.4], [10.0, y + 0.4], [0.0, y + 0.4]],
        "centerline_xy": [[0.0, y], [10.0, y]],
    }


def _ridge(ridge_id, left, right, status, entry=None, exit_=None):
    return {
        "ridge_id": ridge_id,
        "left_aisle_label": left,
        "right_aisle_label": right,
        "status": status,
        "resolution_m": 0.1,
        "entry_u_cells": entry,
        "exit_u_cells": exit_,
        "entry_grid_xy": None if entry is None else [entry, 0.0],
        "exit_grid_xy": None if exit_ is None else [exit_, 0.0],
    }


def test_fusion_prefers_pgm_then_endpoint_eligible_3d_and_preserves_local_only_evidence():
    bundle = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.1,
        "lattice_rows": [_row("L01", 0.0), _row("L02", 2.0), _row("L03", 4.0), _row("L04", 6.0)],
        "ridge_profiles": [
            {"ridge_id": "R1", "ridge_cross_span_cells": [0.8, 1.2]},
            {"ridge_id": "R2", "ridge_cross_span_cells": [2.8, 3.2]},
            {"ridge_id": "R3", "ridge_cross_span_cells": [4.8, 5.2]},
        ],
        "ridge_terminations": [
            _ridge("R1", "L01", "L02", "ok", 2.0, 98.0),
            _ridge("R2", "L02", "L03", "insufficient_structural_support"),
            _ridge("R3", "L03", "L04", "insufficient_structural_support"),
        ],
        "parameters": {"max_side_endpoint_disagreement_m": 0.5},
    }
    audit = {
        "ridge_audits": [
            {
                "ridge_id": "R2",
                "left_aisle_label": "L02",
                "right_aisle_label": "L03",
                "status": "ok_3d_structural_support",
                "entry_u_cells": 3.0,
                "exit_u_cells": 97.0,
                "entry_grid_xy": [3.0, 3.0],
                "exit_grid_xy": [97.0, 3.0],
                "structural_span_fraction": 0.94,
                "evidence_summary": {"supported_bin_count": 80},
            },
            {
                "ridge_id": "R3",
                "left_aisle_label": "L03",
                "right_aisle_label": "L04",
                "status": "insufficient_longitudinal_structural_span",
                "entry_u_cells": 45.0,
                "exit_u_cells": 55.0,
                "entry_grid_xy": [45.0, 5.0],
                "exit_grid_xy": [55.0, 5.0],
                "structural_span_fraction": 0.10,
                "evidence_summary": {"supported_bin_count": 12},
            },
        ]
    }

    fused = fuse_structural_endpoint_evidence(bundle, audit)
    by_id = {item["ridge_id"]: item for item in fused["ridge_terminations"]}

    assert by_id["R1"]["status"] == "ok"
    assert by_id["R1"]["evidence_source"] == "pgm_hard"
    assert by_id["R2"]["status"] == "ok"
    assert by_id["R2"]["evidence_source"] == "height_3d"
    assert by_id["R3"]["status"] != "ok"
    assert by_id["R3"]["local_3d_structure_observed"] is True
    assert by_id["R3"]["three_d_status"] == "insufficient_longitudinal_structural_span"

    assert fused["fusion_summary"] == {
        "pgm_supported_ridge_count": 1,
        "three_d_supported_ridge_count": 1,
        "local_3d_only_ridge_count": 1,
        "unresolved_ridge_count": 1,
    }
    assert fused["policy"]["geometry_only_lattice_supplies_structural_evidence"] is False
    assert fused["policy"]["local_3d_structure_promoted_to_endpoint_support"] is False
