import numpy as np

from agt_map_reconstruction.maps.inferred_lattice_3d_structure import (
    audit_inferred_lattice_3d_structure,
)


def _row(label, center_y, source):
    return {
        "label": label,
        "centerline_xy": [[2.0, center_y], [37.0, center_y]],
        "polygon_xy": [
            [2.0, center_y - 2.0],
            [37.0, center_y - 2.0],
            [37.0, center_y + 2.0],
            [2.0, center_y + 2.0],
        ],
        "geometry_source": source,
    }


def _profile(left, right, v0, v1):
    return {
        "ridge_id": f"R_{left}_{right}",
        "left_aisle_label": left,
        "right_aisle_label": right,
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.10,
        "bin_size_m": 0.50,
        "bin_edges_u_cells": [2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0],
        "bin_center_u_cells": [4.5, 9.5, 14.5, 19.5, 24.5, 29.5, 34.5],
        "ridge_cross_span_cells": [float(v0), float(v1)],
    }


def test_targeted_3d_audit_uses_inferred_slots_only_for_target_selection():
    rows = [
        _row("L01", 5.0, "observed_row_aisle"),
        _row("L02", 15.0, "lattice_inferred_wide_band"),
        _row("L03", 25.0, "lattice_inferred_wide_band"),
    ]
    profiles = [
        _profile("L01", "L02", 7.0, 13.0),
        _profile("L02", "L03", 17.0, 23.0),
    ]
    bundle = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.10,
        "lattice_rows": rows,
        "ridge_profiles": profiles,
        "ridge_terminations": [
            {"ridge_id": "R_L01_L02", "status": "insufficient_structural_support"},
            {"ridge_id": "R_L02_L03", "status": "insufficient_structural_support"},
        ],
    }

    low = np.zeros((30, 40), dtype=np.float64)
    q90 = np.zeros_like(low)
    count = np.full(low.shape, 5, dtype=np.int32)

    result = audit_inferred_lattice_3d_structure(
        bundle,
        low,
        q90,
        count,
        min_points_per_cell=3,
        aisle_reference_half_width_m=0.20,
        min_topographic_relief_m=0.08,
        min_vertical_extent_m=0.15,
        min_support_fraction=0.40,
        min_persistence_m=1.00,
        max_internal_gap_m=0.20,
    )

    assert result["target_ridge_count"] == 2
    assert all(item["status"] == "insufficient_3d_structural_support" for item in result["ridge_audits"])
    assert result["policy"]["inferred_slot_supplies_3d_evidence"] is False
    assert result["policy"]["navigation_map_modified"] is False
    assert result["policy"]["semantic_promotion"] is False


def test_targeted_3d_audit_can_recover_topographic_ridge_support_from_height_grids():
    rows = [
        _row("L01", 5.0, "observed_row_aisle"),
        _row("L02", 15.0, "lattice_inferred_wide_band"),
        _row("L03", 25.0, "observed_row_aisle"),
    ]
    profiles = [
        _profile("L01", "L02", 7.0, 13.0),
        _profile("L02", "L03", 17.0, 23.0),
    ]
    bundle = {
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "resolution_m": 0.10,
        "lattice_rows": rows,
        "ridge_profiles": profiles,
        "ridge_terminations": [
            {"ridge_id": "R_L01_L02", "status": "insufficient_structural_support"},
            {"ridge_id": "R_L02_L03", "status": "insufficient_structural_support"},
        ],
    }

    low = np.zeros((30, 40), dtype=np.float64)
    q90 = np.zeros_like(low)
    count = np.full(low.shape, 5, dtype=np.int32)

    # Both inter-slot ridge bands are elevated by 0.12 m relative to aisle
    # reference strips. There is no vertical-extent cue, so recovery here is
    # specifically from bare-ridge/topographic evidence.
    low[7:13, 8:32] = 0.12
    q90[7:13, 8:32] = 0.12
    low[17:23, 8:32] = 0.12
    q90[17:23, 8:32] = 0.12

    result = audit_inferred_lattice_3d_structure(
        bundle,
        low,
        q90,
        count,
        min_points_per_cell=3,
        aisle_reference_half_width_m=0.20,
        min_topographic_relief_m=0.08,
        min_vertical_extent_m=0.15,
        min_support_fraction=0.40,
        min_persistence_m=1.00,
        max_internal_gap_m=0.20,
    )

    assert result["target_ridge_count"] == 2
    assert all(item["status"] == "ok_3d_structural_support" for item in result["ridge_audits"])
    assert all(item["evidence_summary"]["topographic_supported_bin_count"] > 0 for item in result["ridge_audits"])
    assert result["supported_target_ridge_count"] == 2
