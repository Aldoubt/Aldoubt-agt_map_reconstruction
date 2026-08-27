import numpy as np

from agt_map_reconstruction.maps.structural_lattice_endpoint import (
    build_lattice_structural_endpoint_bundle,
    lattice_slots_to_rows,
)


def _slot(index, center_y, *, source="observed_row_aisle", width=4.0):
    return {
        "slot_id": f"L{index:02d}",
        "lattice_index": index,
        "center_v_cells": float(center_y),
        "polygon_xy": [
            [2.0, center_y - width / 2.0],
            [77.0, center_y - width / 2.0],
            [77.0, center_y + width / 2.0],
            [2.0, center_y + width / 2.0],
        ],
        "centerline_xy": [[2.0, center_y], [77.0, center_y]],
        "width_m": width * 0.10,
        "source": source,
        "evidence_strength": "observed" if source == "observed_row_aisle" else "weak_inferred",
        "source_band_label": f"A{index:02d}",
        "navigation_free_promoted": False,
    }


def _lattice(slots):
    return {
        "schema_version": 1,
        "status": "ok",
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "slots": slots,
        "policy": {
            "inferred_slot_promoted_to_navigation_free": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }


def test_lattice_slots_become_geometry_rows_without_semantic_promotion():
    rows = lattice_slots_to_rows(
        _lattice(
            [
                _slot(1, 6.0),
                _slot(2, 16.0, source="lattice_inferred_wide_band"),
                _slot(3, 26.0),
            ]
        )
    )

    assert [row["label"] for row in rows] == ["L01", "L02", "L03"]
    assert rows[1]["geometry_source"] == "lattice_inferred_wide_band"
    assert rows[1]["geometry_only"] is True
    assert rows[1]["navigation_free_promoted"] is False
    assert all(row["region_class"] == "row_aisle" for row in rows)


def test_inferred_slot_defines_search_geometry_but_does_not_create_structure_evidence():
    free = 254
    base = np.full((40, 80), free, dtype=np.uint8)
    lattice = _lattice(
        [
            _slot(1, 6.0),
            _slot(2, 16.0, source="lattice_inferred_wide_band"),
            _slot(3, 26.0),
        ]
    )

    result = build_lattice_structural_endpoint_bundle(
        base,
        lattice,
        resolution_m=0.10,
        bin_size_m=0.10,
        min_support_fraction=0.50,
        min_persistence_m=0.80,
        max_internal_gap_m=0.20,
        max_side_endpoint_disagreement_m=0.50,
        residual_floor_m=0.30,
        mad_scale=3.0,
        min_inlier_count=2,
        max_fit_rmse_m=0.50,
    )

    assert result["lattice_slot_count"] == 3
    assert result["inferred_slot_count"] == 1
    assert result["ridge_profile_count"] == 2
    assert all(item["status"] == "insufficient_structural_support" for item in result["ridge_terminations"])
    assert result["robust_boundary"]["entry"]["fit_status"] == "insufficient_candidates"
    assert result["policy"]["inferred_slot_supplies_structural_evidence"] is False
    assert result["policy"]["navigation_map_modified"] is False
    assert result["policy"]["semantic_promotion"] is False


def test_real_inter_slot_hard_ridges_can_support_endpoint_fit_across_inferred_slot_geometry():
    free, hard = 254, 0
    base = np.full((40, 80), free, dtype=np.uint8)
    lattice = _lattice(
        [
            _slot(1, 6.0),
            _slot(2, 16.0, source="lattice_inferred_wide_band"),
            _slot(3, 26.0),
            _slot(4, 36.0),
        ]
    )

    # HARD evidence lies only in the inter-slot ridge bands. The inferred aisle
    # at L02 contributes geometry, not occupancy evidence.
    base[9:14, 10:62] = hard
    base[19:24, 11:61] = hard
    base[29:34, 12:60] = hard

    result = build_lattice_structural_endpoint_bundle(
        base,
        lattice,
        resolution_m=0.10,
        bin_size_m=0.10,
        min_support_fraction=0.50,
        min_persistence_m=0.80,
        max_internal_gap_m=0.20,
        max_side_endpoint_disagreement_m=0.50,
        residual_floor_m=0.30,
        mad_scale=3.0,
        min_inlier_count=2,
        max_fit_rmse_m=0.50,
    )

    by_label = {item["label"]: item for item in result["paired_endpoints"]}
    assert by_label["L02"]["entry"]["status"] == "ok_bilateral"
    assert by_label["L02"]["exit"]["status"] == "ok_bilateral"
    assert result["robust_boundary"]["entry"]["fit_status"] == "ok"
    assert result["robust_boundary"]["exit"]["fit_status"] == "ok"
    assert result["policy"]["inferred_slot_promoted_to_navigation_free"] is False
