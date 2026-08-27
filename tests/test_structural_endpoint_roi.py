import copy

import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.structural_endpoint_roi import (
    build_repeated_scan_evaluation_overlay,
    build_structural_endpoint_roi,
    evaluate_structural_endpoint_evidence,
)


def _boundary(entry_u=10.0, exit_u=50.0):
    rows = []
    for label, y in (("A01", 15.0), ("A02", 25.0), ("A03", 35.0)):
        rows.append(
            {
                "label": label,
                "entry": {
                    "status": "ok_bilateral",
                    "structural_grid_xy": [entry_u, y],
                },
                "exit": {
                    "status": "ok_bilateral",
                    "structural_grid_xy": [exit_u, y],
                },
            }
        )
    return {
        "schema_version": 1,
        "resolution_m": 0.10,
        "radius_m": 0.20,
        "row_axis_direction": [1.0, 0.0],
        "cross_row_direction": [0.0, 1.0],
        "row_cross_span": [10.0, 40.0],
        "rows": rows,
        "robust_boundary": {
            "entry": {
                "fit_status": "ok",
                "fit": {"slope_du_dv": 0.0, "intercept_u": entry_u},
            },
            "exit": {
                "fit_status": "ok",
                "fit": {"slope_du_dv": 0.0, "intercept_u": exit_u},
            },
        },
    }


def test_structural_roi_is_outward_from_structural_boundary_and_shifts_only_with_geometry():
    shape = (60, 80)
    entry_a = build_structural_endpoint_roi(shape, _boundary(entry_u=10.0), "entry")
    entry_b = build_structural_endpoint_roi(shape, _boundary(entry_u=15.0), "entry")
    exit_ = build_structural_endpoint_roi(shape, _boundary(exit_u=50.0), "exit")

    assert entry_a[20, 5]
    assert not entry_a[20, 12]
    assert entry_b[20, 12]
    assert not entry_b[20, 16]
    assert exit_[20, 55]
    assert not exit_[20, 45]
    assert not entry_a[5, 5]  # outside frozen cross-row span


def test_repeated_scan_overlay_promotes_only_unknown_with_ground_and_support_inside_roi():
    base = np.full((20, 20), FREE_VALUE, dtype=np.uint8)
    base[5:15, 0:8] = UNKNOWN_VALUE
    base[10, 4] = OCCUPIED_VALUE
    ground = np.full(base.shape, np.nan, dtype=float)
    ground[5:15, 0:8] = 0.0
    support = np.zeros(base.shape, dtype=np.uint32)
    support[8, 3] = 2
    support[10, 4] = 10
    support[8, 12] = 10
    roi = np.zeros(base.shape, dtype=bool)
    roi[5:15, 0:10] = True
    original = base.copy()

    overlay, summary = build_repeated_scan_evaluation_overlay(
        base,
        roi,
        ground,
        support,
        min_repeated_scans=2,
    )

    assert overlay[8, 3] == FREE_VALUE
    assert overlay[10, 4] == OCCUPIED_VALUE
    assert overlay[8, 12] == FREE_VALUE  # existing free remains unchanged
    assert np.array_equal(base, original)
    assert summary["promoted_unknown_cell_count"] == 1
    assert summary["supported_occupied_cell_count_ignored"] == 1
    assert summary["navigation_map_modified"] is False
    assert summary["semantic_promotion"] is False


def test_evidence_evaluation_reuses_arrays_without_mutating_boundary_or_map():
    base = np.full((60, 80), FREE_VALUE, dtype=np.uint8)
    base[10:41, 0:10] = UNKNOWN_VALUE
    ground = np.full(base.shape, np.nan, dtype=float)
    ground[10:41, 0:10] = 0.0
    scan = np.zeros(base.shape, dtype=np.uint32)
    scan[10:41, 5:10] = 3
    ray = np.zeros(base.shape, dtype=np.uint32)
    ray[10:41, 5:10] = 7
    boundary = _boundary(entry_u=10.0)
    boundary_before = copy.deepcopy(boundary)
    map_before = base.copy()

    result = evaluate_structural_endpoint_evidence(
        base,
        boundary,
        ground_reference=ground,
        scan_support_count=scan,
        ray_support_count=ray,
        min_repeated_scans=2,
        radius_m=0.20,
    )

    assert result["entry"]["evidence"]["repeated_scan_supported_unknown_cell_count"] > 0
    assert result["entry"]["evidence"]["ray_supported_unknown_cell_count"] > 0
    assert result["entry"]["candidate_overlay"]["navigation_map_modified"] is False
    assert result["policy"]["frozen_evidence_reused"] is True
    assert boundary == boundary_before
    assert np.array_equal(base, map_before)
