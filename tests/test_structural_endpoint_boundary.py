import numpy as np

from agt_map_reconstruction.maps.structural_endpoint_boundary import (
    fit_structural_endpoint_boundaries,
)


def _record(label, y, entry_x, exit_x, status="ok_bilateral"):
    def _side(x):
        if status != "ok_bilateral":
            return {
                "status": status,
                "structural_grid_xy": None,
                "candidate_grid_xy": [float(x), float(y)],
            }
        return {
            "status": "ok_bilateral",
            "structural_grid_xy": [float(x), float(y)],
            "candidate_grid_xy": [float(x), float(y)],
        }

    return {
        "label": label,
        "entry": _side(entry_x),
        "exit": _side(exit_x),
    }


def test_single_row_length_outlier_does_not_pull_common_boundary():
    records = [
        _record("A01", 10, 10.0, 40.0),
        _record("A02", 20, 10.1, 40.1),
        _record("A03", 30, 9.9, 39.9),
        _record("A04", 40, 10.0, 40.0),
        _record("A05", 50, 35.0, 15.0),  # several metres wrong at 0.1 m/cell
    ]

    result = fit_structural_endpoint_boundaries(
        records,
        row_axis=[1.0, 0.0],
        cross_axis=[0.0, 1.0],
        resolution_m=0.10,
        residual_floor_m=0.30,
        mad_scale=3.0,
        min_inlier_count=3,
    )

    assert result["entry"]["fit_status"] == "ok"
    assert result["exit"]["fit_status"] == "ok"
    assert abs(result["entry"]["fit"]["intercept_u"] - 10.0) < 0.25
    assert abs(result["exit"]["fit"]["intercept_u"] - 40.0) < 0.25

    entry_by_label = {item["label"]: item for item in result["entry"]["rows"]}
    exit_by_label = {item["label"]: item for item in result["exit"]["rows"]}
    assert entry_by_label["A05"]["inlier"] is False
    assert exit_by_label["A05"]["inlier"] is False
    assert entry_by_label["A05"]["residual_m"] > 1.0
    assert exit_by_label["A05"]["residual_m"] < -1.0
    assert result["policy"]["outliers_deleted"] is False
    assert result["policy"]["automatic_acceptance"] is False


def test_ambiguous_rows_are_retained_but_excluded_from_fit():
    records = [
        _record("A01", 10, 10.0, 40.0),
        _record("A02", 20, 10.0, 40.0),
        _record("A03", 30, 10.0, 40.0),
        _record("A04", 40, 25.0, 25.0, status="ambiguous_single_side"),
    ]

    result = fit_structural_endpoint_boundaries(
        records,
        row_axis=[1.0, 0.0],
        cross_axis=[0.0, 1.0],
        resolution_m=0.10,
        residual_floor_m=0.30,
        mad_scale=3.0,
        min_inlier_count=3,
    )

    entry_by_label = {item["label"]: item for item in result["entry"]["rows"]}
    assert entry_by_label["A04"]["source_status"] == "ambiguous_single_side"
    assert entry_by_label["A04"]["used_for_fit"] is False
    assert entry_by_label["A04"]["inlier"] is None
    assert result["entry"]["candidate_count"] == 3
    assert result["entry"]["inlier_count"] == 3


def test_fit_reports_insufficient_candidates_without_fabricating_boundary():
    records = [
        _record("A01", 10, 10.0, 40.0),
        _record("A02", 20, 10.0, 40.0, status="insufficient_structural_support"),
    ]

    result = fit_structural_endpoint_boundaries(
        records,
        row_axis=[1.0, 0.0],
        cross_axis=[0.0, 1.0],
        resolution_m=0.10,
        residual_floor_m=0.30,
        mad_scale=3.0,
        min_inlier_count=3,
    )

    assert result["entry"]["fit_status"] == "insufficient_candidates"
    assert result["entry"]["fit"] is None
    assert result["exit"]["fit_status"] == "insufficient_candidates"
    assert result["exit"]["fit"] is None
