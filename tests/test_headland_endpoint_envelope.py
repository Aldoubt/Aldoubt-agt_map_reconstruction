import numpy as np

from agt_map_reconstruction.maps.headland_endpoint_envelope import (
    analyze_endpoint_side_envelopes,
)
from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    UNKNOWN_VALUE,
)


def _row(label, y0):
    return {
        "label": label,
        "region_class": "row_aisle",
        "polygon_xy": [[10.0, y0], [40.0, y0], [40.0, y0 + 6.0], [10.0, y0 + 6.0]],
        "centerline_xy": [[10.0, y0 + 3.0], [40.0, y0 + 3.0]],
    }


def _handoff(label, y):
    return {
        "label": label,
        "status": "ok",
        "width_clearance_eligible": True,
        "entry_handoff": {"grid_xy": [11.0, y]},
        "exit_handoff": {"grid_xy": [39.0, y]},
    }


def _inputs():
    rows = [_row("A01", 10.0), _row("A02", 20.0), _row("A03", 30.0)]
    handoffs = [_handoff("A01", 13.0), _handoff("A02", 23.0), _handoff("A03", 33.0)]
    return rows, handoffs


def test_exit_envelope_reports_cross_row_free_component():
    base = np.full((50, 70), FREE_VALUE, dtype=np.uint8)
    rows, handoffs = _inputs()

    result = analyze_endpoint_side_envelopes(
        base,
        rows,
        handoffs,
        resolution=0.10,
        radius_m=0.10,
    )

    exit_side = result["sides"]["exit"]
    assert exit_side["strict"]["best_component"] is not None
    assert exit_side["strict"]["best_component"]["cross_row_coverage_fraction"] > 0.95
    assert exit_side["strict"]["best_component"]["endpoint_distance_median_m"] < 0.25
    assert exit_side["strict"]["best_component"]["max_outward_depth_m"] > 2.0


def test_unknown_gap_appears_as_relaxed_evidence_advantage():
    base = np.full((50, 70), FREE_VALUE, dtype=np.uint8)
    # Unknown band immediately outside the common exit endpoint line.
    base[:, 40:46] = UNKNOWN_VALUE
    rows, handoffs = _inputs()

    result = analyze_endpoint_side_envelopes(
        base,
        rows,
        handoffs,
        resolution=0.10,
        radius_m=0.10,
    )

    exit_side = result["sides"]["exit"]
    strict_best = exit_side["strict"]["best_component"]
    relaxed_best = exit_side["relaxed_unknown_allowed"]["best_component"]
    assert strict_best is not None
    assert relaxed_best is not None
    assert strict_best["endpoint_distance_median_m"] > relaxed_best["endpoint_distance_median_m"]
    assert relaxed_best["unknown_cell_fraction"] > 0.0


def test_width_ineligible_rows_are_excluded_from_endpoint_envelope():
    base = np.full((50, 70), FREE_VALUE, dtype=np.uint8)
    rows, handoffs = _inputs()
    handoffs[0]["width_clearance_eligible"] = False

    result = analyze_endpoint_side_envelopes(
        base,
        rows,
        handoffs,
        resolution=0.10,
        radius_m=0.10,
    )

    assert result["eligible_row_count"] == 2
    assert result["eligible_row_labels"] == ["A02", "A03"]
