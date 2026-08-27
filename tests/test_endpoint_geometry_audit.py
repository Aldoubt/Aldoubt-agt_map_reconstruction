import numpy as np

from agt_map_reconstruction.maps.endpoint_geometry_audit import audit_endpoint_geometry


def _row(label, y, reverse=False):
    line = [[10.0, y], [40.0, y]]
    if reverse:
        line = list(reversed(line))
    return {
        "label": label,
        "centerline_xy": line,
        "polygon_xy": [[10.0, y - 2.0], [40.0, y - 2.0], [40.0, y + 2.0], [10.0, y + 2.0]],
    }


def _handoff(label, y, reverse=False):
    entry = [14.0, y]
    exit_ = [35.0, y]
    if reverse:
        entry, exit_ = exit_, entry
    return {
        "label": label,
        "status": "ok",
        "width_clearance_eligible": True,
        "entry_handoff": {"grid_xy": entry, "clearance_m": 0.2},
        "exit_handoff": {"grid_xy": exit_, "clearance_m": 0.2},
        "row_core_fraction": 0.7,
    }


def test_audit_reports_clearance_handoffs_inward_from_raw_endpoints():
    rows = [_row("A01", 10.0), _row("A02", 20.0)]
    handoffs = [_handoff("A01", 10.0), _handoff("A02", 20.0)]

    result = audit_endpoint_geometry(rows, handoffs, resolution_m=0.10)

    assert result["eligible_row_count"] == 2
    assert np.allclose(result["row_axis_direction"], [1.0, 0.0])
    assert result["offset_summary"]["entry_inward"]["median_m"] == 0.4
    assert result["offset_summary"]["exit_inward"]["median_m"] == 0.5
    assert result["raw_endpoint_fit"]["entry"]["intercept_u"] == 10.0
    assert result["clearance_handoff_fit"]["entry"]["intercept_u"] == 14.0
    assert result["raw_endpoint_fit"]["exit"]["intercept_u"] == 40.0
    assert result["clearance_handoff_fit"]["exit"]["intercept_u"] == 35.0
    assert result["policy"]["d3_geometry_modified"] is False


def test_audit_normalizes_reversed_source_row_and_handoff_orientation():
    rows = [_row("A01", 10.0), _row("A02", 20.0, reverse=True)]
    handoffs = [_handoff("A01", 10.0), _handoff("A02", 20.0, reverse=True)]

    result = audit_endpoint_geometry(rows, handoffs, resolution_m=0.10)
    by_label = {item["label"]: item for item in result["rows"]}

    assert by_label["A02"]["source_centerline_forward"] is False
    assert by_label["A02"]["raw_entry_grid_xy"] == [10.0, 20.0]
    assert by_label["A02"]["raw_exit_grid_xy"] == [40.0, 20.0]
    assert by_label["A02"]["handoff_entry_grid_xy"] == [14.0, 20.0]
    assert by_label["A02"]["handoff_exit_grid_xy"] == [35.0, 20.0]


def test_audit_excludes_ineligible_or_missing_handoffs():
    rows = [_row("A01", 10.0), _row("A02", 20.0), _row("A03", 30.0)]
    handoffs = [_handoff("A01", 10.0), _handoff("A02", 20.0)]
    handoffs[1]["width_clearance_eligible"] = False

    result = audit_endpoint_geometry(rows, handoffs, resolution_m=0.10)

    assert result["eligible_row_count"] == 1
    assert result["eligible_row_labels"] == ["A01"]
