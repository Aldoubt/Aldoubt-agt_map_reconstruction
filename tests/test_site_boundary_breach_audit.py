import numpy as np

from agt_map_reconstruction.maps.navigation_export import OCCUPIED_VALUE, UNKNOWN_VALUE
from agt_map_reconstruction.maps.site_boundary_breach_audit import (
    audit_site_boundary_breaches,
)


def _room(base, y0, y1, x0, x1, *, gap=None):
    base[y0, x0 : x1 + 1] = OCCUPIED_VALUE
    base[y1, x0 : x1 + 1] = OCCUPIED_VALUE
    base[y0 : y1 + 1, x0] = OCCUPIED_VALUE
    base[y0 : y1 + 1, x1] = OCCUPIED_VALUE
    if gap is not None:
        gy, gx = gap
        base[gy, gx] = UNKNOWN_VALUE


def test_audit_separates_leaked_hard_and_enclosed_anchors():
    base = np.full((12, 12), UNKNOWN_VALUE, dtype=np.uint8)
    _room(base, 1, 6, 1, 6, gap=(1, 3))
    _room(base, 7, 10, 7, 10)

    anchors = [
        {"slot_id": "L01", "source": "observed_row_aisle", "grid_xy": [3, 3]},
        {"slot_id": "L02", "source": "observed_row_aisle", "grid_xy": [1, 2]},
        {"slot_id": "L03", "source": "observed_row_aisle", "grid_xy": [8, 8]},
    ]

    result, masks = audit_site_boundary_breaches(
        base,
        anchors,
        resolution_m=0.10,
    )

    assert result["status"] == "breach_confirmed"
    assert result["anchor_count"] == 3
    assert result["leaked_anchor_count"] == 1
    assert result["hard_anchor_count"] == 1
    assert result["enclosed_anchor_count"] == 1
    assert result["anchor_quality_warning"] is True

    by_id = {item["slot_id"]: item for item in result["anchors"]}
    assert by_id["L01"]["classification"] == "exterior_reachable"
    assert by_id["L01"]["path_cell_count"] > 1
    assert by_id["L01"]["path_length_m"] > 0.0
    assert by_id["L01"]["border_exit_xy"] is not None
    assert by_id["L02"]["classification"] == "anchor_on_hard"
    assert by_id["L02"]["border_exit_xy"] is None
    assert by_id["L03"]["classification"] == "enclosed_nonhard"

    assert masks["hard_anchor"][2, 1]
    assert masks["leaked_paths"][3, 3]
    assert result["max_path_support_count"] == 1
    assert result["max_path_support_fraction_of_leaked_anchors"] == 1.0
    assert result["policy"]["automatic_wall_gap_closure"] is False
    assert result["policy"]["navigation_map_modified"] is False
    assert result["policy"]["semantic_promotion"] is False


def test_closed_room_has_no_anchor_reachable_breach():
    base = np.full((7, 7), UNKNOWN_VALUE, dtype=np.uint8)
    _room(base, 1, 5, 1, 5)
    anchors = [
        {"slot_id": "L01", "source": "observed_row_aisle", "grid_xy": [3, 3]},
    ]

    result, masks = audit_site_boundary_breaches(
        base,
        anchors,
        resolution_m=0.05,
    )

    assert result["status"] == "no_anchor_reachable_breach"
    assert result["leaked_anchor_count"] == 0
    assert result["hard_anchor_count"] == 0
    assert result["enclosed_anchor_count"] == 1
    assert result["max_path_support_count"] == 0
    assert not np.any(masks["leaked_paths"])
    assert not np.any(masks["max_path_support"])
