import numpy as np

from agt_map_reconstruction.maps.navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
)
from agt_map_reconstruction.maps.site_interior_flood_fill import (
    build_site_interior_flood_fill,
)


def _closed_room():
    base = np.full((7, 7), UNKNOWN_VALUE, dtype=np.uint8)
    base[1, 1:6] = OCCUPIED_VALUE
    base[5, 1:6] = OCCUPIED_VALUE
    base[1:6, 1] = OCCUPIED_VALUE
    base[1:6, 5] = OCCUPIED_VALUE
    base[3, 3] = FREE_VALUE
    return base


def test_closed_hard_boundary_encloses_nonhard_site_interior():
    base = _closed_room()
    result, masks = build_site_interior_flood_fill(base)

    assert result["status"] == "ok"
    assert result["connectivity"] == 4
    assert result["morphology_applied"] is False
    assert result["interior_nonhard_cell_count"] == 9
    assert result["exterior_reachable_nonhard_cell_count"] == 24
    assert np.all(masks["site_interior_nonhard"][2:5, 2:5])
    assert not np.any(masks["site_interior_nonhard"] & (base == OCCUPIED_VALUE))
    assert not np.any(
        masks["site_interior_nonhard"] & masks["exterior_reachable_nonhard"]
    )
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False


def test_boundary_gap_leaks_flood_fill_and_is_not_auto_closed():
    base = _closed_room()
    base[1, 3] = UNKNOWN_VALUE

    result, masks = build_site_interior_flood_fill(base)

    assert result["status"] == "leaked_or_unenclosed"
    assert result["interior_nonhard_cell_count"] == 0
    assert np.count_nonzero(masks["site_interior_nonhard"]) == 0
    assert result["morphology_applied"] is False
    assert result["automatic_wall_gap_closure"] is False


def test_interior_anchor_detects_leak_even_when_other_enclosed_pockets_remain():
    base = np.full((12, 12), UNKNOWN_VALUE, dtype=np.uint8)
    # Intended site room, but with a real gap at the upper wall.
    base[1, 1:7] = OCCUPIED_VALUE
    base[6, 1:7] = OCCUPIED_VALUE
    base[1:7, 1] = OCCUPIED_VALUE
    base[1:7, 6] = OCCUPIED_VALUE
    base[1, 3] = UNKNOWN_VALUE
    # Separate tiny closed pocket remains elsewhere, reproducing the real failure
    # mode where `interior_count > 0` is not enough to prove the site is enclosed.
    base[7, 7:11] = OCCUPIED_VALUE
    base[10, 7:11] = OCCUPIED_VALUE
    base[7:11, 7] = OCCUPIED_VALUE
    base[7:11, 10] = OCCUPIED_VALUE

    anchors = np.zeros(base.shape, dtype=bool)
    anchors[3, 3] = True

    result, masks = build_site_interior_flood_fill(
        base,
        interior_anchor_mask=anchors,
    )

    assert result["interior_nonhard_cell_count"] > 0
    assert result["interior_component_count"] >= 1
    assert result["interior_anchor_validation_requested"] is True
    assert result["interior_anchor_nonhard_cell_count"] == 1
    assert result["interior_anchor_exterior_reachable_cell_count"] == 1
    assert result["interior_anchor_validation_passed"] is False
    assert result["status"] == "leaked_or_unenclosed"
    assert masks["leak_path"][3, 3]
    assert int(np.count_nonzero(masks["leak_path"])) > 1
    assert result["leak_path_xy"]


def test_enclosed_interior_anchor_passes_validation():
    base = _closed_room()
    anchors = np.zeros(base.shape, dtype=bool)
    anchors[3, 3] = True

    result, masks = build_site_interior_flood_fill(
        base,
        interior_anchor_mask=anchors,
    )

    assert result["status"] == "ok"
    assert result["interior_anchor_validation_passed"] is True
    assert result["interior_anchor_enclosed_cell_count"] == 1
    assert result["interior_anchor_exterior_reachable_cell_count"] == 0
    assert not np.any(masks["leak_path"])


def test_only_canonical_navigation_values_are_accepted():
    base = _closed_room()
    base[0, 0] = 123

    try:
        build_site_interior_flood_fill(base)
    except ValueError as exc:
        assert "gray" in str(exc)
    else:
        raise AssertionError("non-canonical map values must be rejected")
