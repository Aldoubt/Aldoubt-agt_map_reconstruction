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


def test_only_canonical_navigation_values_are_accepted():
    base = _closed_room()
    base[0, 0] = 123

    try:
        build_site_interior_flood_fill(base)
    except ValueError as exc:
        assert "gray" in str(exc)
    else:
        raise AssertionError("non-canonical map values must be rejected")
