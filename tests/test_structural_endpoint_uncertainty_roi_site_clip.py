import numpy as np

from agt_map_reconstruction.maps.structural_endpoint_uncertainty_roi_site_clip import (
    clip_uncertainty_roi_to_site_interior,
)


def test_site_clip_removes_exterior_cells_and_preserves_region_disjointness():
    shape = (5, 8)
    site = np.zeros(shape, dtype=bool)
    site[1:4, 2:6] = True

    masks = {
        "entry_conservative_outward": np.zeros(shape, dtype=bool),
        "entry_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "exit_conservative_outward": np.zeros(shape, dtype=bool),
        "exit_boundary_uncertainty": np.zeros(shape, dtype=bool),
        "structurally_unresolved_cross": np.zeros(shape, dtype=bool),
    }
    masks["entry_conservative_outward"][1, 0:3] = True
    masks["entry_boundary_uncertainty"][1, 3] = True
    masks["exit_boundary_uncertainty"][2, 4] = True
    masks["exit_conservative_outward"][2, 5:8] = True
    masks["structurally_unresolved_cross"][3, 1:7] = True

    result, clipped = clip_uncertainty_roi_to_site_interior(masks, site)

    assert result["method"] == "site_interior_clipped_structural_endpoint_uncertainty_roi"
    assert result["regions"]["entry_conservative_outward"]["original_cell_count"] == 3
    assert result["regions"]["entry_conservative_outward"]["clipped_cell_count"] == 1
    assert result["regions"]["entry_conservative_outward"]["removed_exterior_cell_count"] == 2
    assert result["regions"]["exit_conservative_outward"]["clipped_cell_count"] == 1
    assert result["regions"]["structurally_unresolved_cross"]["clipped_cell_count"] == 4

    total = np.zeros(shape, dtype=np.uint8)
    for mask in clipped.values():
        assert not np.any(mask & ~site)
        total += mask.astype(np.uint8)
    assert int(np.max(total)) <= 1

    assert result["unbounded_roi_preserved"] is True
    assert result["site_interior_mask_is_semantic_free"] is False
    assert result["navigation_map_modified"] is False
    assert result["semantic_promotion"] is False


def test_site_clip_rejects_overlapping_input_regions():
    shape = (3, 3)
    overlap = np.ones(shape, dtype=bool)
    empty = np.zeros(shape, dtype=bool)
    masks = {
        "entry_conservative_outward": overlap,
        "entry_boundary_uncertainty": overlap,
        "exit_conservative_outward": empty,
        "exit_boundary_uncertainty": empty,
        "structurally_unresolved_cross": empty,
    }

    try:
        clip_uncertainty_roi_to_site_interior(masks, np.ones(shape, dtype=bool))
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping structural ROI masks must be rejected")
