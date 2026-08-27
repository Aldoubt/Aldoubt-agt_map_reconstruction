"""Clip frozen D3.1 uncertainty ROI masks to flood-filled site interior.

This layer preserves the original unbounded ROI as provenance and only removes
cells outside the enclosed non-HARD site-interior mask.  It does not alter D3.1
structural geometry, resolve structural uncertainty, edit the navigation map, or
promote any mask to semantic free space.
"""

from __future__ import annotations

import numpy as np


_REGION_NAMES = (
    "entry_conservative_outward",
    "entry_boundary_uncertainty",
    "exit_conservative_outward",
    "exit_boundary_uncertainty",
    "structurally_unresolved_cross",
)


def _prepare_masks(roi_masks, site_interior_nonhard):
    site = np.asarray(site_interior_nonhard, dtype=bool)
    if site.ndim != 2:
        raise ValueError("site_interior_nonhard must be 2D")
    prepared = {}
    occupied = np.zeros(site.shape, dtype=np.uint8)
    for name in _REGION_NAMES:
        if name not in roi_masks:
            raise ValueError(f"missing structural ROI mask: {name}")
        mask = np.asarray(roi_masks[name], dtype=bool)
        if mask.shape != site.shape:
            raise ValueError(f"structural ROI mask {name} does not match site mask")
        if np.any(mask & (occupied > 0)):
            raise ValueError(f"structural ROI masks overlap at region {name}")
        occupied[mask] = 1
        prepared[name] = mask
    return prepared, site


def clip_uncertainty_roi_to_site_interior(roi_masks, site_interior_nonhard):
    """Intersect frozen structural ROI partitions with enclosed non-HARD interior."""
    prepared, site = _prepare_masks(roi_masks, site_interior_nonhard)
    clipped = {}
    regions = {}
    for name in _REGION_NAMES:
        original = prepared[name]
        current = original & site
        removed = original & ~site
        original_count = int(np.count_nonzero(original))
        clipped_count = int(np.count_nonzero(current))
        clipped[name] = current
        regions[name] = {
            "original_cell_count": original_count,
            "clipped_cell_count": clipped_count,
            "removed_exterior_cell_count": int(np.count_nonzero(removed)),
            "retained_fraction": (
                0.0 if original_count == 0 else float(clipped_count / original_count)
            ),
        }

    combined = np.zeros(site.shape, dtype=np.uint8)
    for mask in clipped.values():
        combined += mask.astype(np.uint8)
    if np.any(combined > 1):
        raise RuntimeError("site-clipped structural ROI masks are not disjoint")

    result = {
        "schema_version": 1,
        "method": "site_interior_clipped_structural_endpoint_uncertainty_roi",
        "grid_shape_yx": list(site.shape),
        "site_interior_nonhard_cell_count": int(np.count_nonzero(site)),
        "regions": regions,
        "unbounded_roi_preserved": True,
        "site_interior_mask_is_semantic_free": False,
        "structural_geometry_modified": False,
        "structural_uncertainty_modified": False,
        "automatic_acceptance": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
    return result, clipped
