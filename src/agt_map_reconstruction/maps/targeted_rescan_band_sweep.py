"""Outward-depth band diagnostics for targeted endpoint reacquisition.

The frozen P1-D3 endpoint geometry defines the ROI and outward distance.  This
module only summarizes how much unresolved acquisition requirement lies within
user-supplied cumulative depth bands.  It never selects a band automatically,
modifies the navigation map, or promotes semantic free space.
"""

from __future__ import annotations

import numpy as np

from .targeted_rescan_requirement import summarize_targeted_rescan_requirement


def summarize_targeted_rescan_depth_bands(
    requirement_labels,
    endpoint_roi,
    outward_depth_cells,
    *,
    resolution_m,
    max_outward_depth_m_values,
):
    """Summarize cumulative endpoint requirement bands at explicit depths."""
    labels = np.asarray(requirement_labels, dtype=np.uint8)
    roi = np.asarray(endpoint_roi, dtype=bool)
    depth_cells = np.asarray(outward_depth_cells, dtype=np.float64)
    if labels.ndim != 2 or roi.shape != labels.shape or depth_cells.shape != labels.shape:
        raise ValueError("requirement_labels/endpoint_roi/outward_depth_cells shape mismatch")
    resolution = float(resolution_m)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution_m must be positive and finite")

    values = [float(value) for value in max_outward_depth_m_values]
    if not values or any((not np.isfinite(value) or value <= 0.0) for value in values):
        raise ValueError("max_outward_depth_m_values must contain positive finite values")
    if len(set(values)) != len(values):
        raise ValueError("max_outward_depth_m_values must be unique")

    side_total = int(np.count_nonzero(roi))
    depth_m = depth_cells * resolution
    bands = []
    for max_depth_m in values:
        band = roi & (depth_m <= max_depth_m + 1e-12)
        summary = summarize_targeted_rescan_requirement(labels, roi_mask=band)
        summary.update(
            {
                "max_outward_depth_m": max_depth_m,
                "band_fraction_of_endpoint_roi": (
                    float(summary["roi_cell_count"] / side_total) if side_total else 0.0
                ),
                "rescan_required_fraction_of_endpoint_roi": (
                    float(summary["rescan_required_cell_count"] / side_total)
                    if side_total
                    else 0.0
                ),
                "repeated_scan_anchor_fraction_of_endpoint_roi": (
                    float(summary["repeated_scan_anchor_cell_count"] / side_total)
                    if side_total
                    else 0.0
                ),
            }
        )
        bands.append(summary)

    return {
        "schema_version": 1,
        "resolution_m": resolution,
        "endpoint_roi_cell_count": side_total,
        "bands": bands,
        "automatic_band_selection": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    }
