"""Finite endpoint-relative headland depth geometry for P1 evaluation.

Depth is measured outward from the outer edge of the frozen fused structural
endpoint uncertainty band.  The builder uses only frozen structural/lattice
geometry; it does not require a physical greenhouse boundary, inspect map
occupancy classes, modify navigation data, or promote any cell to semantic
free space.
"""

from __future__ import annotations

import numpy as np

from .structural_endpoint_uncertainty_roi import (
    _axes_and_resolution,
    _cross_domain,
    _side_geometry,
    _unresolved_cross_intervals,
)


_DEFAULT_DEPTH_EDGES_M = (0.0, 0.5, 1.0, 2.0, 4.0)


def _validate_depth_edges(depth_edges_m):
    edges = [float(v) for v in depth_edges_m]
    if len(edges) < 2 or not np.isclose(edges[0], 0.0):
        raise ValueError("depth_edges_m must start at 0.0 and contain at least two edges")
    if any(not np.isfinite(v) or v < 0.0 for v in edges):
        raise ValueError("depth_edges_m must be finite and non-negative")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("depth_edges_m must be strictly increasing")
    return edges


def _encode_depth_value(value):
    text = np.format_float_positional(float(value), trim="-")
    if text == "-0":
        text = "0"
    return text.replace(".", "p")


def _depth_mask_key(side, depth_lo_m, depth_hi_m):
    return (
        f"{side}_depth_{_encode_depth_value(depth_lo_m)}_"
        f"{_encode_depth_value(depth_hi_m)}"
    )


def _pairwise_overlap(masks):
    names = list(masks)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if np.any(np.asarray(masks[first], dtype=bool) & np.asarray(masks[second], dtype=bool)):
                return first, second
    return None


def _build_side_depth_masks(
    side,
    *,
    u,
    v,
    resolved_cross,
    uncertainty_payload,
    resolution_m,
    uncertainty_quantile,
    depth_edges_m,
    shape,
):
    geometry = _side_geometry(
        uncertainty_payload,
        side,
        resolution_m,
        uncertainty_quantile,
    )
    center = geometry["slope_du_dv"] * v + geometry["intercept_u"]
    half_cells = geometry["uncertainty_half_width_cells"]
    boundary_uncertainty = resolved_cross & (np.abs(u - center) <= half_cells + 1e-12)

    if side == "entry":
        outward_depth_m = (center - half_cells - u) * float(resolution_m)
    elif side == "exit":
        outward_depth_m = (u - center - half_cells) * float(resolution_m)
    else:
        raise ValueError("side must be entry or exit")

    masks = {f"{side}_boundary_uncertainty": boundary_uncertainty.reshape(shape)}
    bands = []
    for depth_lo, depth_hi in zip(depth_edges_m, depth_edges_m[1:]):
        # depth=0 lies exactly on the outer uncertainty edge and therefore belongs
        # to the boundary-uncertainty partition, never to the first finite band.
        lower_ok = (
            outward_depth_m > 1e-12
            if np.isclose(depth_lo, 0.0)
            else outward_depth_m >= depth_lo - 1e-12
        )
        selected = (
            resolved_cross
            & lower_ok
            & (outward_depth_m < depth_hi - 1e-12)
        )
        key = _depth_mask_key(side, depth_lo, depth_hi)
        masks[key] = selected.reshape(shape)
        bands.append(
            {
                "mask_key": key,
                "depth_min_m": float(depth_lo),
                "depth_max_m": float(depth_hi),
                "cell_count": int(np.count_nonzero(selected)),
            }
        )

    summary = {
        **geometry,
        "depth_zero_definition": "outer_edge_of_fused_structural_uncertainty_band",
        "boundary_uncertainty_mask_key": f"{side}_boundary_uncertainty",
        "boundary_uncertainty_cell_count": int(np.count_nonzero(boundary_uncertainty)),
        "bands": bands,
    }
    return summary, masks


def build_headland_depth_profile(
    fused_bundle,
    uncertainty_payload,
    *,
    grid_shape_yx,
    depth_edges_m=_DEFAULT_DEPTH_EDGES_M,
    uncertainty_quantile="p95",
):
    """Build mutually exclusive finite headland depth masks for entry and exit.

    Depth bands are finite by construction and are restricted to the frozen
    structural cross-row domain. Structurally unresolved ridge cross-spans are
    excluded from every resolved band and emitted separately.
    """
    shape = tuple(int(v) for v in grid_shape_yx)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("grid_shape_yx must be positive (height, width)")
    edges = _validate_depth_edges(depth_edges_m)
    quantile = str(uncertainty_quantile)
    if quantile not in {"p50", "p90", "p95", "max"}:
        raise ValueError("uncertainty_quantile must be p50, p90, p95, or max")

    fused = dict(fused_bundle)
    uncertainty = dict(uncertainty_payload)
    axis, cross, resolution = _axes_and_resolution(fused, uncertainty)
    v_min, v_max = _cross_domain(fused, cross)
    unresolved_ids, unresolved_intervals = _unresolved_cross_intervals(fused)

    yy, xx = np.indices(shape)
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    u = points @ axis
    v = points @ cross
    cross_domain = (v >= v_min - 1e-12) & (v <= v_max + 1e-12)

    unresolved_cross = np.zeros(v.shape, dtype=bool)
    for lo, hi in unresolved_intervals:
        unresolved_cross |= (v >= lo - 1e-12) & (v <= hi + 1e-12)
    unresolved_cross &= cross_domain
    resolved_cross = cross_domain & ~unresolved_cross

    entry_summary, entry_masks = _build_side_depth_masks(
        "entry",
        u=u,
        v=v,
        resolved_cross=resolved_cross,
        uncertainty_payload=uncertainty,
        resolution_m=resolution,
        uncertainty_quantile=quantile,
        depth_edges_m=edges,
        shape=shape,
    )
    exit_summary, exit_masks = _build_side_depth_masks(
        "exit",
        u=u,
        v=v,
        resolved_cross=resolved_cross,
        uncertainty_payload=uncertainty,
        resolution_m=resolution,
        uncertainty_quantile=quantile,
        depth_edges_m=edges,
        shape=shape,
    )

    masks = {
        **entry_masks,
        **exit_masks,
        "structurally_unresolved_cross": unresolved_cross.reshape(shape),
    }
    overlap = _pairwise_overlap(masks)
    if overlap is not None:
        first, second = overlap
        raise ValueError(
            "finite headland profile regions overlap: "
            f"{first} and {second}; reduce max depth or inspect frozen geometry"
        )

    result = {
        "schema_version": 1,
        "method": "finite_structural_headland_depth_profile",
        "grid_shape_yx": list(shape),
        "resolution_m": float(resolution),
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "row_cross_span_cells": [float(v_min), float(v_max)],
        "uncertainty_quantile": quantile,
        "depth_edges_m": edges,
        "max_outward_depth_m": float(edges[-1]),
        "unresolved_ridge_ids": unresolved_ids,
        "unresolved_ridge_count": len(unresolved_ids),
        "structurally_unresolved_cross_cell_count": int(np.count_nonzero(unresolved_cross)),
        "entry": entry_summary,
        "exit": exit_summary,
        "policy": {
            "depth_zero_source": "outer_edge_of_fused_structural_uncertainty_band",
            "depth_zero_boundary_owned_by_uncertainty_band": True,
            "depth_bands_half_open_at_upper_edge": True,
            "finite_outward_extent_enforced": True,
            "physical_site_boundary_required": False,
            "hard_boundary_flood_fill_used": False,
            "unresolved_cross_strip_excluded": True,
            "geometry_only_lattice_supplies_structural_evidence": False,
            "automatic_depth_band_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    return result, masks
