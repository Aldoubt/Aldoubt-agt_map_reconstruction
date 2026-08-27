"""Targeted 3D structural audit for geometry-only inferred lattice regions.

The row lattice decides only where to inspect. Structural support must come
from aligned height/point-count grids. Two complementary, aisle-relative cues
are preserved:

- topographic relief: ridge low-height is elevated relative to adjacent aisle
  low-height reference strips (useful for bare raised beds/ridges);
- vertical-extent contrast: ridge q90-low extent exceeds the adjacent aisle
  q90-low reference extent (useful for vegetation/vertical row structure while
  rejecting canopy/roof extent that is similarly large everywhere).

A short sustained patch proves local 3D structure, but it does not by itself
support both endpoints of a full ridge. Endpoint-eligible 3D evidence therefore
also requires an explicit minimum longitudinal structural span fraction.

Neither cue promotes navigation free space or semantic labels.
"""

from __future__ import annotations

import numpy as np

from .structural_ridge_endpoint import detect_ridge_terminations


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError("direction must contain exactly two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return value / norm


def _validate_grids(low_height, q90_height, point_count):
    low = np.asarray(low_height, dtype=np.float64)
    q90 = np.asarray(q90_height, dtype=np.float64)
    count = np.asarray(point_count)
    if low.ndim != 2 or q90.ndim != 2 or count.ndim != 2:
        raise ValueError("height and point-count grids must be 2D")
    if low.shape != q90.shape or low.shape != count.shape:
        raise ValueError("height and point-count grids must share the same shape")
    return low, q90, count


def _row_center_v(row, cross):
    line = np.asarray(row.get("centerline_xy"), dtype=np.float64)
    if line.shape != (2, 2):
        raise ValueError("lattice row centerline_xy must be 2x2")
    return float(np.mean(line @ cross))


def _target_ridge_ids(bundle):
    rows = {str(item["label"]): item for item in bundle.get("lattice_rows") or []}
    source_status = {
        str(item.get("ridge_id", "")): str(item.get("status", ""))
        for item in bundle.get("ridge_terminations") or []
    }
    targets = []
    for profile in bundle.get("ridge_profiles") or []:
        left = rows.get(str(profile.get("left_aisle_label", "")), {})
        right = rows.get(str(profile.get("right_aisle_label", "")), {})
        inferred = (
            left.get("geometry_source") == "lattice_inferred_wide_band"
            or right.get("geometry_source") == "lattice_inferred_wide_band"
        )
        ridge_id = str(profile.get("ridge_id", ""))
        if inferred and source_status.get(ridge_id) != "ok":
            targets.append(ridge_id)
    return targets


def _audit_profile(
    profile,
    rows_by_label,
    low,
    q90,
    count,
    *,
    axis,
    cross,
    min_points_per_cell,
    aisle_reference_half_width_m,
    min_topographic_relief_m,
    min_vertical_extent_m,
    min_support_fraction,
    min_persistence_m,
    max_internal_gap_m,
    min_structural_span_fraction,
    all_u,
    all_v,
):
    resolution = float(profile["resolution_m"])
    edges = np.asarray(profile["bin_edges_u_cells"], dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("ridge profile must contain bin edges")
    v0, v1 = [float(v) for v in profile["ridge_cross_span_cells"]]
    left = rows_by_label[str(profile["left_aisle_label"])]
    right = rows_by_label[str(profile["right_aisle_label"])]
    left_v = _row_center_v(left, cross)
    right_v = _row_center_v(right, cross)
    ref_half_cells = float(aisle_reference_half_width_m) / resolution

    flat_low = low.ravel()
    flat_q90 = q90.ravel()
    flat_count = count.ravel()
    finite = np.isfinite(flat_low) & np.isfinite(flat_q90)
    count_ok = flat_count >= int(min_points_per_cell)
    valid = finite & count_ok
    flat_vertical_extent = np.maximum(0.0, flat_q90 - flat_low)

    support_fraction = []
    valid_cells = []
    topographic_count = []
    vertical_count = []
    reference_low_median = []
    ridge_low_median = []
    ridge_vertical_extent_median = []
    reference_vertical_extent_median = []
    vertical_extent_contrast_median = []

    ridge_v = (all_v > min(v0, v1) + 1e-12) & (all_v < max(v0, v1) - 1e-12)
    ref_v = (
        (np.abs(all_v - left_v) <= ref_half_cells + 1e-12)
        | (np.abs(all_v - right_v) <= ref_half_cells + 1e-12)
    )

    for index in range(edges.size - 1):
        lo, hi = float(edges[index]), float(edges[index + 1])
        in_u = (all_u >= lo - 1e-12) & (
            (all_u <= hi + 1e-12) if index + 2 == edges.size else (all_u < hi - 1e-12)
        )
        ridge_mask = in_u & ridge_v & valid
        ref_mask = in_u & ref_v & valid
        ridge_values = flat_low[ridge_mask]
        ref_values = flat_low[ref_mask]
        ridge_vertical = flat_vertical_extent[ridge_mask]
        ref_vertical = flat_vertical_extent[ref_mask]
        n = int(ridge_values.size)
        valid_cells.append(n)

        if n == 0 or ref_values.size == 0 or ref_vertical.size == 0:
            support_fraction.append(0.0)
            topographic_count.append(0)
            vertical_count.append(0)
            reference_low_median.append(None)
            ridge_low_median.append(None)
            ridge_vertical_extent_median.append(None)
            reference_vertical_extent_median.append(None)
            vertical_extent_contrast_median.append(None)
            continue

        ref_median = float(np.median(ref_values))
        ref_vertical_median = float(np.median(ref_vertical))
        vertical_contrast = ridge_vertical - ref_vertical_median

        topographic = ridge_values - ref_median >= float(min_topographic_relief_m) - 1e-12
        vertical = vertical_contrast >= float(min_vertical_extent_m) - 1e-12
        structural = topographic | vertical

        support_fraction.append(float(np.count_nonzero(structural) / n))
        topographic_count.append(int(np.count_nonzero(topographic)))
        vertical_count.append(int(np.count_nonzero(vertical)))
        reference_low_median.append(ref_median)
        ridge_low_median.append(float(np.median(ridge_values)))
        ridge_vertical_extent_median.append(float(np.median(ridge_vertical)))
        reference_vertical_extent_median.append(ref_vertical_median)
        vertical_extent_contrast_median.append(float(np.median(vertical_contrast)))

    synthetic = dict(profile)
    synthetic["hard_support_fraction"] = support_fraction
    detected = detect_ridge_terminations(
        synthetic,
        min_support_fraction=float(min_support_fraction),
        min_persistence_m=float(min_persistence_m),
        max_internal_gap_m=float(max_internal_gap_m),
    )

    profile_span_m = max(0.0, float(edges[-1] - edges[0]) * resolution)
    structural_span_m = None
    structural_span_fraction = None
    if detected.get("entry_u_cells") is not None and detected.get("exit_u_cells") is not None:
        structural_span_m = max(
            0.0,
            float(detected["exit_u_cells"] - detected["entry_u_cells"]) * resolution,
        )
        if profile_span_m > 1e-12:
            structural_span_fraction = structural_span_m / profile_span_m

    if detected["status"] != "ok":
        status = "insufficient_3d_structural_support"
    elif structural_span_fraction is None or (
        structural_span_fraction + 1e-12 < float(min_structural_span_fraction)
    ):
        status = "insufficient_longitudinal_structural_span"
    else:
        status = "ok_3d_structural_support"

    supported_bins = int(np.count_nonzero(np.asarray(support_fraction) + 1e-12 >= float(min_support_fraction)))
    return {
        "ridge_id": str(profile["ridge_id"]),
        "left_aisle_label": str(profile["left_aisle_label"]),
        "right_aisle_label": str(profile["right_aisle_label"]),
        "status": status,
        "detector_status": detected["status"],
        "entry_grid_xy": detected.get("entry_grid_xy"),
        "exit_grid_xy": detected.get("exit_grid_xy"),
        "entry_u_cells": detected.get("entry_u_cells"),
        "exit_u_cells": detected.get("exit_u_cells"),
        "profile_span_m": profile_span_m,
        "structural_span_m": structural_span_m,
        "structural_span_fraction": structural_span_fraction,
        "bin_support_fraction": support_fraction,
        "bin_valid_cell_count": valid_cells,
        "bin_topographic_supported_cell_count": topographic_count,
        "bin_vertical_supported_cell_count": vertical_count,
        "bin_reference_low_height_median": reference_low_median,
        "bin_ridge_low_height_median": ridge_low_median,
        "bin_ridge_vertical_extent_median": ridge_vertical_extent_median,
        "bin_reference_vertical_extent_median": reference_vertical_extent_median,
        "bin_vertical_extent_contrast_median": vertical_extent_contrast_median,
        "evidence_summary": {
            "supported_bin_count": supported_bins,
            "supported_bin_fraction": 0.0 if support_fraction == [] else supported_bins / len(support_fraction),
            "topographic_supported_bin_count": int(np.count_nonzero(np.asarray(topographic_count) > 0)),
            "vertical_supported_bin_count": int(np.count_nonzero(np.asarray(vertical_count) > 0)),
            "valid_cell_count": int(np.sum(valid_cells)),
        },
    }


def audit_inferred_lattice_3d_structure(
    structural_bundle,
    low_height,
    q90_height,
    point_count,
    *,
    min_points_per_cell=3,
    aisle_reference_half_width_m=0.20,
    min_topographic_relief_m=0.08,
    min_vertical_extent_m=0.15,
    min_support_fraction=0.40,
    min_persistence_m=1.00,
    max_internal_gap_m=0.20,
    min_structural_span_fraction=0.50,
):
    """Audit only unsupported ridge bands adjacent to inferred lattice slots.

    ``min_vertical_extent_m`` is the required ridge-minus-aisle vertical extent
    contrast, retained under the existing argument name for CLI compatibility.
    ``min_structural_span_fraction`` is an endpoint-quality gate: local 3D
    structure can still be recorded when it fails, but it is not eligible to
    define both ridge endpoints.
    """
    low, q90, count = _validate_grids(low_height, q90_height, point_count)
    bundle = dict(structural_bundle)
    axis = _unit(bundle.get("row_axis_direction"))
    cross = _unit(bundle.get("cross_row_direction"))
    if abs(float(axis @ cross)) > 1e-6:
        cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    resolution = float(bundle.get("resolution_m", 0.0))
    if resolution <= 0.0:
        raise ValueError("structural bundle resolution_m must be > 0")
    if int(min_points_per_cell) < 1:
        raise ValueError("min_points_per_cell must be >= 1")
    if float(aisle_reference_half_width_m) <= 0.0:
        raise ValueError("aisle_reference_half_width_m must be > 0")
    if not 0.0 < float(min_structural_span_fraction) <= 1.0:
        raise ValueError("min_structural_span_fraction must be in (0,1]")

    rows_by_label = {str(item["label"]): item for item in bundle.get("lattice_rows") or []}
    profiles_by_id = {str(item["ridge_id"]): item for item in bundle.get("ridge_profiles") or []}
    target_ids = _target_ridge_ids(bundle)

    yy, xx = np.indices(low.shape)
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    all_u = points @ axis
    all_v = points @ cross

    audits = []
    for ridge_id in target_ids:
        profile = profiles_by_id.get(ridge_id)
        if profile is None:
            raise ValueError(f"missing ridge profile: {ridge_id}")
        audits.append(
            _audit_profile(
                profile,
                rows_by_label,
                low,
                q90,
                count,
                axis=axis,
                cross=cross,
                min_points_per_cell=int(min_points_per_cell),
                aisle_reference_half_width_m=float(aisle_reference_half_width_m),
                min_topographic_relief_m=float(min_topographic_relief_m),
                min_vertical_extent_m=float(min_vertical_extent_m),
                min_support_fraction=float(min_support_fraction),
                min_persistence_m=float(min_persistence_m),
                max_internal_gap_m=float(max_internal_gap_m),
                min_structural_span_fraction=float(min_structural_span_fraction),
                all_u=all_u,
                all_v=all_v,
            )
        )

    supported = sum(1 for item in audits if item["status"] == "ok_3d_structural_support")
    return {
        "schema_version": 3,
        "method": "targeted_inferred_lattice_3d_structural_contrast_audit",
        "grid_shape_yx": list(low.shape),
        "resolution_m": resolution,
        "target_ridge_ids": target_ids,
        "target_ridge_count": len(target_ids),
        "supported_target_ridge_count": supported,
        "unsupported_target_ridge_count": len(target_ids) - supported,
        "ridge_audits": audits,
        "parameters": {
            "min_points_per_cell": int(min_points_per_cell),
            "aisle_reference_half_width_m": float(aisle_reference_half_width_m),
            "min_topographic_relief_m": float(min_topographic_relief_m),
            "min_vertical_extent_contrast_m": float(min_vertical_extent_m),
            "min_support_fraction": float(min_support_fraction),
            "min_persistence_m": float(min_persistence_m),
            "max_internal_gap_m": float(max_internal_gap_m),
            "min_structural_span_fraction": float(min_structural_span_fraction),
        },
        "policy": {
            "target_selection": "unsupported_ridge_adjacent_to_inferred_lattice_slot",
            "inferred_slot_supplies_3d_evidence": False,
            "topographic_cue_is_aisle_relative": True,
            "vertical_extent_cue_is_aisle_relative": True,
            "uniform_vertical_extent_rejected": True,
            "local_3d_structure_does_not_imply_full_ridge_endpoint_support": True,
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
