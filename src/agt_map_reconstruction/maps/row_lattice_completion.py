"""Recover a row-lattice hypothesis without promoting weak geometry to free space.

The semantic row-band classifier intentionally turns unusually wide row-aligned
free bands into ``wide_open_area_candidate`` regions. That is useful for
navigation semantics, but it removes those bands from later structural code if
consumers look only at ``region_class == row_aisle``. In sparse/bare crop rows,
a wide band can also be the result of several missing ridge observations.

This module adds a separate geometry-only lattice layer. Stable observed row
aisles estimate the cross-row pitch; expected slots falling inside a wide band
are retained as weak ``lattice_inferred_wide_band`` hypotheses. Source band IDs
are provenance only: integer lattice indices are fitted from cross-row geometry.
Closely spaced observed bands may be grouped as a split/duplicate observation of
one lattice slot. No inferred or grouped slot modifies the navigation map or is
promoted to semantic free space.
"""

from __future__ import annotations

import numpy as np


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.size != 2:
        raise ValueError("direction must contain exactly two values")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    return value / norm


def _common_axis(regions):
    directions = []
    reference = None
    for region in regions:
        line = np.asarray(region.get("centerline_xy"), dtype=np.float64)
        if line.shape != (2, 2):
            continue
        direction = _unit(line[1] - line[0])
        if reference is None:
            reference = direction
        elif float(direction @ reference) < 0.0:
            direction = -direction
        directions.append(direction)
    if not directions:
        raise ValueError("cannot infer row axis from regions")
    return _unit(np.mean(np.stack(directions, axis=0), axis=0))


def _region_geometry(region, axis, cross):
    polygon = np.asarray(region.get("polygon_xy"), dtype=np.float64)
    line = np.asarray(region.get("centerline_xy"), dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 4:
        raise ValueError("region polygon_xy must be Nx2")
    if line.shape != (2, 2):
        raise ValueError("region centerline_xy must be 2x2")
    pu = polygon @ axis
    pv = polygon @ cross
    center_v = float(np.mean(line @ cross))
    return {
        "region": region,
        "u_min": float(np.min(pu)),
        "u_max": float(np.max(pu)),
        "v_min": float(np.min(pv)),
        "v_max": float(np.max(pv)),
        "center_v": center_v,
        "cross_width_cells": float(np.max(pv) - np.min(pv)),
    }


def _polygon_from_uv(u0, u1, v0, v1, axis, cross):
    return np.asarray(
        [
            u0 * axis + v0 * cross,
            u1 * axis + v0 * cross,
            u1 * axis + v1 * cross,
            u0 * axis + v1 * cross,
        ],
        dtype=np.float64,
    )


def _centerline_from_uv(u0, u1, v, axis, cross):
    return np.asarray(
        [u0 * axis + v * cross, u1 * axis + v * cross],
        dtype=np.float64,
    )


def _initial_pitch_from_observed(observed):
    if len(observed) < 2:
        return None
    centers = np.asarray(
        sorted(float(item["center_v"]) for item in observed), dtype=np.float64
    )
    diffs = np.diff(centers)
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return None
    pitch = float(np.median(diffs))
    # One or two split fragments can create abnormally small gaps. Remove only
    # clearly sub-pitch gaps, then recompute the robust initial scale.
    stable = diffs[diffs >= 0.50 * pitch]
    if stable.size >= 2:
        pitch = float(np.median(stable))
    return pitch


def _cluster_observed(observed, initial_pitch, duplicate_gap_ratio):
    ordered = sorted(observed, key=lambda item: float(item["center_v"]))
    if not ordered:
        return []
    threshold = float(duplicate_gap_ratio) * float(initial_pitch)
    clusters = [[ordered[0]]]
    for item in ordered[1:]:
        gap = float(item["center_v"] - clusters[-1][-1]["center_v"])
        if gap < threshold - 1e-12:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _fit_cluster_lattice(clusters, initial_pitch):
    centers = np.asarray(
        [np.median([float(item["center_v"]) for item in cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    if centers.size < 2:
        return None

    indices = [1]
    for gap in np.diff(centers):
        step = max(1, int(np.rint(float(gap) / float(initial_pitch))))
        indices.append(indices[-1] + step)
    indices = np.asarray(indices, dtype=np.int64)

    design = np.column_stack((np.ones(indices.size), indices.astype(np.float64)))
    phase, pitch = np.linalg.lstsq(design, centers, rcond=None)[0]
    pitch = float(pitch)
    phase = float(phase)
    if pitch <= 1e-9:
        return None
    fitted = phase + indices.astype(np.float64) * pitch
    residuals = centers - fitted
    return {
        "centers": centers,
        "indices": indices,
        "phase": phase,
        "pitch": pitch,
        "fitted": fitted,
        "residuals": residuals,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _slot_id(lattice_index):
    value = int(lattice_index)
    if value >= 0:
        return f"L{value:02d}"
    return f"Lm{abs(value):02d}"


def _slot_from_cluster(
    cluster,
    *,
    lattice_index,
    fitted_center_v,
    residual_cells,
    axis,
    cross,
    resolution_m,
):
    u0 = float(np.median([item["u_min"] for item in cluster]))
    u1 = float(np.median([item["u_max"] for item in cluster]))
    width_cells = float(np.median([item["cross_width_cells"] for item in cluster]))
    half_width = 0.5 * width_cells
    polygon = _polygon_from_uv(
        u0,
        u1,
        float(fitted_center_v) - half_width,
        float(fitted_center_v) + half_width,
        axis,
        cross,
    )
    centerline = _centerline_from_uv(u0, u1, float(fitted_center_v), axis, cross)

    regions = [item["region"] for item in cluster]
    ids = [int(region.get("source_band_id", -1)) for region in regions]
    labels = [
        str(region.get("source_band_label", region.get("label", "")))
        for region in regions
    ]
    member_centers = [float(item["center_v"]) for item in cluster]
    split = len(cluster) > 1
    source = "observed_split_group" if split else "observed_row_aisle"
    strength = "observed_split_ambiguous" if split else "observed"
    representative = regions[0]
    return {
        "slot_id": _slot_id(lattice_index),
        "lattice_index": int(lattice_index),
        "center_v_cells": float(fitted_center_v),
        "polygon_xy": polygon.tolist(),
        "centerline_xy": centerline.tolist(),
        "width_m": float(width_cells * resolution_m),
        "source": source,
        "evidence_strength": strength,
        "source_band_id": int(ids[0]),
        "source_band_label": str(labels[0]),
        "source_band_ids": ids,
        "source_band_labels": labels,
        "observed_member_center_v_cells": member_centers,
        "fit_residual_cells": float(residual_cells),
        "fit_residual_m": float(residual_cells * resolution_m),
        "parent_region_label": str(representative.get("label", "")),
        "parent_region_class": str(representative.get("region_class", "row_aisle")),
        "navigation_free_promoted": False,
    }


def _raw_observed_slots(observed, resolution_m):
    slots = []
    for lattice_index, item in enumerate(
        sorted(observed, key=lambda value: float(value["center_v"])), start=1
    ):
        region = item["region"]
        slots.append(
            {
                "slot_id": _slot_id(lattice_index),
                "lattice_index": int(lattice_index),
                "center_v_cells": float(item["center_v"]),
                "polygon_xy": region["polygon_xy"],
                "centerline_xy": region["centerline_xy"],
                "width_m": float(item["cross_width_cells"] * resolution_m),
                "source": "observed_row_aisle",
                "evidence_strength": "observed_unfitted",
                "source_band_id": int(region.get("source_band_id", lattice_index)),
                "source_band_label": str(region.get("source_band_label", region.get("label", ""))),
                "source_band_ids": [int(region.get("source_band_id", lattice_index))],
                "source_band_labels": [str(region.get("source_band_label", region.get("label", "")))],
                "parent_region_label": str(region.get("label", "")),
                "parent_region_class": str(region.get("region_class", "row_aisle")),
                "navigation_free_promoted": False,
            }
        )
    return slots


def complete_row_lattice(
    regions,
    *,
    resolution_m,
    row_axis=None,
    min_observed_slots=4,
    duplicate_gap_ratio=0.50,
    max_fit_residual_ratio=0.25,
):
    """Complete weak lattice slots inside wide row-aligned bands.

    Only ordinary ``row_aisle`` regions are allowed to estimate lattice pitch.
    ``wide_open_area_candidate`` regions may host inferred slots but never count
    as observed lattice evidence. Source band IDs are retained only as
    provenance; lattice indices are fitted from cross-row geometry.
    """
    regions = [dict(item) for item in regions]
    resolution = float(resolution_m)
    minimum = int(min_observed_slots)
    duplicate_ratio = float(duplicate_gap_ratio)
    residual_ratio = float(max_fit_residual_ratio)
    if resolution <= 0.0:
        raise ValueError("resolution_m must be > 0")
    if minimum < 2:
        raise ValueError("min_observed_slots must be >= 2")
    if not 0.0 < duplicate_ratio < 1.0:
        raise ValueError("duplicate_gap_ratio must be in (0, 1)")
    if residual_ratio <= 0.0:
        raise ValueError("max_fit_residual_ratio must be > 0")
    if not regions:
        return {
            "schema_version": 2,
            "status": "insufficient_observed_lattice",
            "nominal_pitch_cells": None,
            "nominal_pitch_m": None,
            "slots": [],
            "policy": {
                "automatic_acceptance": False,
                "navigation_map_modified": False,
                "semantic_promotion": False,
            },
        }

    axis = _common_axis(regions) if row_axis is None else _unit(row_axis)
    cross = np.array([-axis[1], axis[0]], dtype=np.float64)
    geometry = [_region_geometry(item, axis, cross) for item in regions]

    observed = []
    wide = []
    for item in geometry:
        region = item["region"]
        region_class = str(region.get("region_class", ""))
        if region_class == "row_aisle":
            observed.append(item)
        elif region_class == "wide_open_area_candidate":
            wide.append(item)

    observed.sort(key=lambda item: float(item["center_v"]))
    nominal_width_cells = (
        float(np.median([item["cross_width_cells"] for item in observed]))
        if observed
        else None
    )

    if len(observed) < minimum:
        slots = _raw_observed_slots(observed, resolution)
        return {
            "schema_version": 2,
            "status": "insufficient_observed_lattice",
            "row_axis_direction": axis.tolist(),
            "cross_row_direction": cross.tolist(),
            "observed_band_count": len(observed),
            "observed_slot_count": len(observed),
            "duplicate_observed_band_count": 0,
            "duplicate_observed_groups": [],
            "wide_region_count": len(wide),
            "nominal_pitch_cells": None,
            "nominal_pitch_m": None,
            "nominal_aisle_width_cells": nominal_width_cells,
            "nominal_aisle_width_m": None if nominal_width_cells is None else nominal_width_cells * resolution,
            "slots": slots,
            "policy": {
                "wide_band_promoted_to_observed_aisle": False,
                "inferred_slot_promoted_to_navigation_free": False,
                "duplicate_gap_ratio": duplicate_ratio,
                "max_fit_residual_ratio": residual_ratio,
                "automatic_parameter_selection": False,
                "automatic_acceptance": False,
                "navigation_map_modified": False,
                "semantic_promotion": False,
            },
        }

    initial_pitch = _initial_pitch_from_observed(observed)
    if initial_pitch is None or initial_pitch <= 1e-9:
        raise ValueError("cannot estimate a positive initial lattice pitch")
    clusters = _cluster_observed(observed, initial_pitch, duplicate_ratio)
    if len(clusters) < minimum:
        slots = _raw_observed_slots(observed, resolution)
        return {
            "schema_version": 2,
            "status": "insufficient_unique_observed_slots",
            "row_axis_direction": axis.tolist(),
            "cross_row_direction": cross.tolist(),
            "observed_band_count": len(observed),
            "observed_slot_count": len(clusters),
            "duplicate_observed_band_count": len(observed) - len(clusters),
            "duplicate_observed_groups": [],
            "wide_region_count": len(wide),
            "nominal_pitch_cells": None,
            "nominal_pitch_m": None,
            "slots": slots,
            "policy": {
                "automatic_parameter_selection": False,
                "automatic_acceptance": False,
                "navigation_map_modified": False,
                "semantic_promotion": False,
            },
        }

    fit = _fit_cluster_lattice(clusters, initial_pitch)
    if fit is None:
        raise ValueError("cannot fit a positive row lattice")

    duplicate_groups = []
    for cluster, lattice_index in zip(clusters, fit["indices"]):
        if len(cluster) <= 1:
            continue
        regions_in_group = [item["region"] for item in cluster]
        centers = [float(item["center_v"]) for item in cluster]
        duplicate_groups.append(
            {
                "lattice_index": int(lattice_index),
                "source_band_ids": [int(region.get("source_band_id", -1)) for region in regions_in_group],
                "source_band_labels": [
                    str(region.get("source_band_label", region.get("label", "")))
                    for region in regions_in_group
                ],
                "member_center_v_cells": centers,
                "cross_span_cells": float(max(centers) - min(centers)),
                "cross_span_m": float((max(centers) - min(centers)) * resolution),
            }
        )

    fit_ok = fit["max_abs_residual"] <= residual_ratio * fit["pitch"] + 1e-12
    slots = []
    for cluster, lattice_index, fitted_v, residual in zip(
        clusters,
        fit["indices"],
        fit["fitted"],
        fit["residuals"],
    ):
        slots.append(
            _slot_from_cluster(
                cluster,
                lattice_index=int(lattice_index),
                fitted_center_v=float(fitted_v),
                residual_cells=float(residual),
                axis=axis,
                cross=cross,
                resolution_m=resolution,
            )
        )

    inferred_by_index = {}
    if fit_ok:
        occupied_indices = set(int(value) for value in fit["indices"])
        half_width = 0.5 * float(nominal_width_cells)
        phase = float(fit["phase"])
        pitch = float(fit["pitch"])
        for parent in wide:
            k_min = int(np.ceil((float(parent["v_min"]) - phase) / pitch - 1e-12))
            k_max = int(np.floor((float(parent["v_max"]) - phase) / pitch + 1e-12))
            for lattice_index in range(k_min, k_max + 1):
                if lattice_index in occupied_indices or lattice_index in inferred_by_index:
                    continue
                center_v = phase + float(lattice_index) * pitch
                if center_v < float(parent["v_min"]) - 1e-9 or center_v > float(parent["v_max"]) + 1e-9:
                    continue
                polygon = _polygon_from_uv(
                    float(parent["u_min"]),
                    float(parent["u_max"]),
                    center_v - half_width,
                    center_v + half_width,
                    axis,
                    cross,
                )
                centerline = _centerline_from_uv(
                    float(parent["u_min"]),
                    float(parent["u_max"]),
                    center_v,
                    axis,
                    cross,
                )
                region = parent["region"]
                inferred_by_index[lattice_index] = {
                    "slot_id": _slot_id(lattice_index),
                    "lattice_index": int(lattice_index),
                    "center_v_cells": float(center_v),
                    "polygon_xy": polygon.tolist(),
                    "centerline_xy": centerline.tolist(),
                    "width_m": float(nominal_width_cells * resolution),
                    "source": "lattice_inferred_wide_band",
                    "evidence_strength": "weak_inferred",
                    "source_band_id": int(region.get("source_band_id", -1)),
                    "source_band_label": str(region.get("source_band_label", "")),
                    "parent_region_label": str(region.get("label", "")),
                    "parent_region_class": "wide_open_area_candidate",
                    "navigation_free_promoted": False,
                }

    slots.extend(inferred_by_index.values())
    slots.sort(key=lambda item: float(item["center_v_cells"]))
    return {
        "schema_version": 2,
        "status": "ok" if fit_ok else "poor_observed_lattice_fit",
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "observed_band_count": len(observed),
        "observed_slot_count": len(clusters),
        "duplicate_observed_band_count": len(observed) - len(clusters),
        "duplicate_observed_groups": duplicate_groups,
        "inferred_slot_count": len(inferred_by_index),
        "wide_region_count": len(wide),
        "initial_pitch_cells": float(initial_pitch),
        "nominal_pitch_cells": float(fit["pitch"]),
        "nominal_pitch_m": float(fit["pitch"] * resolution),
        "lattice_phase_v_cells": float(fit["phase"]),
        "fit_rmse_cells": float(fit["rmse"]),
        "fit_rmse_m": float(fit["rmse"] * resolution),
        "fit_max_abs_residual_cells": float(fit["max_abs_residual"]),
        "fit_max_abs_residual_m": float(fit["max_abs_residual"] * resolution),
        "nominal_aisle_width_cells": float(nominal_width_cells),
        "nominal_aisle_width_m": float(nominal_width_cells * resolution),
        "slots": slots,
        "policy": {
            "source_band_id_is_lattice_index": False,
            "wide_band_promoted_to_observed_aisle": False,
            "inferred_slot_promoted_to_navigation_free": False,
            "inferred_slot_role": "geometry_hypothesis_only",
            "split_observed_slot_role": "geometry_hypothesis_only",
            "duplicate_gap_ratio": duplicate_ratio,
            "max_fit_residual_ratio": residual_ratio,
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
