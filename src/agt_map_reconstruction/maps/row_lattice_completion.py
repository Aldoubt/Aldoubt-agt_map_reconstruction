"""Recover a row-lattice hypothesis without promoting weak geometry to free space.

The semantic row-band classifier intentionally turns unusually wide row-aligned
free bands into ``wide_open_area_candidate`` regions.  That is useful for
navigation semantics, but it removes those bands from later structural code if
consumers look only at ``region_class == row_aisle``.  In sparse/bare crop rows,
a wide band can also be the result of several missing ridge observations.

This module adds a separate geometry-only lattice layer.  Stable observed row
aisles estimate the cross-row pitch; expected slots falling inside a wide band
are retained as weak ``lattice_inferred_wide_band`` hypotheses.  These slots do
not modify the navigation map and are never promoted to semantic free space.
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


def _pitch_from_observed(observed):
    if len(observed) < 2:
        return None
    values = []
    ordered = sorted(observed, key=lambda item: int(item["source_band_id"]))
    for left, right in zip(ordered[:-1], ordered[1:]):
        di = int(right["source_band_id"]) - int(left["source_band_id"])
        if di <= 0:
            continue
        dv = float(right["center_v"] - left["center_v"])
        if dv <= 1e-9:
            continue
        values.append(dv / float(di))
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _slot_from_observed(item, lattice_index, resolution_m):
    region = dict(item["region"])
    return {
        "slot_id": f"L{int(lattice_index):02d}",
        "lattice_index": int(lattice_index),
        "center_v_cells": float(item["center_v"]),
        "polygon_xy": region["polygon_xy"],
        "centerline_xy": region["centerline_xy"],
        "width_m": float(item["cross_width_cells"] * resolution_m),
        "source": "observed_row_aisle",
        "evidence_strength": "observed",
        "source_band_id": int(region.get("source_band_id", lattice_index)),
        "source_band_label": str(region.get("source_band_label", region.get("label", ""))),
        "parent_region_label": str(region.get("label", "")),
        "parent_region_class": str(region.get("region_class", "row_aisle")),
        "navigation_free_promoted": False,
    }


def complete_row_lattice(
    regions,
    *,
    resolution_m,
    row_axis=None,
    min_observed_slots=4,
):
    """Complete weak lattice slots inside wide row-aligned bands.

    Only ordinary ``row_aisle`` regions are allowed to estimate lattice pitch.
    ``wide_open_area_candidate`` regions may host inferred slots but never count
    as observed lattice evidence.  Inference is refused when there are too few
    stable observed row aisles.
    """
    regions = [dict(item) for item in regions]
    resolution = float(resolution_m)
    minimum = int(min_observed_slots)
    if resolution <= 0.0:
        raise ValueError("resolution_m must be > 0")
    if minimum < 2:
        raise ValueError("min_observed_slots must be >= 2")
    if not regions:
        return {
            "schema_version": 1,
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
            source_id = int(region.get("source_band_id", region.get("aisle_id", len(observed) + 1)))
            enriched = dict(item)
            enriched["source_band_id"] = source_id
            observed.append(enriched)
        elif region_class == "wide_open_area_candidate":
            wide.append(item)

    observed.sort(key=lambda item: item["center_v"])
    nominal_width_cells = (
        float(np.median([item["cross_width_cells"] for item in observed]))
        if observed
        else None
    )

    slots = []
    if len(observed) < minimum:
        for item in observed:
            source_id = int(item["source_band_id"])
            slots.append(_slot_from_observed(item, source_id, resolution))
        slots.sort(key=lambda item: item["center_v_cells"])
        return {
            "schema_version": 1,
            "status": "insufficient_observed_lattice",
            "row_axis_direction": axis.tolist(),
            "cross_row_direction": cross.tolist(),
            "observed_slot_count": len(observed),
            "wide_region_count": len(wide),
            "nominal_pitch_cells": None,
            "nominal_pitch_m": None,
            "nominal_aisle_width_cells": nominal_width_cells,
            "nominal_aisle_width_m": None if nominal_width_cells is None else nominal_width_cells * resolution,
            "slots": slots,
            "policy": {
                "wide_band_promoted_to_observed_aisle": False,
                "inferred_slot_promoted_to_navigation_free": False,
                "automatic_parameter_selection": False,
                "automatic_acceptance": False,
                "navigation_map_modified": False,
                "semantic_promotion": False,
            },
        }

    pitch = _pitch_from_observed(observed)
    if pitch is None or pitch <= 1e-9:
        raise ValueError("cannot estimate a positive lattice pitch from observed row aisles")

    phase_samples = [
        float(item["center_v"] - int(item["source_band_id"]) * pitch)
        for item in observed
    ]
    phase = float(np.median(np.asarray(phase_samples, dtype=np.float64)))

    occupied_indices = set()
    for item in observed:
        lattice_index = int(np.rint((float(item["center_v"]) - phase) / pitch))
        occupied_indices.add(lattice_index)
        slots.append(_slot_from_observed(item, lattice_index, resolution))

    inferred_by_index = {}
    half_width = 0.5 * float(nominal_width_cells)
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
                "slot_id": f"L{int(lattice_index):02d}",
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
    slots.sort(key=lambda item: item["center_v_cells"])
    return {
        "schema_version": 1,
        "status": "ok",
        "row_axis_direction": axis.tolist(),
        "cross_row_direction": cross.tolist(),
        "observed_slot_count": len(observed),
        "inferred_slot_count": len(inferred_by_index),
        "wide_region_count": len(wide),
        "nominal_pitch_cells": float(pitch),
        "nominal_pitch_m": float(pitch * resolution),
        "lattice_phase_v_cells": float(phase),
        "nominal_aisle_width_cells": float(nominal_width_cells),
        "nominal_aisle_width_m": float(nominal_width_cells * resolution),
        "slots": slots,
        "policy": {
            "wide_band_promoted_to_observed_aisle": False,
            "inferred_slot_promoted_to_navigation_free": False,
            "inferred_slot_role": "geometry_hypothesis_only",
            "automatic_parameter_selection": False,
            "automatic_acceptance": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
