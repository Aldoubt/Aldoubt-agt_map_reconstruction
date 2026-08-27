"""Analyze topology between row-core handoffs and wide open-area candidates."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .navigation_export import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    rasterize_aisles,
)


def _point_component(labels, xy):
    x, y = np.rint(np.asarray(xy, dtype=float)).astype(int)
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def _safe_masks(base_map, resolution, radius_m):
    base = np.asarray(base_map, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("base_map must be 2D")
    if not np.isin(base, [OCCUPIED_VALUE, UNKNOWN_VALUE, FREE_VALUE]).all():
        raise ValueError("base_map contains unsupported gray values")
    if float(resolution) <= 0.0:
        raise ValueError("resolution must be > 0")
    if float(radius_m) < 0.0:
        raise ValueError("radius_m must be >= 0")

    free = base == FREE_VALUE
    hard = base == OCCUPIED_VALUE

    strict_distance = ndimage.distance_transform_edt(free) * float(resolution)
    strict_safe = free & (strict_distance + 1e-12 >= float(radius_m))

    nonhard = ~hard
    hard_distance = ndimage.distance_transform_edt(nonhard) * float(resolution)
    relaxed_safe = nonhard & (hard_distance + 1e-12 >= float(radius_m))
    return strict_safe, relaxed_safe


def _candidate_masks(open_regions, shape):
    result = {}
    for region in open_regions:
        label = str(region.get("label", ""))
        if not label:
            raise ValueError("open-area region missing label")
        result[label] = rasterize_aisles([region], shape)
    return result


def _nearest_candidate(point_xy, candidate_masks, resolution):
    if not candidate_masks:
        return None, None
    x, y = np.rint(np.asarray(point_xy, dtype=float)).astype(int)
    best_label = None
    best_distance = None
    for label, mask in candidate_masks.items():
        distance = ndimage.distance_transform_edt(~mask) * float(resolution)
        if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            value = float("inf")
        else:
            value = float(distance[y, x])
        if best_distance is None or value < best_distance:
            best_label = label
            best_distance = value
    return best_label, best_distance


def _connected_candidates(component_labels, component_id, candidate_masks):
    if component_id <= 0:
        return []
    component = component_labels == int(component_id)
    return sorted(
        label
        for label, mask in candidate_masks.items()
        if np.any(component & mask)
    )


def analyze_handoff_open_area_topology(
    base_map,
    handoffs,
    open_regions,
    resolution,
    radius_m,
):
    """Classify row handoff connectivity to wide open-area candidates.

    Two policies are evaluated without editing the source map:

    - strict: only confirmed free cells are traversable;
    - relaxed diagnostic: unknown is allowed, but hard occupied cells still block.

    A result of ``unknown_bridge_only`` is diagnostic evidence that an open-area
    candidate is topologically reachable only by crossing currently unknown
    space. It is not a navigation acceptance result and must not promote the
    region to ``HEADLAND`` by itself.
    """
    base = np.asarray(base_map, dtype=np.uint8)
    strict_safe, relaxed_safe = _safe_masks(base, resolution, radius_m)
    strict_labels, _ = ndimage.label(strict_safe)
    relaxed_labels, _ = ndimage.label(relaxed_safe)

    candidates = [
        dict(item)
        for item in open_regions
        if str(item.get("region_class", "")) == "wide_open_area_candidate"
    ]
    candidate_masks = _candidate_masks(candidates, base.shape)

    records = []
    for aisle in handoffs:
        label = str(aisle.get("label", ""))
        width_eligible = aisle.get("width_clearance_eligible")
        if width_eligible is False:
            continue
        if str(aisle.get("status", "")) != "ok":
            continue

        for side in ("entry", "exit"):
            pose = aisle.get(f"{side}_handoff") or {}
            xy = pose.get("grid_xy")
            if xy is None:
                continue

            strict_id = _point_component(strict_labels, xy)
            relaxed_id = _point_component(relaxed_labels, xy)
            strict_connected = _connected_candidates(
                strict_labels,
                strict_id,
                candidate_masks,
            )
            relaxed_connected = _connected_candidates(
                relaxed_labels,
                relaxed_id,
                candidate_masks,
            )
            unknown_bridge = sorted(
                label_ for label_ in relaxed_connected if label_ not in strict_connected
            )

            if strict_connected:
                connectivity_class = "strict_connected"
            elif unknown_bridge:
                connectivity_class = "unknown_bridge_only"
            else:
                connectivity_class = "disconnected"

            nearest_label, nearest_distance = _nearest_candidate(
                xy,
                candidate_masks,
                resolution,
            )
            records.append({
                "label": label,
                "side": side,
                "radius_m": float(radius_m),
                "width_clearance_eligible": width_eligible,
                "handoff_grid_xy": [float(xy[0]), float(xy[1])],
                "handoff_map_xy_m": pose.get("map_xy_m"),
                "handoff_cross_track_offset_m": pose.get("cross_track_offset_m"),
                "strict_component_id": int(strict_id),
                "relaxed_component_id": int(relaxed_id),
                "strict_connected_candidates": strict_connected,
                "unknown_bridge_candidates": unknown_bridge,
                "relaxed_connected_candidates": relaxed_connected,
                "connectivity_class": connectivity_class,
                "nearest_open_area_label": nearest_label,
                "nearest_open_area_distance_m": nearest_distance,
            })

    region_summary = []
    for region in candidates:
        region_label = str(region["label"])
        strict_entry = sorted(
            item["label"]
            for item in records
            if item["side"] == "entry"
            and region_label in item["strict_connected_candidates"]
        )
        strict_exit = sorted(
            item["label"]
            for item in records
            if item["side"] == "exit"
            and region_label in item["strict_connected_candidates"]
        )
        unknown_entry = sorted(
            item["label"]
            for item in records
            if item["side"] == "entry"
            and region_label in item["unknown_bridge_candidates"]
        )
        unknown_exit = sorted(
            item["label"]
            for item in records
            if item["side"] == "exit"
            and region_label in item["unknown_bridge_candidates"]
        )
        region_summary.append({
            "label": region_label,
            "region_class": "wide_open_area_candidate",
            "strict_entry_aisles": strict_entry,
            "strict_exit_aisles": strict_exit,
            "unknown_bridge_entry_aisles": unknown_entry,
            "unknown_bridge_exit_aisles": unknown_exit,
            "strict_connection_count": len(strict_entry) + len(strict_exit),
            "unknown_bridge_connection_count": len(unknown_entry) + len(unknown_exit),
            "semantic_promotion": False,
        })

    counts = {
        "strict_connected": int(sum(
            item["connectivity_class"] == "strict_connected" for item in records
        )),
        "unknown_bridge_only": int(sum(
            item["connectivity_class"] == "unknown_bridge_only" for item in records
        )),
        "disconnected": int(sum(
            item["connectivity_class"] == "disconnected" for item in records
        )),
    }
    return {
        "schema_version": 1,
        "radius_m": float(radius_m),
        "policy": {
            "strict": "confirmed free with requested clearance",
            "relaxed_diagnostic": "unknown allowed; hard occupied still blocks",
            "semantic_promotion": False,
        },
        "handoff_count": len(records),
        "open_area_candidate_count": len(candidates),
        "connectivity_counts": counts,
        "handoffs": records,
        "open_area_candidates": region_summary,
    }
