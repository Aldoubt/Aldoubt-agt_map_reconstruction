"""Deterministic corrections from manual PCD review of EXP004 routes."""

import json
from pathlib import Path

import cv2
import numpy as np


HARD_LABELS = (2, 4, 6)  # ridge, wall, pillar


def _rectangle_geometry(rectangle):
    polygon = np.asarray(rectangle["polygon_xy"], dtype=float)
    start = 0.5 * (polygon[0] + polygon[3])
    end = 0.5 * (polygon[1] + polygon[2])
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        raise ValueError("aisle rectangle has zero length")
    unit = delta / length
    normal = np.array([-unit[1], unit[0]])
    width = min(abs(np.dot(polygon[3] - polygon[0], normal)),
                abs(np.dot(polygon[2] - polygon[1], normal)))
    return start, end, unit, normal, float(width)


def _inside_scene(point, scene):
    x, y = np.rint(point).astype(int)
    return 0 <= x < scene.shape[1] and 0 <= y < scene.shape[0] and bool(scene[y, x])


def clip_aisle_to_scene(rectangle, scene_mask, *, samples_per_cell=1.0,
                        required_cross_section_fraction=0.80):
    """Trim only aisle endpoints that extend beyond the scene boundary.

    The rectangle remains a rectangle.  Cross-sections are sampled across its
    width; this avoids using a single center point when a corner crosses a wall.
    """
    scene = np.asarray(scene_mask, dtype=bool)
    start, end, unit, normal, width = _rectangle_geometry(rectangle)
    count = max(2, int(np.ceil(np.linalg.norm(end - start) * samples_per_cell)) + 1)
    valid = []
    for fraction in np.linspace(0.0, 1.0, count):
        center = start + fraction * (end - start)
        cross = np.linspace(-0.5, 0.5, 9)
        support = [_inside_scene(center + normal * width * value, scene) for value in cross]
        valid.append(sum(support) / len(support) >= required_cross_section_fraction)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return rectangle, False
    first, last = int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])
    if first == 0 and last == count - 1:
        return rectangle, False
    f0 = first / (count - 1)
    f1 = last / (count - 1)
    new_start = start + f0 * (end - start)
    new_end = start + f1 * (end - start)
    updated = dict(rectangle)
    updated["polygon_xy"] = [
        (new_start - normal * width / 2).tolist(),
        (new_end - normal * width / 2).tolist(),
        (new_end + normal * width / 2).tolist(),
        (new_start + normal * width / 2).tolist(),
    ]
    updated["length_m"] = float(np.linalg.norm(new_end - new_start) *
                                 rectangle.get("length_m", 1.0) /
                                 max(np.linalg.norm(end - start), 1e-9))
    return updated, True


def _rasterize(rectangle, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(np.asarray(rectangle["polygon_xy"], dtype=float)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _station_region(rectangle, shape, station_range):
    """Rasterize a longitudinal subregion of an aisle rectangle."""
    start, end, unit, normal, width = _rectangle_geometry(rectangle)
    count = max(1, int(rectangle.get("control_point_count", 60)) - 1)
    first, last = [float(value) / count for value in station_range]
    first, last = np.clip([first, last], 0.0, 1.0)
    p0 = start + first * (end - start)
    p1 = start + last * (end - start)
    region = dict(rectangle)
    region["polygon_xy"] = [
        (p0 - normal * width / 2).tolist(),
        (p1 - normal * width / 2).tolist(),
        (p1 + normal * width / 2).tolist(),
        (p0 + normal * width / 2).tolist(),
    ]
    return _rasterize(region, shape)


def apply_review_corrections(labels, scene_mask, payload, review):
    """Apply reviewed free-space corrections while preserving hard geometry."""
    result = np.asarray(labels, dtype=np.uint8).copy()
    scene = np.asarray(scene_mask, dtype=bool)
    updated = dict(payload)
    updated["rectangles"] = []
    changes = []
    review_aisles = review.get("aisles", {})
    for rectangle in payload.get("rectangles", []):
        label = rectangle.get("label", f"A{int(rectangle['aisle_id']):02d}")
        item = review_aisles.get(label, {})
        # A wall cell is a boundary for route geometry even when it belongs to
        # the broad scene footprint. This trims end caps that visually extend
        # through the greenhouse wall band.
        clip_scene = scene & (result != 4)
        clipped, was_clipped = clip_aisle_to_scene(rectangle, clip_scene)
        if was_clipped:
            changes.append({"label": label, "change": "clip_to_scene_boundary"})
        area = _rasterize(clipped, result.shape) & scene
        clear_area = area
        if item.get("clear_station_range") is not None:
            clear_area = _station_region(
                clipped, result.shape, item["clear_station_range"]
            ) & scene
        if item.get("review_status") == "pass":
            # Manual review clears debris/vegetation/unknown evidence, but
            # never clears a ridge, wall, or pillar without an explicit reason.
            clearable = area & ~np.isin(result, HARD_LABELS)
            result[clearable] = 1
            changes.append({"label": label, "change": "promote_reviewed_aisle_nonhard_to_free"})
            clear_labels = np.asarray(item.get("clear_labels", []), dtype=np.uint8)
            if clear_labels.size:
                for value in clear_labels:
                    # A label listed explicitly by the human reviewer is an
                    # intentional local override. Unlisted pillars remain
                    # hard geometry; this avoids globally deleting pillars
                    # while allowing confirmed debris/vegetation regions to
                    # be cleared.
                    cells = clear_area & (result == value)
                    result[cells] = 1
                changes.append({"label": label, "change": "clear_reviewed_false_labels",
                                "labels": clear_labels.tolist()})
            if "false ridge" in item.get("reason", "").lower():
                ridge = area & (result == 2)
                result[ridge] = 1
                changes.append({"label": label, "change": "remove_reviewed_false_ridge"})
        updated["rectangles"].append(clipped)
    # Ridges were originally fitted from the vegetation bands and can extend
    # beyond the greenhouse wall even when aisle rectangles are clipped. Keep
    # their semantic identity, but trim only their longitudinal end caps to
    # the same interior scene boundary.
    updated["ridge_rectangles"] = []
    clip_scene = scene & (result != 4)
    for ridge in payload.get("ridge_rectangles", []):
        clipped, was_clipped = clip_aisle_to_scene(ridge, clip_scene)
        if was_clipped:
            changes.append({"label": ridge.get("label", "ridge"),
                            "change": "clip_ridge_to_scene_boundary"})
        updated["ridge_rectangles"].append(clipped)
    updated["aisle_count"] = len(updated["rectangles"])
    updated["review_corrections"] = changes
    return result, updated


def apply_review_route_status(result, review):
    """Apply only explicit human pass/block decisions to route results."""
    result = dict(result)
    review_aisles = review.get("aisles", {})
    for item in result.get("aisles", []):
        decision = review_aisles.get(item.get("label"), {}).get("review_status")
        if decision == "blocked":
            item["passed"] = False
            item["route_recovered_from_b1"] = False
            item["route_points"] = []
            item["control_offsets_m"] = []
            item["failure_reason"] = "manual_review_blocked"
            item["failure_region"] = "manual_review"
        elif decision == "pass" and item.get("failure_reason"):
            item["review_override"] = True
            item["review_reason"] = review_aisles[item["label"]].get("reason", "")
    passed = [item for item in result.get("aisles", []) if item.get("passed")]
    failed = [item for item in result.get("aisles", []) if not item.get("passed")]
    summary = dict(result.get("summary", {}))
    summary.update({
        "pass_count": len(passed),
        "fail_count": len(failed),
        "failed_aisles": [item["label"] for item in failed],
        "manual_review_applied": True,
    })
    result["summary"] = summary
    result["manual_review"] = review
    return result


def load_review(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
