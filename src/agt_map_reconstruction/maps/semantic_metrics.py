"""Ordered geometric metrics for editable agricultural map semantics."""

import csv
import json
from pathlib import Path

import numpy as np


def compute_ridge_metrics(payload):
    """Return ordered ridge dimensions, adjacent spacing, and quality flags."""
    ridges = list(payload.get("ridge_rectangles", []))
    if not ridges:
        return {"summary": {"ridge_count": 0}, "ridges": []}
    rows = []
    reference_normal = None
    for index, ridge in enumerate(ridges, 1):
        polygon = np.asarray(ridge["metric_polygon_xy"], dtype=float)
        center = polygon.mean(axis=0)
        edges = np.roll(polygon, -1, axis=0) - polygon
        lengths = np.linalg.norm(edges, axis=1)
        long_index = int(np.argmax(lengths))
        direction = edges[long_index] / max(lengths[long_index], 1e-9)
        normal = np.array([-direction[1], direction[0]])
        # Polygon winding can reverse when a transformed row crosses a
        # coordinate-axis tie.  Align all normals before projecting the
        # transverse center-to-center distance; otherwise valid neighbors can
        # incorrectly report zero spacing.
        if reference_normal is None:
            reference_normal = normal
        elif float(np.dot(normal, reference_normal)) < 0.0:
            normal = -normal
        rows.append({
            "ridge_id": index,
            "label": f"R{index:02d}",
            "center_x_m": float(center[0]),
            "center_y_m": float(center[1]),
            "length_m": float(ridge.get("length_m", lengths.max())),
            "width_m": float(ridge.get("width_m", lengths.min())),
            "normal": normal,
        })
    spacings = []
    for left, right in zip(rows, rows[1:]):
        normal = left["normal"] + right["normal"]
        normal /= max(np.linalg.norm(normal), 1e-9)
        delta = np.array([right["center_x_m"] - left["center_x_m"],
                          right["center_y_m"] - left["center_y_m"]])
        center_spacing = abs(float(np.dot(delta, normal)))
        clear_gap = center_spacing - (left["width_m"] + right["width_m"]) / 2.0
        spacings.append((center_spacing, clear_gap))
    widths = np.array([row["width_m"] for row in rows])
    lengths = np.array([row["length_m"] for row in rows])
    spacing_values = np.array([item[0] for item in spacings])
    median_width = float(np.median(widths))
    median_spacing = float(np.median(spacing_values)) if len(spacing_values) else None
    for index, row in enumerate(rows):
        row["previous_center_spacing_m"] = None if index == 0 else float(spacings[index - 1][0])
        row["previous_clear_gap_m"] = None if index == 0 else float(spacings[index - 1][1])
        row["next_center_spacing_m"] = None if index == len(rows) - 1 else float(spacings[index][0])
        row["next_clear_gap_m"] = None if index == len(rows) - 1 else float(spacings[index][1])
        row["width_outlier"] = bool(row["width_m"] < 0.5 * median_width or row["width_m"] > 1.5 * median_width)
        adjacent = [value for value in (row["previous_center_spacing_m"], row["next_center_spacing_m"]) if value is not None]
        row["spacing_outlier"] = bool(
            median_spacing is not None and any(abs(value - median_spacing) > 0.30 * median_spacing for value in adjacent)
        )
        row.pop("normal")
    return {
        "summary": {
            "ridge_count": len(rows),
            "median_width_m": median_width,
            "width_range_m": [float(widths.min()), float(widths.max())],
            "median_length_m": float(np.median(lengths)),
            "length_range_m": [float(lengths.min()), float(lengths.max())],
            "median_center_spacing_m": median_spacing,
            "center_spacing_range_m": ([float(spacing_values.min()), float(spacing_values.max())]
                                       if len(spacing_values) else None),
            "width_outlier_count": sum(row["width_outlier"] for row in rows),
            "spacing_outlier_count": sum(row["spacing_outlier"] for row in rows),
        },
        "ridges": rows,
    }


def write_ridge_metrics(payload, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = compute_ridge_metrics(payload)
    (output / "ridge_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fields = [
        "ridge_id", "label", "center_x_m", "center_y_m", "length_m", "width_m",
        "previous_center_spacing_m", "previous_clear_gap_m",
        "next_center_spacing_m", "next_clear_gap_m", "width_outlier", "spacing_outlier",
    ]
    with (output / "ridge_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["ridges"])
    return metrics
