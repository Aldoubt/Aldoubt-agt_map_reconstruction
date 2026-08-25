"""Export semantic row/aisle segmentation to navigation and review assets."""

from pathlib import Path
import json

import numpy as np
import yaml
from scipy import ndimage
from skimage.measure import find_contours


def _row_to_original(row, column, shape, angle):
    """Map a point from the row-aligned array back to source raster coordinates."""
    height, width = shape
    center_row = (height - 1) / 2.0
    center_column = (width - 1) / 2.0
    dr = float(row) - center_row
    dc = float(column) - center_column
    cosine, sine = np.cos(angle), np.sin(angle)
    original_row = center_row + cosine * dr + sine * dc
    original_column = center_column - sine * dr + cosine * dc
    return [float(original_column), float(original_row)]


def _metric_polygon(polygon, origin_xy, resolution):
    return [[float(origin_xy[0] + point[0] * resolution),
             float(origin_xy[1] + point[1] * resolution)] for point in polygon]


def _boundary_polygon(scene_mask, origin_xy, resolution):
    contours = find_contours(np.asarray(scene_mask, dtype=float), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    return _metric_polygon(
        [[float(column), float(row)] for row, column in contour[::max(1, len(contour) // 200)]],
        origin_xy, resolution,
    )


def _build_rectangles(items, key, *, result, origin_xy, resolution):
    shape = tuple(result["labels"].shape)
    angle = float(result["row_angle_rad"])
    rectangles = []
    for index, item in enumerate(items, 1):
        r0, r1 = item["row_start"], item["row_stop"]
        c0, c1 = item["column_start"], item["column_stop"]
        row_polygon = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
        polygon = [_row_to_original(row, column, shape, angle) for row, column in row_polygon]
        prefix = key
        rectangles.append({
            f"{prefix}_id": index,
            "label": f"{prefix[:1].upper()}{index:02d}",
            "polygon_xy": polygon,
            "metric_polygon_xy": _metric_polygon(polygon, origin_xy, resolution),
            "width_m": float(item["width_m"]),
            "length_m": float(item["length_m"]),
            "between_rows": item.get("between_rows"),
        })
    return rectangles


def build_aisle_rectangles(result, *, origin_xy=(0.0, 0.0), resolution=None):
    """Build editable source-grid and metric rectangles for aisles."""
    resolution = float(resolution or result["config"]["resolution"])
    return _build_rectangles(result["aisles"], "aisle", result=result, origin_xy=origin_xy, resolution=resolution)


def build_ridge_rectangles(result, *, origin_xy=(0.0, 0.0), resolution=None):
    """Build editable source-grid and metric rectangles for ridges."""
    resolution = float(resolution or result["config"]["resolution"])
    rows = result["rows"]
    # The first and last row-like bands are exported as fixed greenhouse walls.
    # Only internal bands belong in the editable ridge sequence.
    # With a single detected band there is no way to infer an end-wall pair;
    # retain the historical single-band behavior for small synthetic/partial
    # maps.  On a real greenhouse (two or more bands), the outer bands are
    # reserved for walls and only strictly internal bands are ridges.
    internal_rows = rows[1:-1] if len(rows) > 2 else (rows if len(rows) == 1 else [])
    return _build_rectangles(internal_rows, "ridge", result=result, origin_xy=origin_xy, resolution=resolution)


def build_wall_rectangles(result, *, origin_xy=(0.0, 0.0), resolution=None):
    """Build fixed rectangles for the first and last row-like wall bands."""
    resolution = float(resolution or result["config"]["resolution"])
    rows = result["rows"]
    selected = rows[:1] + (rows[-1:] if len(rows) > 1 else [])
    return _build_rectangles(selected, "wall", result=result, origin_xy=origin_xy, resolution=resolution)


def write_pgm(labels, path, scene_mask=None):
    """Write a semantic occupancy PGM: aisle=254, unknown=205, outside=127."""
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    image = np.full(labels.shape, 205, dtype=np.uint8)
    image[labels == 1] = 254
    image[np.isin(labels, [2, 3, 4, 5, 6])] = 0
    if scene_mask is not None:
        scene_mask = np.asarray(scene_mask, dtype=bool)
        if scene_mask.shape != labels.shape:
            raise ValueError("scene_mask must match labels shape")
        image[~scene_mask] = 127
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        handle.write(np.flipud(image).tobytes())


def write_semantic_pgm(labels, path):
    """Write raw uint8 semantic IDs without occupancy remapping."""
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{labels.shape[1]} {labels.shape[0]}\n255\n".encode("ascii"))
        handle.write(np.flipud(labels).tobytes())


def export_navigation_assets(result, output, *, origin_xy=(0.0, 0.0), resolution=None):
    """Write PGM, semantic arrays, editable aisle rectangles, and metadata."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resolution = float(resolution or result["config"]["resolution"])
    origin_xy = [float(origin_xy[0]), float(origin_xy[1])]
    labels = np.asarray(result["labels"], dtype=np.uint8)
    rectangles = build_aisle_rectangles(result, origin_xy=origin_xy, resolution=resolution)
    ridge_rectangles = build_ridge_rectangles(result, origin_xy=origin_xy, resolution=resolution)
    wall_rectangles = build_wall_rectangles(result, origin_xy=origin_xy, resolution=resolution)
    np.save(output / "semantic_labels.npy", labels, allow_pickle=False)
    np.save(output / "aisle_mask.npy", labels == 1, allow_pickle=False)
    np.save(output / "wall_mask.npy", labels == 4, allow_pickle=False)
    np.save(output / "scene_mask.npy", result["scene_mask"], allow_pickle=False)
    write_pgm(labels, output / "navigation_map.pgm", result["scene_mask"])
    write_semantic_pgm(labels, output / "semantic_map.pgm")
    payload = {
        "schema_version": 1,
        "grid_shape_yx": list(labels.shape),
        "resolution_m": resolution,
        "origin_xy": origin_xy,
        "row_angle_rad": float(result["row_angle_rad"]),
        "aisle_count": len(rectangles),
        "ridge_count": len(ridge_rectangles),
        "wall_count": len(wall_rectangles),
        "semantic_labels": {"0": "unknown", "1": "aisle", "2": "ridge", "3": "obstacle_candidate", "4": "wall"},
        "rectangles": rectangles,
        "ridge_rectangles": ridge_rectangles,
        "wall_rectangles": wall_rectangles,
        "boundary_polygon_metric_xy": _boundary_polygon(result["scene_mask"], origin_xy, resolution),
    }
    (output / "aisle_rectangles.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    yaml_payload = {
        "image": "navigation_map.pgm",
        "resolution": resolution,
        "origin": [origin_xy[0], origin_xy[1], 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.90,
        "semantic_labels": payload["semantic_labels"],
        "semantic_image": "semantic_map.pgm",
        "aisle_count": len(rectangles),
        "ridge_count": len(ridge_rectangles),
        "wall_count": len(wall_rectangles),
        "note": "PGM is semantic occupancy: aisle free, ridge/obstacle occupied, unknown unknown.",
    }
    (output / "navigation_map.yaml").write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    return payload
