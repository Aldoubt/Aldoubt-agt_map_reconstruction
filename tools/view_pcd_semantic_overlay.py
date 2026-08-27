#!/usr/bin/env python3
"""Lightweight interactive PCD/semantic-map reviewer."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2
from matplotlib.path import Path as PolygonPath


def _ground_z(points, polygon, percentile=10.0):
    polygon = np.asarray(polygon, dtype=float)
    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    window = points[
        (points[:, 0] >= lower[0]) & (points[:, 0] <= upper[0])
        & (points[:, 1] >= lower[1]) & (points[:, 1] <= upper[1])
    ]
    if len(window) == 0:
        return float(np.percentile(points[:, 2], percentile))
    inside = PolygonPath(polygon).contains_points(window[:, :2])
    values = window[inside, 2]
    return float(np.percentile(values if len(values) else window[:, 2], percentile))


def _z_colors(points):
    z = points[:, 2]
    low, high = np.percentile(z, [2, 98])
    t = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
    stops = np.array([0.0, 0.22, 0.48, 0.74, 1.0])
    anchors = np.array([
        [0.035, 0.075, 0.350], [0.000, 0.420, 0.680],
        [0.070, 0.610, 0.390], [0.820, 0.470, 0.090],
        [0.650, 0.035, 0.045],
    ])
    return np.column_stack([
        np.interp(t, stops, anchors[:, channel]) for channel in range(3)
    ])


def _neutral_colors(points):
    """Neutral blue-gray point colors keep vegetation from reading as semantic green."""
    z = points[:, 2]
    low, high = np.percentile(z, [2, 98])
    t = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
    value = 0.34 + 0.30 * t
    return np.column_stack((value * 0.88, value * 0.93, value))


def _lines(items, color, points, z_offset, overlay_z=None):
    vertices, lines, colors = [], [], []
    for item in items:
        polygon = np.asarray(item["metric_polygon_xy"], dtype=float)
        z = (overlay_z if overlay_z is not None else _ground_z(points, polygon)) + z_offset
        base = len(vertices)
        vertices.extend([[x, y, z] for x, y in polygon])
        for index in range(len(polygon)):
            lines.append([base + index, base + (index + 1) % len(polygon)])
            colors.append(color)
    if not vertices:
        return None
    result = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    result.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return result


def _route_lines(items, color, origin, resolution, z):
    vertices, lines, colors = [], [], []
    for item in items:
        route = item.get("route_points", [])
        if len(route) < 2:
            continue
        base = len(vertices)
        vertices.extend([[origin[0] + point["x_cell"] * resolution,
                          origin[1] + point["y_cell"] * resolution, z]
                         for point in route])
        for index in range(len(route) - 1):
            lines.append([base + index, base + index + 1])
            colors.append(color)
    if not vertices:
        return None
    result = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    result.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return result


def _text_lines(text, anchor, z, color, scale=0.012):
    """Make small camera-independent ASCII labels from OpenCV stroke pixels."""
    canvas = np.zeros((36, max(80, 13 * len(text)), 1), dtype=np.uint8)
    cv2.putText(canvas, text, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                255, 2, cv2.LINE_AA)
    vertices, lines, colors = [], [], []
    base_x, base_y = float(anchor[0]), float(anchor[1])
    for row, column in zip(*np.nonzero(canvas[:, :, 0] > 100)):
        x = base_x + column * scale
        y = base_y + (canvas.shape[0] - row) * scale
        index = len(vertices)
        vertices.extend([[x, y, z], [x + scale, y, z]])
        lines.append([index, index + 1])
        colors.append(color)
    if not vertices:
        return None
    result = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    result.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return result


def _cross(center, z, color, size=0.22):
    x, y = float(center[0]), float(center[1])
    vertices = [[x - size, y - size, z], [x + size, y + size, z],
                [x - size, y + size, z], [x + size, y - size, z]]
    result = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray([[0, 1], [2, 3]], dtype=np.int32)),
    )
    result.colors = o3d.utility.Vector3dVector(np.asarray([color, color]))
    return result


def _reason_code(item):
    if item.get("failure_reason") == "manual_review_blocked":
        return "BLOCKED"
    return {"entry": "ENTRY", "interior": "INTERIOR", "exit": "EXIT"}.get(
        item.get("failure_region"), "FAIL"
    )


def _failure_point(item, rectangle, resolution):
    polygon = np.asarray(rectangle["metric_polygon_xy"], dtype=float)
    start = 0.5 * (polygon[0] + polygon[3])
    end = 0.5 * (polygon[1] + polygon[2])
    match = re.search(r"station_(\d+)", str(item.get("failure_reason", "")))
    if match:
        station = int(match.group(1))
        count = max(1, int(item.get("control_point_count", 1)) - 1)
        fraction = float(np.clip(station / count, 0.0, 1.0))
    else:
        fraction = 0.5 if item.get("failure_region") == "manual_review" else 0.0
    return start + fraction * (end - start)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=Path, required=True, help="dense .npy cache")
    parser.add_argument("--rectangles", type=Path, required=True)
    parser.add_argument("--route-csv", type=Path)
    parser.add_argument("--route-json", type=Path)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--z-offset", type=float, default=0.05)
    parser.add_argument("--point-palette", choices=("neutral", "height"), default="neutral")
    parser.add_argument("--overlay-mode", choices=("fixed", "terrain"), default="fixed",
                        help="fixed puts all semantic frames on one reference plane")
    args = parser.parse_args(argv)

    points = np.load(args.cloud, allow_pickle=False).astype(np.float64)
    payload = json.loads(args.rectangles.read_text(encoding="utf-8"))
    route_payload = json.loads(args.route_json.read_text(encoding="utf-8")) if args.route_json else None
    failed = set()
    if args.route_csv:
        with args.route_csv.open(newline="", encoding="utf-8") as stream:
            failed = {row["label"] for row in csv.DictReader(stream)
                      if row["passed"].lower() != "true"}

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    colors = _neutral_colors(points) if args.point_palette == "neutral" else _z_colors(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    overlay_z = float(np.percentile(points[:, 2], 10.0)) if args.overlay_mode == "fixed" else None
    geometries = [cloud]
    map_origin = np.asarray(payload.get("origin_xy", [0.0, 0.0]), dtype=float)
    axis_size = max(0.5, float(np.ptp(points[:, :2], axis=0).min()) * 0.08)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[map_origin[0], map_origin[1],
                (overlay_z if overlay_z is not None else float(np.percentile(points[:, 2], 10.0))) + args.z_offset],
    ))
    if route_payload:
        origin = np.asarray(payload.get("origin_xy", [0.0, 0.0]), dtype=float)
        resolution = float(payload.get("resolution_m", 0.05))
        route_z = (overlay_z if overlay_z is not None else float(np.percentile(points[:, 2], 10.0))) + args.z_offset + 0.02
        by_label = {item.get("label"): item for item in payload.get("rectangles", [])}
        for item in route_payload.get("aisles", []):
            rectangle = by_label.get(item.get("label"))
            if rectangle is None:
                continue
            polygon = np.asarray(rectangle["metric_polygon_xy"], dtype=float)
            center = polygon.mean(axis=0)
            if item.get("passed"):
                text = _text_lines(item["label"], center, route_z + 0.04, [0.1, 1.0, 0.1])
            else:
                text = _text_lines(item["label"] + " " + _reason_code(item),
                                    center, route_z + 0.04, [1.0, 0.1, 0.1])
                geometries.append(_cross(_failure_point(item, rectangle, resolution),
                                         route_z + 0.06, [1.0, 0.0, 0.0]))
            if text is not None:
                geometries.append(text)
        routes = [item for item in route_payload.get("aisles", []) if item.get("passed")]
        geometry = _route_lines(routes, [0.1, 1.0, 0.1], origin, resolution, route_z)
        if geometry is not None:
            geometries.append(geometry)
    for key, good_color, bad_color in (
        ("rectangles", [0.1, 0.95, 0.2], [0.95, 0.05, 0.05]),
        ("ridge_rectangles", [1.0, 0.60, 0.05], [1.0, 0.60, 0.05]),
        ("wall_rectangles", [1.0, 0.05, 0.05], [1.0, 0.05, 0.05]),
    ):
        items = payload.get(key, [])
        if key == "rectangles":
            good = [item for item in items if item.get("label") not in failed]
            bad = [item for item in items if item.get("label") in failed]
            for subset, color in ((good, good_color), (bad, bad_color)):
                geometry = _lines(subset, color, points, args.z_offset, overlay_z)
                if geometry is not None:
                    geometries.append(geometry)
        else:
            geometry = _lines(items, good_color, points, args.z_offset, overlay_z)
            if geometry is not None:
                geometries.append(geometry)

    print(f"points: {len(points):,}")
    print("green=route pass, red=route fail, orange=ridge, dark red=wall")
    print("Open3D controls: left-drag rotate, middle-drag pan, wheel zoom")
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="PCD Semantic Navigation Review", width=1440, height=900)
    for geometry in geometries:
        viewer.add_geometry(geometry)
    options = viewer.get_render_option()
    options.point_size = float(args.point_size)
    options.background_color = np.array([0.04, 0.04, 0.04])
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    raise SystemExit(main())
