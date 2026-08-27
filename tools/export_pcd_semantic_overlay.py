#!/usr/bin/env python3
"""Export a colored PCD + semantic aisle/ridge overlay for visual review.

The PLY is intended for CloudCompare/Open3D.  A NumPy voxel cache can be used
to avoid rereading the 85M-point source cloud on every review.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
import open3d as o3d


def _load_points(args):
    if args.pcd_cache:
        return np.load(args.pcd_cache, allow_pickle=False).astype(np.float32)
    from agt_map_reconstruction.io.pcd_loader import load_pcd
    return load_pcd(args.pcd)


def _z_colors(points):
    z = points[:, 2]
    low, high = np.percentile(z, [2, 98])
    t = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
    # Muted CloudCompare-like blue -> cyan -> green -> orange -> red.
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
    z = points[:, 2]
    low, high = np.percentile(z, [2, 98])
    t = np.clip((z - low) / max(high - low, 1e-6), 0.0, 1.0)
    value = 0.34 + 0.30 * t
    return np.column_stack((value * 0.88, value * 0.93, value))


def _ground_z(points, polygon, percentile):
    polygon = np.asarray(polygon, dtype=float)
    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    window = points[
        (points[:, 0] >= lower[0]) & (points[:, 0] <= upper[0])
        & (points[:, 1] >= lower[1]) & (points[:, 1] <= upper[1])
    ]
    if len(window) == 0:
        return float(np.percentile(points[:, 2], percentile))
    inside = PolygonPath(polygon).contains_points(window[:, :2])
    selected = window[inside, 2]
    return float(np.percentile(selected if len(selected) else window[:, 2], percentile))


def _line_set(items, key, color, points, percentile, z_offset, overlay_z=None):
    vertices, lines, colors = [], [], []
    for item in items:
        polygon = np.asarray(item["metric_polygon_xy"], dtype=float)
        z = (overlay_z if overlay_z is not None else _ground_z(points, polygon, percentile)) + z_offset
        base = len(vertices)
        vertices.extend([[x, y, z] for x, y in polygon])
        for index in range(len(polygon)):
            lines.append([base + index, base + (index + 1) % len(polygon)])
            colors.append(color)
    if not vertices:
        return None
    geometry = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    geometry.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return geometry


def _route_line_set(route_items, color, origin, resolution, z):
    vertices, lines, colors = [], [], []
    for item in route_items:
        route = item.get("route_points", [])
        if len(route) < 2:
            continue
        base = len(vertices)
        vertices.extend([
            [origin[0] + point["x_cell"] * resolution,
             origin[1] + point["y_cell"] * resolution, z]
            for point in route
        ])
        for index in range(len(route) - 1):
            lines.append([base + index, base + index + 1])
            colors.append(color)
    if not vertices:
        return None
    geometry = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(vertices)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    geometry.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return geometry


def _write_top_view(points, payload, failed, output, percentile, route_payload=None):
    figure, axis = plt.subplots(figsize=(14, 10))
    sample = points[::max(1, len(points) // 250_000)]
    axis.scatter(sample[:, 0], sample[:, 1], color=(0.48, 0.52, 0.58),
                 s=0.15, rasterized=True)
    for item in payload.get("rectangles", []):
        polygon = np.asarray(item["metric_polygon_xy"] + [item["metric_polygon_xy"][0]])
        label = item.get("label", f"A{item.get('aisle_id', 0):02d}")
        color = "red" if label in failed else "lime"
        axis.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.5)
        center = polygon[:-1].mean(axis=0)
        axis.text(center[0], center[1], label, color=color, fontsize=7)
    if route_payload:
        origin = np.asarray(payload.get("origin_xy", [0.0, 0.0]), dtype=float)
        resolution = float(payload.get("resolution_m", 0.05))
        for item in route_payload.get("aisles", []):
            route = item.get("route_points", [])
            if len(route) < 2:
                continue
            xy = np.asarray([[origin[0] + point["x_cell"] * resolution,
                              origin[1] + point["y_cell"] * resolution]
                             for point in route])
            axis.plot(xy[:, 0], xy[:, 1], color="lime", linewidth=1.0)
    for key, color in (("ridge_rectangles", "orange"), ("wall_rectangles", "red")):
        for item in payload.get(key, []):
            polygon = np.asarray(item["metric_polygon_xy"] + [item["metric_polygon_xy"][0]])
            axis.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.0)
            center = polygon[:-1].mean(axis=0)
            axis.text(center[0], center[1], item.get("label", ""), color=color, fontsize=6)
    axis.set_aspect("equal")
    axis.set_title("PCD + semantic aisle/ridge overlay (red aisles failed EXP004-B2)")
    axis.set_xlabel("map x (m)")
    axis.set_ylabel("map y (m)")
    origin = np.asarray(payload.get("origin_xy", [0.0, 0.0]), dtype=float)
    axis.scatter([origin[0]], [origin[1]], color="black", s=24, zorder=8)
    axis.quiver(origin[0], origin[1], 1.0, 0.0, color="red", angles="xy",
                scale_units="xy", scale=1, width=0.003, zorder=8)
    axis.quiver(origin[0], origin[1], 0.0, 1.0, color="green", angles="xy",
                scale_units="xy", scale=1, width=0.003, zorder=8)
    axis.text(origin[0], origin[1], " O(map)  X→  Y↑", fontsize=8,
              color="black", zorder=9)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcd", type=Path)
    source.add_argument("--pcd-cache", type=Path)
    parser.add_argument("--rectangles", type=Path, required=True)
    parser.add_argument("--route-csv", type=Path)
    parser.add_argument("--route-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-percentile", type=float, default=10.0)
    parser.add_argument("--z-offset", type=float, default=0.03)
    parser.add_argument("--point-palette", choices=("neutral", "height"), default="neutral")
    parser.add_argument("--overlay-mode", choices=("fixed", "terrain"), default="fixed")
    parser.add_argument("--world-origin", type=float, nargs=2, metavar=("X", "Y"),
                        help="world origin for the raster; shifts metric polygons")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    points = _load_points(args)
    payload = json.loads(args.rectangles.read_text(encoding="utf-8"))
    route_payload = json.loads(args.route_json.read_text(encoding="utf-8")) if args.route_json else None
    if args.world_origin is not None:
        old = np.asarray(payload.get("origin_xy", [0.0, 0.0]), dtype=float)
        shift = np.asarray(args.world_origin, dtype=float) - old
        for key in ("rectangles", "ridge_rectangles", "wall_rectangles"):
            for item in payload.get(key, []):
                item["metric_polygon_xy"] = (
                    np.asarray(item["metric_polygon_xy"], dtype=float) + shift
                ).tolist()
        boundary = payload.get("boundary_polygon_metric_xy", [])
        if boundary:
            payload["boundary_polygon_metric_xy"] = (
                np.asarray(boundary, dtype=float) + shift
            ).tolist()
        payload["origin_xy"] = [float(args.world_origin[0]), float(args.world_origin[1])]
    failed = set()
    if args.route_csv:
        with args.route_csv.open(newline="", encoding="utf-8") as stream:
            failed = {row["label"] for row in csv.DictReader(stream) if row["passed"].lower() != "true"}

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    colors = _neutral_colors(points) if args.point_palette == "neutral" else _z_colors(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    overlay_z = float(np.percentile(points[:, 2], args.ground_percentile)) if args.overlay_mode == "fixed" else None
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
        route_z = (overlay_z if overlay_z is not None else
                   float(np.percentile(points[:, 2], args.ground_percentile))) + args.z_offset + 0.02
        routes = [item for item in route_payload.get("aisles", []) if item.get("passed")]
        geometry = _route_line_set(routes, [0.1, 1.0, 0.1], origin, resolution, route_z)
        if geometry is not None:
            geometries.append(geometry)
    for key, color in (("rectangles", [0.1, 0.95, 0.2]),
                       ("ridge_rectangles", [1.0, 0.60, 0.05]),
                       ("wall_rectangles", [1.0, 0.05, 0.05])):
        geometry = _line_set(payload.get(key, []), key, color, points,
                             args.ground_percentile, args.z_offset, overlay_z)
        if geometry is not None:
            geometries.append(geometry)
    o3d.io.write_point_cloud(str(args.output / "pcd_semantic_overlay.ply"), cloud)
    for index, geometry in enumerate(geometries[1:], 1):
        path = args.output / f"semantic_overlay_{index}.ply"
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(path), geometry)
        else:
            o3d.io.write_line_set(str(path), geometry)
    _write_top_view(points, payload, failed, args.output / "pcd_semantic_overlay_top.png",
                    args.ground_percentile, route_payload)
    metadata = {
        "point_count": int(len(points)),
        "source": str(args.pcd_cache or args.pcd),
        "aisle_count": len(payload.get("rectangles", [])),
        "ridge_count": len(payload.get("ridge_rectangles", [])),
        "wall_count": len(payload.get("wall_rectangles", [])),
        "failed_aisles": sorted(failed),
        "ground_percentile": args.ground_percentile,
        "z_offset_m": args.z_offset,
        "point_palette": args.point_palette,
        "overlay_mode": args.overlay_mode,
        "overlay_z_m": overlay_z,
    }
    (args.output / "pcd_semantic_overlay.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "aisle_rectangles_world.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
