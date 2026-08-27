#!/usr/bin/env python3
"""Run EXP002 agricultural corridor recovery test."""

import argparse
import csv
from pathlib import Path

import yaml

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.maps.grid_map import build_traversability_map
from agt_map_reconstruction.maps.row_direction import estimate_row_direction
from agt_map_reconstruction.maps.corridor import extract_corridor, skeletonize_corridor
from agt_map_reconstruction.visualization.grid import save_grid


def save_centerline_csv(centerline, path):
    ys, xs = centerline.nonzero()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in zip(xs, ys):
            writer.writerow([int(x), int(y)])


def build_metadata_payload(maps, angle, direction, point_count, corridor):
    return {
        "experiment": "EXP002",
        "grid": maps["metadata"].to_dict(),
        "row_angle_rad": float(angle),
        "row_direction": [float(v) for v in direction],
        "points": int(point_count),
        "corridor_cells": int(corridor.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', required=True)
    parser.add_argument('--output', default='results/EXP002')
    parser.add_argument('--resolution', type=float, default=0.05)
    args = parser.parse_args()

    points = load_pcd(args.pcd)

    maps = build_traversability_map(points, resolution=args.resolution)
    height = maps['height']
    relative_height = maps['relative_height']
    traversability = maps['traversability']

    angle, direction = estimate_row_direction(relative_height)

    corridor = extract_corridor(
        traversability,
        row_direction=direction,
    )
    centerline = skeletonize_corridor(corridor)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    save_grid(height, out / 'height.png', 'height')
    save_grid(relative_height, out / 'relative_height.png', 'relative_height')
    save_grid(traversability, out / 'traversability.png', 'traversability')
    save_grid(corridor, out / 'corridor.png', 'corridor')
    save_grid(centerline, out / 'centerline.png', 'centerline')
    save_centerline_csv(centerline, out / 'centerline.csv')

    metadata_payload = build_metadata_payload(
        maps=maps,
        angle=angle,
        direction=direction,
        point_count=len(points),
        corridor=corridor,
    )
    with open(out / 'metadata.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(metadata_payload, f, sort_keys=False)

    print('points:', len(points))
    print('row_angle:', angle)
    print('direction:', direction)
    print('corridor:', int(corridor.sum()))


if __name__ == '__main__':
    main()
