#!/usr/bin/env python3
"""Run local agricultural traversability test."""

import argparse
from pathlib import Path

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.maps.grid_map import build_traversability_map
from agt_map_reconstruction.maps.corridor import extract_corridor, skeletonize_corridor
from agt_map_reconstruction.visualization.grid import save_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', required=True)
    parser.add_argument('--output', default='results/corridor_test')
    args = parser.parse_args()

    points = load_pcd(args.pcd)

    maps = build_traversability_map(points)
    height = maps['height']
    relative_height = maps['relative_height']
    traversability = maps['traversability']

    corridor = extract_corridor(traversability)
    centerline = skeletonize_corridor(corridor)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    save_grid(height, out / 'height.png', 'height')
    save_grid(relative_height, out / 'relative_height.png', 'relative_height')
    save_grid(traversability, out / 'traversability.png', 'traversability')
    save_grid(corridor, out / 'corridor.png', 'corridor')
    save_grid(centerline, out / 'centerline.png', 'centerline')

    print('points:', len(points))
    print('free:', int(traversability.sum()))
    print('corridor:', int(corridor.sum()))


if __name__ == '__main__':
    main()
