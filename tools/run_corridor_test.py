#!/usr/bin/env python3
"""Run local agricultural traversability test."""

import argparse
from pathlib import Path
import numpy as np

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.maps.grid_map import points_to_height_grid, traversability_from_height
from agt_map_reconstruction.maps.corridor import extract_corridor, skeletonize_corridor
from agt_map_reconstruction.visualization.grid import save_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', required=True)
    parser.add_argument('--output', default='results/corridor_test')
    args = parser.parse_args()

    points = load_pcd(args.pcd)
    height = points_to_height_grid(points)
    traversability = traversability_from_height(height)
    corridor = extract_corridor(traversability)
    centerline = skeletonize_corridor(corridor)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_grid(height, out / 'height.png', 'height')
    save_grid(traversability, out / 'traversability.png', 'traversability')
    save_grid(corridor, out / 'corridor.png', 'corridor')
    save_grid(centerline, out / 'centerline.png', 'centerline')

    print('points:', len(points))
    print('free:', int(traversability.sum()))
    print('corridor:', int(corridor.sum()))


if __name__ == '__main__':
    main()
