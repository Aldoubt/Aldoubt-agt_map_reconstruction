#!/usr/bin/env python3
"""Run agricultural LiDAR segmentation and traversability comparison."""

import argparse
from pathlib import Path

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.visualization.compare import save_segmentation
from agt_map_reconstruction.algorithms.height_threshold import segment as height_segment
from agt_map_reconstruction.algorithms.morphological_pmf import segment as pmf_segment
from agt_map_reconstruction.maps.grid_map import build_traversability_map
from agt_map_reconstruction.visualization.grid import save_grid_maps


ALGORITHMS = {
    "height_threshold": height_segment,
    "morphological_pmf": pmf_segment,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    points = load_pcd(args.pcd)
    root = Path(args.output)

    for name, algo in ALGORITHMS.items():
        result = algo(points)
        out = root / name
        save_segmentation(result, out)

        maps = build_traversability_map(result["ground"])
        save_grid_maps(maps, out)

        print(name, len(result["ground"]), len(result["non_ground"]))


if __name__ == "__main__":
    main()
