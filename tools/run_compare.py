#!/usr/bin/env python3
"""Run first-stage agricultural LiDAR segmentation comparison."""

import argparse
from pathlib import Path

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.visualization.compare import save_segmentation
from agt_map_reconstruction.algorithms.height_threshold import segment as height_segment
from agt_map_reconstruction.algorithms.morphological_pmf import segment as pmf_segment


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
        save_segmentation(result, root / name)
        print(name, len(result["ground"]), len(result["non_ground"]))


if __name__ == "__main__":
    main()
