"""Run ground segmentation benchmark.

Example:
python tools/run_benchmark.py --pcd map.pcd
"""

import argparse
import numpy as np

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.algorithms.ground_segmentation import height_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', required=True)
    args = parser.parse_args()

    cloud = load_pcd(args.pcd)
    points = np.asarray(cloud.points)
    result = height_threshold(points)

    print('input points:', len(points))
    print('ground:', len(result.ground))
    print('non_ground:', len(result.non_ground))


if __name__ == '__main__':
    main()
