"""Run ground-segmentation benchmark with optional map preprocessing.

Examples:
PYTHONPATH=src python tools/run_benchmark.py --pcd map.pcd

PYTHONPATH=src python tools/run_benchmark.py \
  --pcd map.pcd \
  --preprocess-config configs/preprocessing/outdoor_default.yaml
"""

import argparse

from agt_map_reconstruction.algorithms.ground_segmentation import height_threshold
from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.preprocessing import clean_points, load_cleaning_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True)
    parser.add_argument(
        "--preprocess-config",
        help="Optional YAML cleaning pipeline applied before segmentation.",
    )
    args = parser.parse_args()

    points = load_pcd(args.pcd)
    raw_count = len(points)

    if args.preprocess_config:
        cleaning = clean_points(points, load_cleaning_config(args.preprocess_config))
        points = cleaning.points
        print(
            "preprocessing:",
            f"{cleaning.metadata['input_points']} -> {cleaning.metadata['output_points']}",
            f"(removed {cleaning.metadata['removed_points']})",
        )

    result = height_threshold(points)

    print("raw input points:", raw_count)
    print("segmentation input points:", len(points))
    print("ground:", len(result.ground))
    print("non_ground:", len(result.non_ground))


if __name__ == "__main__":
    main()
