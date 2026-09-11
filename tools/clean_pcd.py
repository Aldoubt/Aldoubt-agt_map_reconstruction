"""Clean an offline PCD before map reconstruction.

Example:
PYTHONPATH=src python tools/clean_pcd.py \
  --pcd global_map.pcd \
  --config configs/preprocessing/outdoor_default.yaml \
  --output results/preprocessing/global_map_clean.pcd \
  --removed-output results/preprocessing/global_map_removed.pcd \
  --report results/preprocessing/cleaning_report.json
"""

import argparse
import json
from pathlib import Path

from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.io.pcd_writer import write_pcd
from agt_map_reconstruction.preprocessing import clean_points, load_cleaning_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--removed-output")
    parser.add_argument("--report")
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()

    config = load_cleaning_config(args.config)
    result = clean_points(load_pcd(args.pcd), config)

    write_pcd(args.output, result.points, compressed=args.compressed)
    if args.removed_output:
        write_pcd(args.removed_output, result.removed_points, compressed=args.compressed)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"input points: {result.metadata['input_points']}")
    print(f"output points: {result.metadata['output_points']}")
    print(f"removed points: {result.metadata['removed_points']}")
    for stage in result.metadata["stages"]:
        print(
            f"{stage['stage']}: "
            f"{stage['input_points']} -> {stage['output_points']} "
            f"(removed {stage['removed_points']})"
        )


if __name__ == "__main__":
    main()
