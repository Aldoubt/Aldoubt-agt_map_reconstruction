#!/usr/bin/env python3
"""Extract crop rows, walls, aisles, and MK-mini width checks from a height grid."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agt_map_reconstruction.maps.row_structure import RowStructureConfig, analyze_row_structure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("height_grid", type=Path)
    parser.add_argument("--obstacle-grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-angle-deg", type=float, default=0.0)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--vehicle-width-m", type=float, default=0.60)
    parser.add_argument("--ridge-height-threshold-m", type=float, default=0.05)
    parser.add_argument("--min-row-width-m", type=float, default=0.10)
    parser.add_argument("--max-row-width-m", type=float, default=1.00)
    args = parser.parse_args(argv)
    height = np.load(args.height_grid)
    obstacle = np.load(args.obstacle_grid) if args.obstacle_grid else None
    if obstacle is not None and not obstacle.dtype == bool:
        obstacle = obstacle == 2
    angle = np.radians(args.row_angle_deg)
    result = analyze_row_structure(
        height,
        row_direction=(np.cos(angle), np.sin(angle)),
        config=RowStructureConfig(
            resolution=args.resolution,
            vehicle_width_m=args.vehicle_width_m,
            ridge_height_threshold_m=args.ridge_height_threshold_m,
            min_row_width_m=args.min_row_width_m,
            max_row_width_m=args.max_row_width_m,
        ),
        origin_xy=(args.origin_x, args.origin_y),
        obstacle_grid=obstacle,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    (args.output / "structure.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    for name in ("ridge_mask", "aisle_mask", "wall_mask", "filled_height"):
        np.save(args.output / f"{name}.npy", result[name], allow_pickle=False)
    figure, axis = plt.subplots(figsize=(12, 10))
    finite = np.isfinite(result["filled_height"])
    image = np.ma.masked_where(~finite, result["filled_height"])
    axis.imshow(image, origin="lower", cmap="terrain")
    axis.contour(result["ridge_mask"], levels=[0.5], colors="red", linewidths=0.8)
    axis.contour(result["aisle_mask"], levels=[0.5], colors="cyan", linewidths=0.8)
    axis.contour(result["wall_mask"], levels=[0.5], colors="black", linewidths=1.0)
    axis.set_title("rows, aisles, and wall candidates")
    figure.savefig(args.output / "structure_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
