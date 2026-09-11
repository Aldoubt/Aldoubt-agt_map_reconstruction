#!/usr/bin/env python3
"""Run one or more registered segmentation plugins as a reproducible experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agt_map_reconstruction.algorithms import (
    list_algorithm_specs,
    list_algorithms,
    load_plugin_module,
)
from agt_map_reconstruction.benchmark import benchmark_algorithm, write_benchmark_record
from agt_map_reconstruction.experiment import (
    build_experiment_manifest,
    write_experiment_manifest,
)
from agt_map_reconstruction.io.pcd_loader import load_pcd
from agt_map_reconstruction.maps.grid_map import build_traversability_map
from agt_map_reconstruction.visualization.compare import save_segmentation
from agt_map_reconstruction.visualization.grid import save_grid_maps


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError("benchmark config root must be a mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plugin benchmark -> experiment manifest runner"
    )
    parser.add_argument("--pcd", help="Input PCD file")
    parser.add_argument(
        "--algorithm",
        action="append",
        dest="algorithms",
        help="Algorithm name; repeat to select multiple. Defaults to all registered plugins.",
    )
    parser.add_argument(
        "--plugin-module",
        action="append",
        default=[],
        help="Python module that registers extra algorithms via register_algorithm().",
    )
    parser.add_argument("--config", help="YAML benchmark configuration")
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--experiment-id")
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--list-algorithms", action="store_true")
    args = parser.parse_args()

    for module_name in args.plugin_module:
        load_plugin_module(module_name)

    if args.list_algorithms:
        for spec in list_algorithm_specs():
            print(
                f"{spec.name}\tversion={spec.version}\tfamily={spec.family}\t"
                f"{spec.description}"
            )
        return 0

    if not args.pcd:
        parser.error("--pcd is required unless --list-algorithms is used")

    config = _load_config(args.config)
    algorithm_configs = config.get("algorithms", {})
    if algorithm_configs and not isinstance(algorithm_configs, dict):
        raise ValueError("config.algorithms must be a mapping")

    algorithms = args.algorithms or list_algorithms()
    unknown_configs = sorted(set(algorithm_configs) - set(algorithms))
    if unknown_configs:
        print(
            "warning: config contains algorithms not selected: "
            + ", ".join(unknown_configs)
        )

    points = load_pcd(args.pcd)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    runs = []
    for name in algorithms:
        algo_config = algorithm_configs.get(name, {})
        if not isinstance(algo_config, dict):
            raise ValueError(f"config for {name!r} must be a mapping")

        record = benchmark_algorithm(points, name, algo_config)
        out = root / name
        benchmark_path = write_benchmark_record(record, out)

        artifacts = [str(benchmark_path.relative_to(root))]
        if not args.no_visualization:
            save_segmentation(record.result, out)
            artifacts.extend(
                [
                    str((out / "ground.png").relative_to(root)),
                    str((out / "non_ground.png").relative_to(root)),
                    str((out / "overlay.png").relative_to(root)),
                    str((out / "ground_height_grid.png").relative_to(root)),
                ]
            )
            if len(record.result.ground_points):
                maps = build_traversability_map(record.result.ground_points)
                save_grid_maps(maps, out)
                artifacts.extend(
                    [
                        str((out / "height_map.png").relative_to(root)),
                        str((out / "relative_height.png").relative_to(root)),
                        str((out / "traversability.png").relative_to(root)),
                    ]
                )

        runs.append(
            {
                "algorithm": record.algorithm,
                "plugin_version": record.plugin_version,
                "benchmark_record": str(benchmark_path.relative_to(root)),
                "metrics": record.metrics,
                "artifacts": artifacts,
            }
        )

        print(
            f"{name}: input={record.metrics['input_points']} "
            f"ground={record.metrics['ground_points']} "
            f"non_ground={record.metrics['non_ground_points']} "
            f"elapsed={record.metrics['elapsed_seconds']:.3f}s"
        )

    experiment_id = args.experiment_id or Path(args.pcd).stem
    manifest = build_experiment_manifest(
        experiment_id=experiment_id,
        source={
            "type": "pcd",
            "path": str(Path(args.pcd)),
            "point_count": int(len(points)),
        },
        runs=runs,
        notes=args.notes,
    )
    manifest_path = write_experiment_manifest(manifest, root)
    print(f"experiment manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
