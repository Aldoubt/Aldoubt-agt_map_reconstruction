#!/usr/bin/env python3
"""Compare P1-E2 streaming ray-evidence sampling experiments.

Each run directory is expected to contain:

- streaming_observation_evidence_manifest.json
- ray_support_sweep.json
- scan_support_sweep.json

The tool is descriptive only. It never selects a sampling configuration or an
acceptance threshold automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-duration P1-E2 streaming experiments while keeping ray-count "
            "and unique-scan support separate."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Repeat for each experiment directory, for example baseline=results/.../s10_p20",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    return parser


def _parse_run(value):
    if "=" not in value:
        raise ValueError(f"--run must use LABEL=DIR syntax: {value!r}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("--run label must be non-empty")
    path = Path(raw_path).expanduser().resolve()
    return label, path


def _load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_map(payload):
    return {int(item["min_support_rays"]): item for item in payload.get("thresholds", [])}


def collect_run(label, directory):
    manifest_path = directory / "streaming_observation_evidence_manifest.json"
    ray_sweep_path = directory / "ray_support_sweep.json"
    scan_sweep_path = directory / "scan_support_sweep.json"
    manifest = _load_json(manifest_path)
    ray_sweep = _load_json(ray_sweep_path)
    scan_sweep = _load_json(scan_sweep_path)

    streaming = manifest.get("streaming", {})
    summary = manifest.get("summary", {})
    ray_thresholds = _threshold_map(ray_sweep)
    scan_thresholds = _threshold_map(scan_sweep)
    thresholds = sorted(set(ray_thresholds) & set(scan_thresholds))
    if not thresholds:
        raise ValueError(f"run {label!r} has no common ray/scan sweep thresholds")

    rows = []
    for basis, threshold_payloads in (
        ("ray", ray_thresholds),
        ("scan", scan_thresholds),
    ):
        for threshold in thresholds:
            item = threshold_payloads[threshold]
            for side_name in ("entry", "exit"):
                side = item["sides"][side_name]
                rows.append(
                    {
                        "label": label,
                        "support_basis": basis,
                        "min_support": threshold,
                        "side": side_name,
                        "supported_unknown": int(side["supported_unknown_cell_count"]),
                        "roi_unknown_fraction": float(side["supported_unknown_fraction_of_roi_unknown"]),
                        "component_count": int(side["component_count"]),
                        "largest_component": int(side["largest_component_cell_count"]),
                        "raw_cross_span": float(side["raw_supported_cross_row_span_fraction"]),
                        "raw_depth_m": float(side["raw_supported_max_outward_depth_m"]),
                        "new_strict_safe": int(side["new_strict_safe_cell_count"]),
                    }
                )

    return {
        "label": label,
        "directory": str(directory),
        "scan_stride": int(streaming["scan_stride"]),
        "export_point_stride": int(streaming["export_point_stride"]),
        "selected_scans": int(summary["selected_scan_count"]),
        "sampled_points": int(summary["sampled_point_count"]),
        "pose_supported_rays": int(summary["pose_supported_ray_count"]),
        "pose_rejected_before": int(summary["pose_rejected_before_trajectory"]),
        "pose_rejected_after": int(summary["pose_rejected_after_trajectory"]),
        "pose_rejected_gap": int(summary["pose_rejected_large_gap"]),
        "ray_supported_cells": int(summary.get("ray_supported_cell_count", summary.get("supported_cell_count", 0))),
        "scan_supported_cells": int(summary.get("scan_supported_cell_count", 0)),
        "max_scan_support_count": int(summary.get("max_scan_support_count", 0)),
        "thresholds": thresholds,
        "rows": rows,
        "sources": {
            "manifest": str(manifest_path),
            "ray_support_sweep": str(ray_sweep_path),
            "scan_support_sweep": str(scan_sweep_path),
        },
    }


def _write_csv(path, runs):
    rows = []
    for run in runs:
        for row in run["rows"]:
            rows.append(
                {
                    "label": run["label"],
                    "scan_stride": run["scan_stride"],
                    "export_point_stride": run["export_point_stride"],
                    **row,
                }
            )
    fieldnames = [
        "label",
        "scan_stride",
        "export_point_stride",
        "support_basis",
        "min_support",
        "side",
        "supported_unknown",
        "roi_unknown_fraction",
        "component_count",
        "largest_component",
        "raw_cross_span",
        "raw_depth_m",
        "new_strict_safe",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_parser().parse_args()
    parsed = [_parse_run(value) for value in args.run]
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("--run labels must be unique")

    runs = [collect_run(label, directory) for label, directory in parsed]
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "runs": runs,
        "policy": {
            "ray_and_unique_scan_support_reported_separately": True,
            "automatic_sampling_selection": False,
            "automatic_threshold_selection": False,
            "semantic_promotion": False,
        },
    }
    json_path = output / "sampling_comparison.json"
    csv_path = output / "sampling_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, runs)

    print("output:", output)
    for run in runs:
        print(
            f"{run['label']}: S{run['scan_stride']}/P{run['export_point_stride']} "
            f"selected_scans={run['selected_scans']} sampled_points={run['sampled_points']} "
            f"pose_supported_rays={run['pose_supported_rays']} "
            f"max_scan_support={run['max_scan_support_count']}"
        )
        for basis in ("ray", "scan"):
            for threshold in (1, 2, 3, 5):
                selected = [
                    row for row in run["rows"]
                    if row["support_basis"] == basis and row["min_support"] == threshold
                ]
                if len(selected) != 2:
                    continue
                entry = next(row for row in selected if row["side"] == "entry")
                exit_ = next(row for row in selected if row["side"] == "exit")
                print(
                    f"  {basis}>={threshold}: "
                    f"entry_unknown={entry['supported_unknown']} "
                    f"entry_span={entry['raw_cross_span']:.3f} "
                    f"entry_strict={entry['new_strict_safe']} "
                    f"exit_unknown={exit_['supported_unknown']} "
                    f"exit_span={exit_['raw_cross_span']:.3f} "
                    f"exit_strict={exit_['new_strict_safe']}"
                )
    print("automatic_sampling_selection: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
