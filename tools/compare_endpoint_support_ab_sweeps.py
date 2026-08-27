#!/usr/bin/env python3
"""Compare frozen P1-D3 support-threshold A/B sweeps across sampling runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten multiple scan/ray support-threshold endpoint A/B sweep JSON files "
            "into one measured comparison table. No threshold or sampling run is selected."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run label and scan_endpoint_ab_sweep.json path; may be repeated.",
    )
    parser.add_argument("--output", required=True)
    return parser


def _parse_run(value):
    if "=" not in value:
        raise ValueError("--run must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label must be non-empty")
    path = Path(raw_path).expanduser().resolve()
    return label, path


def load_rows(run_specs):
    rows = []
    support_basis = None
    thresholds_by_run = {}
    sources = {}
    for spec in run_specs:
        label, path = _parse_run(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        basis = str(payload.get("support_basis", ""))
        if basis not in {"ray", "scan"}:
            raise ValueError(f"{path}: unsupported support_basis={basis!r}")
        if support_basis is None:
            support_basis = basis
        elif basis != support_basis:
            raise ValueError("all compared sweeps must use the same support_basis")

        seen = []
        for item in payload.get("thresholds", []):
            threshold = int(item["min_support"])
            seen.append(threshold)
            comparison = item["comparison"]
            if comparison.get("geometry_frozen") is not True:
                raise ValueError(f"{path}: comparison geometry is not frozen")
            for side_name in ("entry", "exit"):
                side = comparison["sides"][side_name]
                baseline = side["baseline"]
                candidate = side["candidate"]
                delta = side["delta"]
                rows.append(
                    {
                        "label": label,
                        "support_basis": basis,
                        "min_support": threshold,
                        "side": side_name,
                        "supported_cell_count": int(item["supported_cell_count"]),
                        "supported_unknown_cell_count": int(
                            item["overlay_summary"]["ray_supported_unknown_cell_count"]
                        ),
                        "baseline_coverage": baseline["cross_row_coverage_fraction"],
                        "candidate_coverage": candidate["cross_row_coverage_fraction"],
                        "coverage_gain": delta["cross_row_coverage_fraction"],
                        "baseline_endpoint_median_m": baseline["endpoint_distance_median_m"],
                        "candidate_endpoint_median_m": candidate["endpoint_distance_median_m"],
                        "endpoint_reduction_m": delta["endpoint_distance_reduction_m"],
                        "baseline_depth_m": baseline["max_outward_depth_m"],
                        "candidate_depth_m": candidate["max_outward_depth_m"],
                        "depth_gain_m": delta["max_outward_depth_gain_m"],
                    }
                )
        thresholds_by_run[label] = seen
        sources[label] = str(path)

    if not rows:
        raise ValueError("no threshold rows found")
    return {
        "schema_version": 1,
        "support_basis": support_basis,
        "runs": [spec.split("=", 1)[0].strip() for spec in run_specs],
        "thresholds_by_run": thresholds_by_run,
        "rows": rows,
        "sources": sources,
        "policy": {
            "geometry_frozen_required": True,
            "automatic_sampling_selection": False,
            "automatic_threshold_selection": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }


def main():
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = load_rows(args.run)

    json_path = output / "endpoint_support_ab_comparison.json"
    csv_path = output / "endpoint_support_ab_comparison.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    fields = list(result["rows"][0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])

    print("output_json:", json_path)
    print("output_csv:", csv_path)
    print("support_basis:", result["support_basis"])
    print("runs:", ", ".join(result["runs"]))
    print("automatic_sampling_selection: false")
    print("automatic_threshold_selection: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
