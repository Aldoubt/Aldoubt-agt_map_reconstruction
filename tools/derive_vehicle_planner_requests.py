#!/usr/bin/env python3
"""Derive P1-G1.3 vehicle-ready planner inputs from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agt_map_reconstruction.maps.vehicle_planner_request_derivation import (
    derive_vehicle_planner_requests,
    write_vehicle_planner_request_bundle,
)


def _load_payload(value):
    path = Path(value).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain an object: {path}")
    return path, payload


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build runtime-compatible vehicle planner requests from frozen "
            "G1.2b recovery, G1.2c feasibility, planner-pair, and gap artifacts."
        )
    )
    parser.add_argument("--lateral-recovery", required=True)
    parser.add_argument("--feasibility-audit", required=True)
    parser.add_argument("--planner-pairs", required=True)
    parser.add_argument("--gap-diagnostics", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    lateral_path, lateral = _load_payload(args.lateral_recovery)
    feasibility_path, feasibility = _load_payload(args.feasibility_audit)
    pairs_path, pairs = _load_payload(args.planner_pairs)
    gap_path, gap = _load_payload(args.gap_diagnostics)

    result = derive_vehicle_planner_requests(
        lateral,
        feasibility,
        pairs,
        gap,
    )
    paths = write_vehicle_planner_request_bundle(result, args.output)

    summary = result["summary"]
    print(f"method: {result['method']}")
    print(f"ready_pair_side_count: {summary['ready_pair_side_count']}")
    print(f"excluded_pair_side_count: {summary['excluded_pair_side_count']}")
    print(f"positive_pair_side_count: {summary['positive_pair_side_count']}")
    print(f"negative_pair_side_count: {summary['negative_pair_side_count']}")
    print(f"directional_request_count: {summary['directional_request_count']}")
    print(f"source_topology_radius_m: {result['source_topology_radius_m']}")
    print(f"radius_role: {result['radius_role']}")
    print(f"output: {Path(args.output).expanduser().resolve()}")
    print(f"lateral_recovery: {lateral_path}")
    print(f"feasibility_audit: {feasibility_path}")
    print(f"planner_pairs: {pairs_path}")
    print(f"gap_diagnostics: {gap_path}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
