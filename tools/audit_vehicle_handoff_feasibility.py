#!/usr/bin/env python3
"""Audit frozen G1.2b vehicle handoff feasibility without map access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agt_map_reconstruction.maps.vehicle_handoff_feasibility_audit import (
    audit_vehicle_handoff_feasibility,
    write_vehicle_handoff_feasibility_bundle,
)


def _load_payload(path):
    path = Path(path).expanduser().resolve()
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
            "Read-only P1-G1.2c audit of vehicle handoff anchor and pair-side "
            "feasibility from frozen G1.2b recovery artifacts."
        )
    )
    parser.add_argument("--lateral-recovery", required=True)
    parser.add_argument("--aisles", required=True)
    parser.add_argument("--planner-pairs", required=True)
    parser.add_argument("--gap-diagnostics", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    lateral_path, lateral_recovery = _load_payload(args.lateral_recovery)
    aisles_path, aisles_payload = _load_payload(args.aisles)
    planner_pairs_path, planner_pairs = _load_payload(args.planner_pairs)
    gap_path, gap_diagnostics = _load_payload(args.gap_diagnostics)

    rectangles = aisles_payload.get("rectangles")
    if not isinstance(rectangles, list):
        raise ValueError("aisle artifact requires rectangles list")

    result = audit_vehicle_handoff_feasibility(
        lateral_recovery,
        rectangles,
        planner_pairs,
        gap_diagnostics,
    )
    result["sources"] = {
        "lateral_recovery": str(lateral_path),
        "aisles": str(aisles_path),
        "planner_pairs": str(planner_pairs_path),
        "gap_diagnostics": str(gap_path),
    }

    output = Path(args.output).expanduser().resolve()
    write_vehicle_handoff_feasibility_bundle(result, output)

    summary = result["summary"]
    print("output:", output)
    print("method:", result["method"])
    print("anchor_count:", summary["anchor_count"])
    print("vehicle_anchor_valid_count:", summary["vehicle_anchor_valid_count"])
    print("footprint_wider_than_aisle_count:", summary["footprint_wider_than_aisle_count"])
    print("no_map_valid_pose_in_aisle_band_count:", summary["no_map_valid_pose_in_aisle_band_count"])
    print("pair_side_count:", summary["pair_side_count"])
    print("positive_pair_side_count:", summary["positive_pair_side_count"])
    print("negative_pair_side_count:", summary["negative_pair_side_count"])
    print("pair_vehicle_ready_count:", summary["pair_vehicle_ready_count"])
    print("pair_vehicle_not_ready_count:", summary["pair_vehicle_not_ready_count"])

    for item in result["pair_sides"]:
        print(
            f"{item['case_id']}: expectation={item['expectation_class']} "
            f"ready={item['pair_vehicle_ready']} "
            f"first={item['first_anchor_class']} "
            f"second={item['second_anchor_class']} "
            f"max_inset={item['max_longitudinal_inset_m']} "
            f"max_lateral={item['max_abs_lateral_shift_m']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
