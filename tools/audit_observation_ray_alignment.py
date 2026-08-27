#!/usr/bin/env python3
"""Audit exported observation rays against the frozen navigation map gauge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a schema-v1 observation ray bundle with a navigation map YAML/PGM. "
            "This is diagnostic only and does not transform rays or modify semantics."
        )
    )
    parser.add_argument("--rays", required=True, help="observation_rays.npz")
    parser.add_argument("--map-yaml", required=True, help="navigation_base_map.yaml")
    parser.add_argument("--output", required=True, help="output JSON path")
    return parser


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.observation_alignment_audit import (
        audit_observation_ray_alignment,
        load_navigation_grid,
    )
    from agt_map_reconstruction.maps.observation_ray_bundle import (
        load_observation_ray_bundle,
    )

    bundle = load_observation_ray_bundle(args.rays, expected_frame_id=None)
    navigation = load_navigation_grid(args.map_yaml)
    result = audit_observation_ray_alignment(bundle, navigation)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    origins = result["origins"]
    endpoints = result["endpoints"]
    print("output:", output)
    print("ray_count:", result["ray_count"])
    print("ray_frame_id:", result["ray_frame_id"])
    print("map_resolution_m:", result["map"]["resolution_m"])
    print("map_shape_yx:", [result["map"]["height"], result["map"]["width"]])
    print("map_origin:", result["map"]["origin"])
    print("origin_in_bounds_fraction:", f"{origins['in_bounds_fraction']:.6f}")
    print("endpoint_in_bounds_fraction:", f"{endpoints['in_bounds_fraction']:.6f}")
    for name in ("free", "unknown", "occupied", "other"):
        print(
            f"endpoint_{name}_fraction_of_in_bounds:",
            f"{endpoints['classes'][name]['fraction_of_in_bounds']:.6f}",
        )
    print("origin_xyz_extent_m:", origins["xyz_extent_m"])
    print("endpoint_xyz_extent_m:", endpoints["xyz_extent_m"])
    print("map_world_aabb_xy_m:", result["map"]["world_aabb_xy_m"])
    print("ray_length_median_m:", f"{result['ray_length_m']['median']:.6f}")
    print("ray_length_p95_m:", f"{result['ray_length_m']['p95']:.6f}")
    print("ray_length_max_m:", f"{result['ray_length_m']['max']:.6f}")
    print("automatic_alignment_acceptance: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
