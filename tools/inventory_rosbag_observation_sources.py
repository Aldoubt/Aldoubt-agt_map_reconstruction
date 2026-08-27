#!/usr/bin/env python3
"""Inspect rosbag2 metadata for LiDAR / pose / TF source candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read rosbag2 metadata.yaml and list observation-source candidates. "
            "This does not deserialize messages or automatically choose a source."
        )
    )
    parser.add_argument("bag", help="rosbag2 directory or metadata.yaml path")
    parser.add_argument("--output", help="optional JSON output path")
    return parser


def _metadata_path(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "metadata.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"rosbag2 metadata not found: {path}")
    return path


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.io.rosbag_observation_inventory import (
        inventory_rosbag2_metadata,
    )

    metadata_path = _metadata_path(args.bag)
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    result = inventory_rosbag2_metadata(payload)
    result["source_metadata"] = str(metadata_path)

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("output:", output)

    print("storage_identifier:", result["storage_identifier"])
    print("message_count:", result["message_count"])
    for topic in result["topics"]:
        roles = ",".join(topic["candidate_roles"]) if topic["candidate_roles"] else "-"
        print(
            f"topic: {topic['name']} type={topic['type']} count={topic['message_count']} "
            f"family={topic['message_family']} roles={roles}"
        )
    for role in ("lidar_returns", "pose_or_odometry", "transform", "imu"):
        print(f"{role}_candidates:", result["candidates"][role])
    print("automatic_source_selection: false")


if __name__ == "__main__":
    main()
