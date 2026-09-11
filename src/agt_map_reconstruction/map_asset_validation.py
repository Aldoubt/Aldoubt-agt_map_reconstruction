"""Validation contract for exported navigation-map bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


MAP_ASSET_SCHEMA = "agt.map_reconstruction.map_asset_validation/v1"
REQUIRED_FILES = (
    "navigation_base_map.pgm",
    "navigation_base_map.yaml",
    "candidate_mask.npy",
    "static_obstacle_mask.npy",
    "validation.json",
)
CANONICAL_VALUES = {0, 205, 254}


def validate_navigation_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    errors: list[str] = []
    warnings: list[str] = []

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        return {
            "schema": MAP_ASSET_SCHEMA,
            "stage": "map_asset_validation",
            "valid": False,
            "errors": [f"missing required asset: {name}" for name in missing],
            "warnings": warnings,
            "metrics": {},
        }

    image = cv2.imread(str(root / "navigation_base_map.pgm"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        errors.append("navigation_base_map.pgm could not be decoded")
        return {
            "schema": MAP_ASSET_SCHEMA,
            "stage": "map_asset_validation",
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "metrics": {},
        }

    observed = {int(value) for value in np.unique(image)}
    unexpected = sorted(observed - CANONICAL_VALUES)
    if unexpected:
        errors.append(f"unexpected PGM gray values: {unexpected}")

    with (root / "navigation_base_map.yaml").open("r", encoding="utf-8") as stream:
        map_yaml = yaml.safe_load(stream) or {}

    if map_yaml.get("mode") != "trinary":
        errors.append("map YAML mode must be 'trinary'")
    try:
        resolution = float(map_yaml.get("resolution", 0.0))
    except (TypeError, ValueError):
        resolution = 0.0
    if resolution <= 0.0:
        errors.append("map YAML resolution must be > 0")

    try:
        free_thresh = float(map_yaml.get("free_thresh"))
        occupied_thresh = float(map_yaml.get("occupied_thresh"))
        if not (0.0 <= free_thresh < occupied_thresh <= 1.0):
            errors.append("map YAML requires 0 <= free_thresh < occupied_thresh <= 1")
    except (TypeError, ValueError):
        errors.append("map YAML thresholds must be numeric")

    image_ref = map_yaml.get("image")
    if not isinstance(image_ref, str) or not image_ref:
        errors.append("map YAML image field is missing")
    elif not (root / image_ref).is_file():
        errors.append(f"map YAML image does not exist: {image_ref}")

    candidate = np.load(root / "candidate_mask.npy", allow_pickle=False)
    static = np.load(root / "static_obstacle_mask.npy", allow_pickle=False)
    expected_shape = tuple(image.shape)
    if tuple(candidate.shape) != expected_shape:
        errors.append(
            f"candidate_mask shape {candidate.shape} != map shape {expected_shape}"
        )
    if tuple(static.shape) != expected_shape:
        errors.append(
            f"static_obstacle_mask shape {static.shape} != map shape {expected_shape}"
        )

    static_as_free = None
    if tuple(static.shape) == expected_shape:
        # PGM is vertically flipped during export; flip it back to the
        # repository's internal (y, x) convention before mask comparison.
        internal_map = np.flipud(image)
        static_mask = static.astype(bool)
        static_as_free = int(np.count_nonzero(static_mask & (internal_map == 254)))
        if static_as_free:
            errors.append(
                f"{static_as_free} static obstacle cells are exported as free"
            )

    with (root / "validation.json").open("r", encoding="utf-8") as stream:
        original_validation = json.load(stream)
    if original_validation.get("static_obstacle_semantics_valid") is False:
        errors.append("validation.json reports invalid static obstacle semantics")
    if original_validation.get("map_server_yaml_valid") is False:
        errors.append("validation.json reports invalid map-server YAML")

    if not original_validation:
        warnings.append("validation.json is empty")

    return {
        "schema": MAP_ASSET_SCHEMA,
        "stage": "map_asset_validation",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": {
            "shape": list(image.shape),
            "resolution_m": resolution,
            "observed_gray_values": sorted(observed),
            "candidate_cells": int(np.count_nonzero(candidate)),
            "static_obstacle_cells": int(np.count_nonzero(static)),
            "static_obstacle_as_free_cells": static_as_free,
        },
    }


def write_map_asset_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
