#!/usr/bin/env python3
"""Diagnose aisle geometry from an existing navigation validation.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose width limits, connectivity anomalies, and wide aisle outliers."
    )
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from agt_map_reconstruction.maps.aisle_geometry_diagnostics import (
        diagnose_aisle_geometry,
    )

    source = Path(args.validation).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    validation = json.loads(source.read_text(encoding="utf-8"))
    result = diagnose_aisle_geometry(validation)
    result["source_validation"] = str(source)

    (output / "aisle_geometry_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "aisle_id",
        "label",
        "width_m",
        "length_m",
        "theoretical_half_width_m",
        "conservative_width_radius_limit_m",
        "max_passing_radius_m",
        "minimum_clearance_mode",
        "first_unexpected_failed_radius_m",
        "wide_width_outlier",
    ]
    with (output / "aisle_geometry_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["aisles"]:
            writer.writerow({key: row.get(key) for key in fieldnames})

    summary = result["summary"]
    print("output:", output)
    print("minimum_width_limited:", summary["minimum_width_limited"])
    print(
        "minimum_connectivity_limited:",
        summary["minimum_connectivity_limited"],
    )
    print(
        "unexpected_connectivity_failures:",
        summary["unexpected_connectivity_failures"],
    )
    print("wide_width_outliers:", summary["wide_width_outliers"])


if __name__ == "__main__":
    main()
