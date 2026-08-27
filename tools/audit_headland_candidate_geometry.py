#!/usr/bin/env python3
"""Audit whether wide row-band candidates geometrically resemble headlands."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Audit wide_open_area_candidate geometry relative to common row "
            "entry/exit endpoint lines without promoting any region to HEADLAND."
        )
    )
    parser.add_argument("--row-band-regions", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_candidate_geometry import (
        analyze_headland_candidate_geometry,
    )

    source = Path(args.row_band_regions).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    payload = json.loads(source.read_text(encoding="utf-8"))
    grid = payload.get("grid", {})
    shape = (int(grid["height"]), int(grid["width"]))
    result = analyze_headland_candidate_geometry(
        payload.get("regions", []),
        grid_shape=shape,
    )
    result["source_row_band_regions"] = str(source)

    (output / "headland_candidate_geometry.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    rows = result.get("candidates", [])
    fields = [
        "label",
        "source_band_label",
        "row_axis_alignment",
        "cross_row_overlap_fraction",
        "entry_outward_fraction",
        "exit_outward_fraction",
        "semantic_promotion",
    ]
    with (output / "headland_candidate_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key) for key in fields})

    print("output:", output)
    print("row_aisles:", result["row_aisle_count"])
    print("candidates:", result["candidate_count"])
    print("row_axis_direction:", result["row_axis_direction"])
    print("row_cross_span:", result["row_cross_span"])
    for item in rows:
        print(
            f"{item['label']}: "
            f"row_axis_alignment={item['row_axis_alignment']:.3f} "
            f"cross_row_overlap={item['cross_row_overlap_fraction']:.3f} "
            f"entry_outward={item['entry_outward_fraction']:.3f} "
            f"exit_outward={item['exit_outward_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
