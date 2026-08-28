#!/usr/bin/env python3
"""Compare finite headland depth evidence with the historical unbounded diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build a reference-only comparison between the finite headland depth "
            "profile and historical unbounded outward evidence. The spatial domains "
            "are explicitly non-equivalent and no improvement score is computed."
        )
    )
    parser.add_argument("--headland-depth-evidence", required=True)
    parser.add_argument("--unbounded-evidence", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    from agt_map_reconstruction.maps.headland_depth_reference_comparison import (
        compare_headland_depth_with_unbounded,
    )

    finite_path = Path(args.headland_depth_evidence).expanduser().resolve()
    unbounded_path = Path(args.unbounded_evidence).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    finite = json.loads(finite_path.read_text(encoding="utf-8"))
    unbounded = json.loads(unbounded_path.read_text(encoding="utf-8"))
    result = compare_headland_depth_with_unbounded(finite, unbounded)
    result["sources"] = {
        "headland_depth_evidence": str(finite_path),
        "unbounded_evidence": str(unbounded_path),
    }
    (output / "headland_depth_vs_unbounded_reference.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("output:", output)
    print("method:", result["method"])
    for side in ("entry", "exit"):
        historical = result[side]["historical_unbounded"]
        finite_agg = result[side]["finite_depth_aggregate"]
        print(
            f"{side}: historical_unbounded_unknown={historical['unknown_cell_count']} "
            f"historical_ground_ceiling="
            f"{historical['ground_reference_ceiling_fraction_of_unknown']}"
        )
        print(
            f"{side}: finite_depth_unknown={finite_agg['unknown_cell_count']} "
            f"finite_ground_ceiling="
            f"{finite_agg['ground_reference_ceiling_fraction_of_unknown']}"
        )
    print("spatial_domains_equivalent: false")
    print("historical_unbounded_metrics_used_for_acceptance: false")
    print("finite_depth_profile_is_primary: true")
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
