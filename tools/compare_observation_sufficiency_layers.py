#!/usr/bin/env python3
"""Compare two observation-sufficiency summaries without selecting a winner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", required=True)
    return parser


def _rows(payload, label):
    rows = []
    scopes = {"full_map": payload["full_map"], **payload.get("endpoint_rois", {})}
    for scope_name, scope in scopes.items():
        for class_name, metrics in scope["classes"].items():
            rows.append(
                {
                    "label": label,
                    "scope": scope_name,
                    "class_name": class_name,
                    "count": int(metrics["count"]),
                    "fraction_of_roi": float(metrics["fraction_of_roi"]),
                    "fraction_of_unknown": metrics.get("fraction_of_unknown"),
                }
            )
    return rows


def main():
    args = build_parser().parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline.get("min_repeated_scans") != candidate.get("min_repeated_scans"):
        raise ValueError("min_repeated_scans mismatch")

    base_rows = _rows(baseline, args.baseline_label)
    cand_rows = _rows(candidate, args.candidate_label)
    base_index = {(r["scope"], r["class_name"]): r for r in base_rows}
    cand_index = {(r["scope"], r["class_name"]): r for r in cand_rows}
    if set(base_index) != set(cand_index):
        raise ValueError("scope/class mismatch between sufficiency summaries")

    deltas = []
    for key in sorted(base_index):
        b = base_index[key]
        c = cand_index[key]
        deltas.append(
            {
                "scope": key[0],
                "class_name": key[1],
                "baseline_count": b["count"],
                "candidate_count": c["count"],
                "count_delta": c["count"] - b["count"],
                "baseline_fraction_of_roi": b["fraction_of_roi"],
                "candidate_fraction_of_roi": c["fraction_of_roi"],
                "fraction_of_roi_delta": c["fraction_of_roi"] - b["fraction_of_roi"],
                "baseline_fraction_of_unknown": b["fraction_of_unknown"],
                "candidate_fraction_of_unknown": c["fraction_of_unknown"],
                "fraction_of_unknown_delta": (
                    None
                    if b["fraction_of_unknown"] is None or c["fraction_of_unknown"] is None
                    else c["fraction_of_unknown"] - b["fraction_of_unknown"]
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "min_repeated_scans": baseline["min_repeated_scans"],
        "rows": deltas,
        "sources": {"baseline": str(baseline_path), "candidate": str(candidate_path)},
        "policy": {
            "automatic_acceptance": False,
            "automatic_sampling_selection": False,
            "navigation_map_modified": False,
            "semantic_promotion": False,
        },
    }
    (output / "observation_sufficiency_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    with (output / "observation_sufficiency_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = list(deltas[0].keys()) if deltas else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deltas)

    print("output:", output)
    for row in deltas:
        if row["scope"] in ("entry", "exit") and row["class_name"].startswith("unknown_"):
            print(
                f"{row['scope']} {row['class_name']}: "
                f"{row['baseline_count']} -> {row['candidate_count']} "
                f"delta={row['count_delta']:+d}"
            )
    print("automatic_acceptance: false")
    print("navigation_map_modified: false")
    print("semantic_promotion: false")


if __name__ == "__main__":
    main()
