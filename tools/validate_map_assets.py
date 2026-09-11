#!/usr/bin/env python3
"""Validate a Navigation Map V2 asset bundle before runtime integration."""

from __future__ import annotations

import argparse
from pathlib import Path

from agt_map_reconstruction.map_asset_validation import (
    validate_navigation_bundle,
    write_map_asset_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate generated navigation-map assets"
    )
    parser.add_argument("--bundle", required=True, help="Navigation-map bundle directory")
    parser.add_argument(
        "--report",
        help="Output report path; defaults to <bundle>/asset_validation.json",
    )
    args = parser.parse_args()

    report = validate_navigation_bundle(args.bundle)
    report_path = Path(args.report) if args.report else Path(args.bundle) / "asset_validation.json"
    write_map_asset_report(report, report_path)

    print(f"valid: {report['valid']}")
    print(f"report: {report_path}")
    for message in report["errors"]:
        print(f"error: {message}")
    for message in report["warnings"]:
        print(f"warning: {message}")

    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
