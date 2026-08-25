#!/usr/bin/env python3
"""Export ordered ridge dimensions and adjacent spacing as CSV and JSON."""

import argparse
import json
from pathlib import Path

from agt_map_reconstruction.maps.semantic_metrics import write_ridge_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads((args.map_dir / "aisle_rectangles.json").read_text(encoding="utf-8"))
    metrics = write_ridge_metrics(payload, args.output or args.map_dir)
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
