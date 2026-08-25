#!/usr/bin/env python3
"""Inspect existing EXP003 arrays without loading the source PCD."""

import argparse
from pathlib import Path

from agt_map_reconstruction.maps.diagnostics import (
    save_discrete_previews,
    summarize_run,
    write_summary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "diagnostics.json"
    summary = summarize_run(args.run_dir)
    write_summary(summary, output)
    save_discrete_previews(args.run_dir, output.parent)
    print(f"diagnostics: {output}")


if __name__ == "__main__":
    main()
