#!/usr/bin/env python3
"""Compatibility entrypoint for the old comparison command.

The canonical CLI is tools/run_benchmark.py. This wrapper intentionally reuses
that implementation so algorithm discovery, configs, manifests, and outputs do
not drift into a second benchmark stack.
"""

from run_benchmark import main


if __name__ == "__main__":
    print("note: run_compare.py is deprecated; using the plugin benchmark pipeline")
    raise SystemExit(main())
