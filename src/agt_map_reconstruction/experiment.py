"""Experiment manifest helpers.

An experiment manifest is the handoff between a plugin benchmark and later
map-reconstruction/validation stages. It records provenance without embedding
large generated assets in source code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_SCHEMA = "agt.map_reconstruction.experiment/v1"


def build_experiment_manifest(
    *,
    experiment_id: str,
    source: Mapping[str, Any],
    runs: Iterable[Mapping[str, Any]],
    notes: str = "",
) -> dict[str, Any]:
    normalized_id = experiment_id.strip()
    if not normalized_id:
        raise ValueError("experiment_id must not be empty")
    return {
        "schema": EXPERIMENT_SCHEMA,
        "stage": "experiment",
        "experiment_id": normalized_id,
        "source": dict(source),
        "runs": [dict(run) for run in runs],
        "notes": notes,
    }


def write_experiment_manifest(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment.json"
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
