"""Reusable benchmark execution and result serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .algorithms import SegmentationResult, get_algorithm


BENCHMARK_SCHEMA = "agt.map_reconstruction.benchmark/v1"


@dataclass(frozen=True)
class BenchmarkRecord:
    algorithm: str
    plugin_version: str
    config: dict[str, Any]
    result: SegmentationResult
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_SCHEMA,
            "stage": "plugin_benchmark",
            "algorithm": self.algorithm,
            "plugin_version": self.plugin_version,
            "config": self.config,
            "metrics": self.metrics,
            "metadata": self.result.metadata,
        }


def benchmark_algorithm(
    points: np.ndarray,
    algorithm: str,
    config: Mapping[str, Any] | None = None,
) -> BenchmarkRecord:
    spec = get_algorithm(algorithm)
    normalized_config = dict(config or {})

    start = perf_counter()
    result = spec.run(points, normalized_config)
    elapsed = perf_counter() - start

    input_count = int(len(points))
    ground_count = int(len(result.ground_points))
    non_ground_count = int(len(result.non_ground_points))
    if ground_count + non_ground_count != input_count:
        raise ValueError(
            f"Plugin {algorithm!r} lost or duplicated points: "
            f"{ground_count}+{non_ground_count}!={input_count}"
        )

    metrics = {
        "input_points": input_count,
        "ground_points": ground_count,
        "non_ground_points": non_ground_count,
        "ground_ratio": (ground_count / input_count) if input_count else 0.0,
        "non_ground_ratio": (non_ground_count / input_count) if input_count else 0.0,
        "elapsed_seconds": float(elapsed),
    }

    return BenchmarkRecord(
        algorithm=spec.name,
        plugin_version=spec.version,
        config=normalized_config,
        result=result,
        metrics=metrics,
    )


def write_benchmark_record(record: BenchmarkRecord, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "benchmark.json"
    path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
