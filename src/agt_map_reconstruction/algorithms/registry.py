"""Registry for built-in and external segmentation plugins."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable, Mapping, Any

import numpy as np

from .base import SegmentationResult, validate_points


SegmentFunction = Callable[[np.ndarray, Mapping[str, Any] | None], SegmentationResult]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    segment: SegmentFunction
    description: str = ""
    family: str = "ground_segmentation"
    version: str = "1"

    def run(self, points: np.ndarray, config: Mapping[str, Any] | None = None) -> SegmentationResult:
        result = self.segment(validate_points(points), config or {})
        if not isinstance(result, SegmentationResult):
            raise TypeError(
                f"Plugin {self.name!r} returned {type(result).__name__}; "
                "expected SegmentationResult"
            )
        metadata = dict(result.metadata)
        metadata.setdefault("algorithm", self.name)
        metadata.setdefault("plugin_version", self.version)
        metadata.setdefault("family", self.family)
        return SegmentationResult(result.ground_points, result.non_ground_points, metadata)


_ALGORITHMS: dict[str, AlgorithmSpec] = {}
_BUILTINS_LOADED = False


def register_algorithm(
    name: str,
    *,
    description: str = "",
    family: str = "ground_segmentation",
    version: str = "1",
):
    """Decorator used by built-in and third-party plugin modules."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("algorithm name must not be empty")

    def wrapper(func: SegmentFunction):
        if normalized in _ALGORITHMS:
            raise ValueError(f"Algorithm already registered: {normalized}")
        _ALGORITHMS[normalized] = AlgorithmSpec(
            name=normalized,
            segment=func,
            description=description,
            family=family,
            version=version,
        )
        return func

    return wrapper


def load_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    importlib.import_module("agt_map_reconstruction.algorithms.height_threshold")
    importlib.import_module("agt_map_reconstruction.algorithms.morphological_pmf")


def load_plugin_module(module_name: str) -> None:
    """Import an external module that registers plugins via register_algorithm."""
    importlib.import_module(module_name)


def get_algorithm(name: str) -> AlgorithmSpec:
    load_builtin_plugins()
    if name not in _ALGORITHMS:
        available = ", ".join(sorted(_ALGORITHMS)) or "<none>"
        raise KeyError(f"Unknown algorithm: {name}. Available: {available}")
    return _ALGORITHMS[name]


def run_algorithm(
    name: str,
    points: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> SegmentationResult:
    return get_algorithm(name).run(points, config)


def list_algorithms() -> list[str]:
    load_builtin_plugins()
    return sorted(_ALGORITHMS)


def list_algorithm_specs() -> list[AlgorithmSpec]:
    load_builtin_plugins()
    return [_ALGORITHMS[name] for name in sorted(_ALGORITHMS)]
