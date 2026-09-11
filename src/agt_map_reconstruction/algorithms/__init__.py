"""Ground-segmentation plugin API."""

from .base import SegmentationResult
from .registry import (
    AlgorithmSpec,
    get_algorithm,
    list_algorithm_specs,
    list_algorithms,
    load_plugin_module,
    register_algorithm,
    run_algorithm,
)

__all__ = [
    "AlgorithmSpec",
    "SegmentationResult",
    "get_algorithm",
    "list_algorithm_specs",
    "list_algorithms",
    "load_plugin_module",
    "register_algorithm",
    "run_algorithm",
]
