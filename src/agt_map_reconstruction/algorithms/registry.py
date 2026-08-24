from dataclasses import dataclass
from typing import Callable


@dataclass
class SegmentationResult:
    ground_points: object
    non_ground_points: object
    metadata: dict


_ALGORITHMS = {}


def register_algorithm(name: str):
    def wrapper(func: Callable):
        _ALGORITHMS[name] = func
        return func
    return wrapper


def get_algorithm(name: str):
    if name not in _ALGORITHMS:
        raise KeyError(f"Unknown algorithm: {name}")
    return _ALGORITHMS[name]


def list_algorithms():
    return sorted(_ALGORITHMS.keys())
