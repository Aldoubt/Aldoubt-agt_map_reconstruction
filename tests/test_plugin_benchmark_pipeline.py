import numpy as np

from agt_map_reconstruction.algorithms import (
    SegmentationResult,
    get_algorithm,
    list_algorithms,
    register_algorithm,
    run_algorithm,
)
from agt_map_reconstruction.benchmark import (
    BENCHMARK_SCHEMA,
    benchmark_algorithm,
)
from agt_map_reconstruction.experiment import (
    EXPERIMENT_SCHEMA,
    build_experiment_manifest,
)


def _points():
    return np.asarray(
        [
            [0.0, 0.0, 0.00],
            [1.0, 0.0, 0.02],
            [2.0, 0.0, 0.04],
            [0.0, 1.0, 0.50],
            [1.0, 1.0, 0.80],
        ],
        dtype=np.float32,
    )


def test_builtin_algorithms_are_registry_driven():
    names = list_algorithms()
    assert "height_threshold" in names
    assert "morphological_pmf" in names

    spec = get_algorithm("height_threshold")
    assert spec.family == "ground_segmentation"
    assert spec.version


def test_all_builtin_plugins_return_canonical_result():
    points = _points()
    for name in list_algorithms():
        result = run_algorithm(name, points)
        assert isinstance(result, SegmentationResult)
        assert result.ground_points.ndim == 2
        assert result.non_ground_points.ndim == 2
        assert len(result.ground_points) + len(result.non_ground_points) == len(points)
        assert result["ground"] is result.ground_points
        assert result.ground is result.ground_points
        assert result.metadata["algorithm"] == name


def test_external_plugin_can_register_through_public_contract():
    name = "unit_test_all_ground"
    if name not in list_algorithms():

        @register_algorithm(name, description="test plugin", version="test")
        def _segment(points, config=None):
            return SegmentationResult(
                points,
                np.empty((0, 3), dtype=points.dtype),
                {"config": dict(config or {})},
            )

    result = run_algorithm(name, _points(), {"demo": True})
    assert len(result.ground_points) == len(_points())
    assert result.metadata["algorithm"] == name
    assert result.metadata["plugin_version"] == "test"


def test_benchmark_record_has_stable_schema_and_partition_metrics():
    points = _points()
    record = benchmark_algorithm(
        points,
        "height_threshold",
        {"height_threshold": 0.1},
    )
    payload = record.to_dict()

    assert payload["schema"] == BENCHMARK_SCHEMA
    assert payload["stage"] == "plugin_benchmark"
    assert payload["metrics"]["input_points"] == len(points)
    assert (
        payload["metrics"]["ground_points"]
        + payload["metrics"]["non_ground_points"]
        == len(points)
    )


def test_experiment_manifest_is_explicit_handoff_contract():
    manifest = build_experiment_manifest(
        experiment_id="EXP_TEST",
        source={"type": "pcd", "path": "map.pcd"},
        runs=[{"algorithm": "height_threshold", "benchmark_record": "x/benchmark.json"}],
    )

    assert manifest["schema"] == EXPERIMENT_SCHEMA
    assert manifest["stage"] == "experiment"
    assert manifest["experiment_id"] == "EXP_TEST"
    assert manifest["runs"][0]["algorithm"] == "height_threshold"
