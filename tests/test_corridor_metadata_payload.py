import importlib.util
from pathlib import Path

import numpy as np

from agt_map_reconstruction.maps.grid_geometry import GridMetadata


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_corridor_test.py"
    spec = importlib.util.spec_from_file_location("run_corridor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp002_metadata_payload_includes_grid_coordinate_contract():
    tool = _load_tool()
    assert callable(getattr(tool, "build_metadata_payload", None))

    grid = GridMetadata(
        resolution=0.05,
        origin_x=-12.5,
        origin_y=4.25,
        width=797,
        height=912,
    )
    maps = {"metadata": grid}
    direction = np.array([1.0, 0.0])
    corridor = np.ones((2, 3), dtype=bool)

    payload = tool.build_metadata_payload(
        maps=maps,
        angle=0.31,
        direction=direction,
        point_count=85912613,
        corridor=corridor,
    )

    assert payload["experiment"] == "EXP002"
    assert payload["grid"] == grid.to_dict()
    assert payload["row_angle_rad"] == 0.31
    assert payload["row_direction"] == [1.0, 0.0]
    assert payload["points"] == 85912613
    assert payload["corridor_cells"] == 6


def test_exp002_main_writes_grid_coordinate_contract_to_metadata_yaml(tmp_path, monkeypatch):
    import sys
    import yaml

    tool = _load_tool()
    grid = GridMetadata(
        resolution=0.05,
        origin_x=-2.0,
        origin_y=7.0,
        width=3,
        height=2,
    )
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    array = np.ones((2, 3), dtype=float)
    corridor = np.ones((2, 3), dtype=bool)

    monkeypatch.setattr(tool, "load_pcd", lambda path: points)
    monkeypatch.setattr(
        tool,
        "build_traversability_map",
        lambda value, resolution=0.05: {
            "height": array,
            "relative_height": array,
            "traversability": array,
            "metadata": grid,
        },
    )
    monkeypatch.setattr(tool, "estimate_row_direction", lambda value: (0.31, np.array([1.0, 0.0])))
    monkeypatch.setattr(tool, "extract_corridor", lambda *args, **kwargs: corridor)
    monkeypatch.setattr(tool, "skeletonize_corridor", lambda value: value)
    monkeypatch.setattr(tool, "save_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run_corridor_test.py", "--pcd", "dummy.pcd", "--output", str(tmp_path)])

    tool.main()

    payload = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    assert payload["grid"] == grid.to_dict()
    assert payload["row_angle_rad"] == 0.31
    assert payload["corridor_cells"] == 6


def test_exp002_main_forwards_resolution_to_grid_builder(tmp_path, monkeypatch):
    import sys

    tool = _load_tool()
    grid = GridMetadata(
        resolution=0.10,
        origin_x=0.0,
        origin_y=0.0,
        width=2,
        height=2,
    )
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    array = np.ones((2, 2), dtype=float)
    corridor = np.ones((2, 2), dtype=bool)
    seen = {}

    monkeypatch.setattr(tool, "load_pcd", lambda path: points)

    def fake_build(value, resolution=0.05):
        seen["resolution"] = resolution
        return {
            "height": array,
            "relative_height": array,
            "traversability": array,
            "metadata": grid,
        }

    monkeypatch.setattr(tool, "build_traversability_map", fake_build)
    monkeypatch.setattr(tool, "estimate_row_direction", lambda value: (0.0, np.array([1.0, 0.0])))
    monkeypatch.setattr(tool, "extract_corridor", lambda *args, **kwargs: corridor)
    monkeypatch.setattr(tool, "skeletonize_corridor", lambda value: value)
    monkeypatch.setattr(tool, "save_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_corridor_test.py",
            "--pcd", "dummy.pcd",
            "--output", str(tmp_path),
            "--resolution", "0.10",
        ],
    )

    tool.main()

    assert seen["resolution"] == 0.10
