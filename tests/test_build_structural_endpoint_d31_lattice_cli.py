import importlib.util
from pathlib import Path


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_structural_endpoint_d31_lattice.py"
    spec = importlib.util.spec_from_file_location("build_structural_endpoint_d31_lattice", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_requires_frozen_map_and_lattice_assets():
    module = _load_tool()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--map", "navigation.pgm",
            "--row-lattice-completion", "row_lattice_completion.json",
            "--output", "out",
        ]
    )

    assert args.map == "navigation.pgm"
    assert args.row_lattice_completion == "row_lattice_completion.json"
    assert args.output == "out"
    assert args.bin_size_m == 0.10
    assert args.min_support_fraction == 0.50
    assert args.max_fit_rmse_m == 0.50
