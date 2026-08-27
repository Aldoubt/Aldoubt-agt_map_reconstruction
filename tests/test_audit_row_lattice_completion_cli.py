import importlib.util
from pathlib import Path


def _load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "audit_row_lattice_completion.py"
    spec = importlib.util.spec_from_file_location("audit_row_lattice_completion", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_exposes_conservative_lattice_audit_contract():
    module = _load_tool()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--map", "navigation.pgm",
            "--row-band-regions", "row_band_regions.json",
            "--output", "out",
        ]
    )

    assert args.map == "navigation.pgm"
    assert args.row_band_regions == "row_band_regions.json"
    assert args.output == "out"
    assert args.min_observed_slots == 4
