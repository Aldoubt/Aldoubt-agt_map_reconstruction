# Plugin Benchmark → Experiment → Map Asset Validation

This document defines the stable execution contract for AGT Map Reconstruction.

## Architecture

```text
PCD / point cloud
      |
      v
registered segmentation plugin
      |
      v
SegmentationResult
      |
      v
plugin benchmark
      |
      +-- benchmark.json
      +-- visualizations
      +-- elevation / traversability artifacts
      |
      v
experiment manifest
      |
      +-- experiment.json
      |
      v
reconstruction experiments
      |
      v
Navigation Map V2 bundle
      |
      v
map asset validation
      |
      +-- asset_validation.json
      |
      v
MapManager / localization / Nav2 handoff
```

The runtime navigation stack is intentionally outside this repository.

## 1. Plugin contract

All ground-segmentation implementations use the public API in:

```text
src/agt_map_reconstruction/algorithms/
├── base.py
├── registry.py
├── height_threshold.py
└── morphological_pmf.py
```

A plugin exposes:

```python
segment(points: np.ndarray, config: Mapping) -> SegmentationResult
```

Canonical result:

```python
SegmentationResult(
    ground_points=np.ndarray,       # Nx3
    non_ground_points=np.ndarray,   # Mx3
    metadata=dict,
)
```

The result also provides legacy aliases `ground`, `non_ground`, and mapping-style
access to avoid breaking the existing visualization helpers.

### Registering a plugin

```python
from agt_map_reconstruction.algorithms import (
    SegmentationResult,
    register_algorithm,
)

@register_algorithm(
    "my_segmenter",
    description="Example external segmenter",
    version="1",
)
def segment(points, config=None):
    ...
    return SegmentationResult(ground, non_ground, {"config": dict(config or {})})
```

An external module can be loaded without editing the repository registry:

```bash
python tools/run_benchmark.py \
  --plugin-module my_package.agt_plugin \
  --algorithm my_segmenter \
  --pcd map.pcd
```

This is the preferred extension mechanism for Patchwork++, CSF, PCL PMF, or
future learned segmenters.

## 2. Built-in plugins

Current built-ins:

| Plugin | Status | Meaning |
|---|---|---|
| `height_threshold` | implemented | global percentile/height baseline |
| `morphological_pmf` | implemented | lightweight PMF-inspired histogram baseline |

The second implementation is **not** a canonical Progressive Morphological
Filter. Production PMF, CSF, Patchwork++, and LineFit should be added as
separate plugins rather than hidden behind the current baseline name.

List available plugins:

```bash
python tools/run_benchmark.py --list-algorithms
```

## 3. Benchmark configuration

Example:

```yaml
algorithms:
  height_threshold:
    height_threshold: 0.15
    ground_percentile: 10.0

  morphological_pmf:
    height_threshold: 0.25
    bins: 200
```

Repository example:

```text
configs/benchmark_example.yaml
```

Run:

```bash
python tools/run_benchmark.py \
  --pcd /path/to/map.pcd \
  --config configs/benchmark_example.yaml \
  --output results/EXP001
```

Select a subset by repeating `--algorithm`:

```bash
python tools/run_benchmark.py \
  --pcd map.pcd \
  --algorithm height_threshold \
  --algorithm morphological_pmf
```

## 4. Benchmark output contract

Each algorithm owns its own output directory:

```text
results/EXP001/
├── experiment.json
├── height_threshold/
│   ├── benchmark.json
│   ├── ground.png
│   ├── non_ground.png
│   ├── overlay.png
│   ├── ground_height_grid.png
│   ├── height_map.png
│   ├── relative_height.png
│   └── traversability.png
└── morphological_pmf/
    └── ...
```

`benchmark.json` schema:

```text
agt.map_reconstruction.benchmark/v1
```

It records:

- plugin name and version;
- effective config;
- input / ground / non-ground counts;
- ground and non-ground ratios;
- execution time;
- plugin metadata.

A plugin must preserve the input partition:

```text
ground_points + non_ground_points == input_points
```

The benchmark layer rejects a plugin that loses or duplicates points.

## 5. Experiment contract

The root `experiment.json` is the provenance handoff between plugin execution
and higher-level reconstruction experiments.

Schema:

```text
agt.map_reconstruction.experiment/v1
```

It records:

- experiment ID;
- source PCD path and point count;
- selected plugin runs;
- benchmark record paths;
- metrics;
- generated artifacts;
- operator/research notes.

Large PCDs and generated results remain outside source control. The manifest is
the machine-readable record of what produced an experiment output.

Human conclusions and cross-experiment decisions remain in:

```text
docs/EXPERIMENTS.md
```

## 6. Reconstruction stage

Experiment outputs may feed the existing map pipeline:

```text
segmented / reconstructed geometry
      |
      v
relative elevation
      |
      v
traversability
      |
      v
row / corridor recovery
      |
      v
semantic labels
      |
      v
Navigation Map V2
```

The reconstruction algorithms are not required to be segmentation plugins.
Only interchangeable algorithm families should use the plugin registry.

## 7. Map asset contract

A Navigation Map V2 bundle must contain:

```text
navigation_base_map.pgm
navigation_base_map.yaml
candidate_mask.npy
static_obstacle_mask.npy
validation.json
```

Validate before runtime integration:

```bash
python tools/validate_map_assets.py \
  --bundle results/EXP003/navigation-map-v2
```

Output:

```text
asset_validation.json
```

Schema:

```text
agt.map_reconstruction.map_asset_validation/v1
```

Validation currently checks:

- all required files exist;
- PGM is decodable;
- PGM uses only canonical `0 / 205 / 254` values;
- YAML uses Nav2 `trinary` mode;
- resolution and thresholds are valid;
- YAML image reference exists;
- candidate/static masks match map dimensions;
- static obstacle cells are never exported as free;
- the original generation validation does not already report a semantic/YAML failure.

A failed asset validation must block handoff to MapManager/Nav2.

## 8. Acceptance gates

The repository now has three explicit gates:

### Gate A — Plugin

A plugin is accepted when:

- it registers successfully;
- it returns `SegmentationResult`;
- both arrays are Nx3;
- it preserves the input partition;
- its parameters are explicit.

### Gate B — Experiment

An experiment is accepted when:

- every run has `benchmark.json`;
- `experiment.json` records source and selected algorithms;
- generated assets can be traced back to a run/config;
- conclusions are recorded in `docs/EXPERIMENTS.md`.

### Gate C — Map asset

A Navigation Map V2 bundle is accepted when:

- `validate_map_assets.py` returns exit code 0;
- static geometry is not promoted to free;
- YAML/PGM/masks are internally consistent;
- robot-footprint and route-level tests are run as required by the experiment stage.

## 9. Test command

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

CI runs the same test suite on Ubuntu 22.04 / Python 3.10.

## Design rule

**Plugins compare algorithms. Experiments preserve provenance. Asset validation
protects the runtime boundary.**

Do not mix those responsibilities into one script or move Nav2 runtime behavior
back into this repository.
