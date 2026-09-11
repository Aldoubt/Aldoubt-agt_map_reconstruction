# Development Log

## Project: agt_map_reconstruction

Purpose:

A standalone offline LiDAR map reconstruction, traversability, experiment, and
navigation-map validation repository. Runtime localization, local perception,
planning, control, behavior trees, and vehicle interfaces remain outside this
project.

---

# 2026-09-11 — Plugin Benchmark Pipeline Refactor

Status: implemented on `feature/plugin-benchmark-pipeline`.

## Problem

The repository had three incompatible algorithm execution paths:

1. `ground_segmentation.py` defined one `SegmentationResult`;
2. `registry.py` defined another `SegmentationResult`;
3. `height_threshold.py` and `morphological_pmf.py` returned dictionaries.

In addition:

- `run_benchmark.py` bypassed the registry;
- `run_compare.py` maintained a second hard-coded algorithm table;
- benchmark outputs had no stable machine-readable provenance manifest;
- Navigation Map V2 had generation validation but no standalone pre-runtime
  bundle validator.

## Implemented

### Canonical plugin API

Added:

```text
src/agt_map_reconstruction/algorithms/base.py
src/agt_map_reconstruction/algorithms/__init__.py
```

Canonical result:

```text
SegmentationResult
├── ground_points
├── non_ground_points
└── metadata
```

The old `ground_segmentation.py` is now a compatibility layer only.

### Registry-driven built-ins

Built-in plugins:

- `height_threshold`;
- `morphological_pmf` (explicitly documented as PMF-inspired, not canonical PMF).

External modules can register algorithms through `register_algorithm()` and be
loaded with `--plugin-module`.

### Plugin benchmark

Added:

```text
src/agt_map_reconstruction/benchmark.py
```

Each algorithm writes:

```text
benchmark.json
schema = agt.map_reconstruction.benchmark/v1
```

The benchmark enforces:

```text
ground_count + non_ground_count == input_count
```

### Experiment manifest

Added:

```text
src/agt_map_reconstruction/experiment.py
experiment.json
schema = agt.map_reconstruction.experiment/v1
```

This is the machine-readable handoff from algorithm comparison to later
reconstruction experiments.

### Map asset validation

Added:

```text
src/agt_map_reconstruction/map_asset_validation.py
tools/validate_map_assets.py
```

The validator checks the complete Navigation Map V2 bundle before it is handed
to MapManager/localization/Nav2.

### Tool consolidation

Canonical benchmark CLI:

```text
tools/run_benchmark.py
```

Legacy `tools/run_compare.py` now delegates to the canonical runner instead of
maintaining a second algorithm list.

### Engineering

Added:

- `configs/benchmark_example.yaml`;
- plugin/benchmark tests;
- map asset validation tests;
- GitHub Actions CI;
- generated-results `.gitignore`;
- Python/test constraints in `pyproject.toml`.

---

# Current Stable Architecture

```text
plugin
  |
  v
benchmark.json
  |
  v
experiment.json
  |
  v
reconstruction experiment
  |
  v
Navigation Map V2
  |
  v
asset_validation.json
  |
  v
runtime handoff
```

Human experiment decisions remain in `docs/EXPERIMENTS.md`.

---

# Existing Experiment Progress

## EXP001 — Ground Segmentation

Completed baseline work:

- height threshold;
- PMF-inspired baseline.

Next algorithm integrations should be plugins rather than new hard-coded
benchmark scripts.

Candidate adapters:

- Patchwork++;
- canonical PMF/PCL;
- CSF;
- LineFit.

## EXP002 — Agricultural Structure Recovery

Implemented:

- local elevation normalization;
- relative elevation;
- traversability baseline;
- row direction estimation;
- row-aware corridor recovery;
- centerline generation.

## EXP003 / EXP003.1 — Navigation Map V2

Implemented:

- Nav2-compatible PGM/YAML export;
- static/candidate semantic separation;
- pillar/wall/ridge hard-obstacle priority;
- clearance validation;
- static-obstacle semantic correction.

## EXP004-A — Polygon Footprint

Implemented and replay validated.

## EXP004-B1 — Constant Lateral Offset

Implemented and replay validated.

## EXP004-B2 — Smooth Lateral Route

Implemented and replay validated.

Main conclusion:

Many remaining failures occur at aisle entry/exit boundaries, not in the
in-aisle interior.

## Next Experiment — EXP004-C

Priority:

- explicit headland handoff poses;
- measured wheelbase/steering geometry;
- minimum-turning-radius / Ackermann constraints;
- failure classification between map, unknown, headland depth, and kinematics.

Do not increase in-aisle planning complexity before validating this handoff.

---

# Repository Boundary

This repository owns:

- offline point-cloud algorithm benchmarking;
- reconstruction experiments;
- elevation/traversability/corridor reasoning;
- navigation-map generation;
- footprint/route geometry validation;
- map asset acceptance.

This repository does not own:

- global localization;
- continuous localization tracking;
- local online obstacle perception;
- Nav2 planner/controller runtime;
- behavior trees;
- robot base control.

That boundary is intentional.
