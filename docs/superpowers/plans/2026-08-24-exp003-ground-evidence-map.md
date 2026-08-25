# EXP003 Ground Evidence Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, conservative ground/evidence/costmap pipeline from the authoritative global PCD.

**Architecture:** A chunked rasterizer produces per-cell low-height and density statistics. A separate ground-evidence module estimates bounded continuous ground, classifies four evidence states, and creates an inflated navigation costmap; an experiment module owns immutable outputs and traceability.

**Tech Stack:** Python 3, NumPy, SciPy, Matplotlib, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-08-24-exp003-ground-evidence-map-design.md`

## Global Constraints

- EXP002 behavior and artifacts remain unchanged.
- `docs/EXPERIMENTS.md` remains the only canonical experiment record.
- Unsupported cells remain unknown; interpolation is bounded and separately labeled.
- The 85M-point input is processed in chunks without allocating one index per point.
- Real-data quality is not claimed without inspecting the operator-produced run.

---

### Task 1: Robust elevation statistics

**Files:**
- Create: `src/agt_map_reconstruction/maps/elevation_statistics.py`
- Test: `tests/test_exp003.py`

**Interfaces:**
- Produces: `ElevationStatistics`, `points_to_elevation_statistics(points, resolution, chunk_size, low_quantile, histogram_bins)`

- [ ] **Step 1: Write failing tests** for grid origin, point counts, finite-point filtering, rejection of a single low outlier by the lower histogram estimate, and chunk-size invariance.
- [ ] **Step 2: Run** `python -m pytest tests/test_exp003.py -q` and confirm missing-module/API failures.
- [ ] **Step 3: Implement** a two-pass chunked rasterizer: bounds/count/min/max first, fixed-bin per-cell histogram second, then select the requested lower cumulative bin.
- [ ] **Step 4: Run** `python -m pytest tests/test_exp003.py -q` and confirm the rasterizer tests pass.

### Task 2: Ground evidence and navigation costmap

**Files:**
- Create: `src/agt_map_reconstruction/maps/ground_evidence.py`
- Modify: `tests/test_exp003.py`

**Interfaces:**
- Produces: `EvidenceClass`, `GroundEvidenceConfig`, `build_ground_evidence(low_height, point_count, config)`, `build_navigation_costmap(evidence, config)`

- [ ] **Step 1: Write failing tests** demonstrating measured free ground, elevated obstacles, low-density unknown cells, bounded interpolated holes, large unknown holes, and metric obstacle inflation.
- [ ] **Step 2: Run** `python -m pytest tests/test_exp003.py -q` and confirm missing-API failures.
- [ ] **Step 3: Implement** NaN-aware percentile ground estimation, distance-bounded interpolation, four-state classification, and Euclidean inflation.
- [ ] **Step 4: Run** `python -m pytest tests/test_exp003.py -q` and confirm all map semantics pass.

### Task 3: Reproducible EXP003 runner and artifacts

**Files:**
- Create: `src/agt_map_reconstruction/experiments/exp003.py`
- Create: `tools/run_ground_evidence_test.py`
- Modify: `tests/test_exp003.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `Exp003Config`, `run_exp003(points, config)`, `write_exp003_results(result, run_dir, metadata)` and a CLI accepting `--pcd`, `--output`, `--run-id`, `--hash-pcd`, plus all algorithm parameters.

- [ ] **Step 1: Write failing tests** for configuration validation, immutable run creation, expected NPY/PNG/YAML artifacts, metric fields, and PCD-frame origin preservation.
- [ ] **Step 2: Run** `python -m pytest tests/test_exp003.py -q` and confirm missing experiment APIs fail.
- [ ] **Step 3: Implement** orchestration, authoritative numeric outputs, previews, metadata, metrics, duplicate-run rejection, and CLI argument plumbing.
- [ ] **Step 4: Run** `python -m pytest tests/test_exp003.py -q` and confirm artifact tests pass.

### Task 4: Canonical experiment record and verification

**Files:**
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/experiments/EXP_003_corridor_baseline.md`
- Modify: `README.md`
- Modify: `docs/DEVELOPMENT_LOG.md`

**Interfaces:**
- Consumes: verified EXP003 CLI and artifact schema.
- Produces: one canonical record with exact command, output tree, limitations, acceptance state, and deferred EXP004 probe.

- [ ] **Step 1: Convert** the old EXP003 corridor file into a pointer that identifies it as a pre-EXP002 historical note without duplicating conclusions.
- [ ] **Step 2: Update** `docs/EXPERIMENTS.md` with EXP002 real-run observations and EXP003 implementation/acceptance status; document EXP004 only as deferred.
- [ ] **Step 3: Run** the complete test suite, `python -m compileall -q src tools tests`, `git diff --check`, CLI `--help`, and a synthetic PCD end-to-end smoke run.
- [ ] **Step 4: Inspect** metadata, metrics, evidence labels, cost values, duplicate-run failure, and Git diff before committing.
