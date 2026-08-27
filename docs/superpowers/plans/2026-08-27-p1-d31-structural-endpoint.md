# P1-D3.1 Structural Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover structural row terminations from bilateral ridge/occupied support, robustly fit common entry/exit structural boundaries, visualize them against geometric endpoints and clearance handoffs, and reuse frozen E0/E1/E2 evidence on the new D3.1 ROI without replaying rosbag.

**Architecture:** D3.1 is a separate topology authority layered on top of frozen P1 assets. A structural-profile module samples HARD/UNKNOWN evidence in narrow strips immediately outside each aisle, an endpoint detector finds persistent bilateral-support termination per row, a robust fitter estimates common entry/exit boundaries while retaining outliers, and CLI tools render full-map context plus re-evaluate frozen evidence arrays in the new ROI. Existing D3 outputs and `navigation_base_map.pgm` remain read-only.

**Tech Stack:** Python 3, NumPy, SciPy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-p1-d31-structural-endpoint-design.md`

## Global Constraints

- Existing P1-D3 remains a historical geometric-endpoint baseline and is never overwritten.
- Clearance handoff remains a navigation reachability asset and is never used as a fallback structural endpoint.
- UNKNOWN is never counted as structural support and is never promoted to FREE.
- D3.1 must use frozen upstream assets only; no rosbag replay or ray regeneration.
- Ambiguous rows remain explicit and are not automatically repaired.
- Robust common-boundary fitting is descriptive only; no automatic headland acceptance.
- `navigation_base_map.pgm` is read-only.
- `docs/EXPERIMENTS.md` remains the only canonical experiment record.

---

### Task 1: Bilateral Structural Support Profiles

**Files:**
- Create: `src/agt_map_reconstruction/maps/structural_endpoint_profile.py`
- Create: `tests/test_structural_endpoint_profile.py`

**Interfaces:**
- Consumes: `base_map: np.ndarray`, one `row_aisle` record with `polygon_xy` and `centerline_xy`, `resolution_m: float`, explicit strip width / longitudinal bin parameters.
- Produces: `build_structural_support_profile(...) -> dict` containing row-axis/cross-axis geometry, longitudinal bins, left/right HARD support fractions, left/right UNKNOWN fractions, bilateral-support flags, and source parameters.

- [ ] **Step 1: Write the failing synthetic tests**

Cover a corridor that extends to the map edge while both side structural strips terminate early, and a case where UNKNOWN occupies one strip. Assert that HARD contributes to structural support while UNKNOWN is reported separately and never counted as support.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_structural_endpoint_profile.py
```

Expected: FAIL because `structural_endpoint_profile` does not exist yet.

- [ ] **Step 3: Implement the minimal profile builder**

Use the row centerline to define a normalized longitudinal axis. Derive left/right strips from the aisle polygon cross-row bounds, rasterize only cells inside the requested strip widths, bin cells longitudinally at fixed `bin_size_m`, and compute per-bin HARD/UNKNOWN fractions. Keep profile generation independent of clearance handoff.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the same pytest command; expected PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agt_map_reconstruction/maps/structural_endpoint_profile.py tests/test_structural_endpoint_profile.py
git commit -m "feat: build bilateral structural support profiles"
```

### Task 2: Persistent Per-Row Structural Endpoint Detection

**Files:**
- Create: `src/agt_map_reconstruction/maps/structural_endpoint_detection.py`
- Create: `tests/test_structural_endpoint_detection.py`

**Interfaces:**
- Consumes: one profile dict from `build_structural_support_profile` plus explicit `min_support_fraction`, `min_persistence_m`, and `max_internal_gap_m`.
- Produces: `detect_structural_endpoints(profile, ...) -> dict` with independent entry/exit records and status in `ok_bilateral`, `ambiguous_single_side`, or `insufficient_structural_support`.

- [ ] **Step 1: Write failing tests for the three required statuses**

Synthetic profiles must prove: bilateral structure ending before free corridor -> endpoint at structural termination; one reliable side only -> ambiguous; no persistent structure -> insufficient. Add a short internal hole that must not split a valid persistent run.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_structural_endpoint_detection.py
```

- [ ] **Step 3: Implement persistent-run detection**

Threshold left/right support independently, close only gaps no longer than `max_internal_gap_m`, find the longest/persistent support run in row orientation, require bilateral persistence for `ok_bilateral`, and retain single-side candidates only as ambiguous evidence. Never substitute raw endpoint/handoff on failure.

- [ ] **Step 4: Run tests and verify GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/agt_map_reconstruction/maps/structural_endpoint_detection.py tests/test_structural_endpoint_detection.py
git commit -m "feat: detect persistent structural row endpoints"
```

### Task 3: Robust Common Structural Boundary Fit

**Files:**
- Create: `src/agt_map_reconstruction/maps/structural_endpoint_boundary.py`
- Create: `tests/test_structural_endpoint_boundary.py`

**Interfaces:**
- Consumes: per-row structural endpoint records and row/cross axes.
- Produces: `fit_structural_endpoint_boundaries(...) -> dict` containing common entry/exit robust line fits, every row residual, inlier/outlier flag, fit method, and thresholds.

- [ ] **Step 1: Write failing tests**

Use 5 synthetic rows with 4 aligned endpoints and one endpoint displaced by several metres. Assert that the common boundary remains near the 4-row consensus and the displaced row remains present but flagged as an outlier. Add reversed-centerline orientation coverage.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_structural_endpoint_boundary.py
```

- [ ] **Step 3: Implement robust fitting**

Use median/MAD-based residual gating around an initial linear fit in `(cross_row v, longitudinal u)` space, then refit on inliers. Preserve all candidates and expose `residual_m`, `inlier`, method name, and threshold. Do not optimize thresholds automatically.

- [ ] **Step 4: Run tests and verify GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/agt_map_reconstruction/maps/structural_endpoint_boundary.py tests/test_structural_endpoint_boundary.py
git commit -m "feat: robustly fit structural endpoint boundaries"
```

### Task 4: D3.1 Bundle Builder and Full-Map Geometry Audit

**Files:**
- Create: `tools/build_structural_endpoint_d31.py`
- Create: `tests/test_build_structural_endpoint_d31_cli.py`

**Interfaces:**
- Consumes: `navigation_base_map.pgm`, `row_band_regions.json`, `aisle_handoffs.json`, explicit structural-profile and detection parameters.
- Produces under one output directory: `structural_endpoint_profiles.json`, `structural_endpoint_boundary.json`, `structural_endpoint_context.png`, `entry_structural_endpoint_context.png`, `exit_structural_endpoint_context.png`.

- [ ] **Step 1: Write failing CLI contract tests**

Assert parser requirements, deterministic output keys, policy flags (`navigation_map_modified=false`, `semantic_promotion=false`, `automatic_acceptance=false`), and that output retains geometric endpoint, clearance handoff, and structural endpoint side-by-side.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_build_structural_endpoint_d31_cli.py
```

- [ ] **Step 3: Implement CLI orchestration and visualization**

Read the canonical PGM in grid orientation, filter exactly the existing clearance-width-eligible aisles, build profiles/endpoints, fit robust boundaries, write provenance-rich JSON, and render the full map with recovered aisles, raw endpoints, clearance handoffs, structural endpoints, common structural lines, ambiguous rows, and outliers.

- [ ] **Step 4: Run tests and verify GREEN**

- [ ] **Step 5: Commit**

```bash
git add tools/build_structural_endpoint_d31.py tests/test_build_structural_endpoint_d31_cli.py
git commit -m "feat: build P1 D3.1 structural endpoint assets"
```

### Task 5: Re-evaluate Frozen E0/E1/E2 Evidence in the D3.1 ROI

**Files:**
- Create: `src/agt_map_reconstruction/maps/structural_endpoint_roi.py`
- Create: `tools/evaluate_structural_endpoint_evidence.py`
- Create: `tests/test_structural_endpoint_roi.py`

**Interfaces:**
- Consumes: `structural_endpoint_boundary.json`, frozen navigation map, existing ground-reference grid, existing ray/scan support grids.
- Produces: D3.1 entry/exit ROI masks and evaluation JSON reporting strict-safe topology plus observation-sufficiency counts against the new structural boundary.

- [ ] **Step 1: Write failing tests**

Assert that ROI lies outward from the structural boundary and within the same frozen cross-row span, that changing only the boundary shifts ROI while the evidence arrays are unchanged, and that no navigation map mutation occurs.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_structural_endpoint_roi.py
```

- [ ] **Step 3: Implement ROI reconstruction and evaluator**

Reuse the existing D3 geometric conventions for row/cross axes and strict-safe clearance, but source entry/exit line fits from D3.1 structural boundary JSON. Read existing `ground_reference.npy`, `ray_free_support_count.npy`, and `scan_free_support_count.npy` only; do not regenerate them.

- [ ] **Step 4: Run tests and verify GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/agt_map_reconstruction/maps/structural_endpoint_roi.py tools/evaluate_structural_endpoint_evidence.py tests/test_structural_endpoint_roi.py
git commit -m "feat: evaluate frozen evidence on D3.1 structural ROI"
```

### Task 6: Real Greenhouse Replay and Canonical Experiment Record

**Files:**
- Modify: `docs/EXPERIMENTS.md`
- No new experiment-summary document.

**Interfaces:**
- Consumes: real greenhouse frozen map/regions/handoffs and already generated E0/E1/E2 grids.
- Produces: one D3.1 output directory plus canonical measured conclusions in `docs/EXPERIMENTS.md` after user/local replay verification.

- [ ] **Step 1: Run the full focused test suite**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_structural_endpoint_profile.py \
  tests/test_structural_endpoint_detection.py \
  tests/test_structural_endpoint_boundary.py \
  tests/test_build_structural_endpoint_d31_cli.py \
  tests/test_structural_endpoint_roi.py \
  tests/test_endpoint_geometry_audit.py
```

Expected: all PASS.

- [ ] **Step 2: Build real D3.1 structural endpoint assets**

Run `tools/build_structural_endpoint_d31.py` against the exact source map/regions/handoffs referenced by frozen D3. Record all explicit parameters in output provenance.

- [ ] **Step 3: Inspect real full-map and endpoint crops**

Confirm structural endpoints correspond to row termination rather than greenhouse walls or safe-core truncations. Ambiguous rows must remain explicit.

- [ ] **Step 4: Re-evaluate frozen E0/E1/E2 arrays on the new ROI**

Run `tools/evaluate_structural_endpoint_evidence.py` using the already generated confidence-gated ground reference and strongest frozen scan-support grid. Do not replay rosbag.

- [ ] **Step 5: Update the canonical experiment record**

Write only measured D3.1 results to `docs/EXPERIMENTS.md`: previous D3 raw-endpoint semantics, D3.1 structural definition, ambiguous/outlier rows, robust fit statistics, reuse of frozen E0/E1/E2 evidence, and whether endpoint-scale topology changes under the structural ROI.

- [ ] **Step 6: Commit**

```bash
git add docs/EXPERIMENTS.md
git commit -m "docs: freeze P1 D3.1 structural endpoint results"
```
