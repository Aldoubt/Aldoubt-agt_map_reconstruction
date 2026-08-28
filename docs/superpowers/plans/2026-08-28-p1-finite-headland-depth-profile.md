# P1 Finite Headland Depth Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a finite, endpoint-relative headland depth evaluation interface that measures how trusted-ground, ray, and repeated-scan evidence decay with outward distance from the frozen D3.1 structural termination uncertainty band.

**Architecture:** Reuse the frozen fused structural endpoint and uncertainty artifacts to construct mutually exclusive finite entry/exit depth-band masks. Reuse the existing frozen navigation/ground/ray/scan grids to evaluate evidence in each band, then reuse the K8/K16 confidence grids for per-depth ground-gate sensitivity. Preserve the failed unbounded/site-flood diagnostics as reference-only assets; do not infer or repair a physical greenhouse boundary.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pytest, existing `agt_map_reconstruction.maps` geometry/evidence modules and JSON/NPY result conventions.

**Spec:** `docs/superpowers/specs/2026-08-28-p1-finite-headland-depth-profile-design.md`

## Global Constraints

- D3.1 fused structural geometry remains frozen; do not change lattice slots, PGM/3D fusion rules, ridge endpoints, p95 uncertainty, or unresolved-ridge classification.
- Depth zero is the outward edge of the frozen structural uncertainty band, not the center trend.
- Default diagnostic edges are exactly `0.0, 0.5, 1.0, 2.0, 4.0` metres; they are not automatically optimized or selected.
- Depth bands are restricted to the frozen structural/lattice cross-row domain.
- `R_L18_L19` and any other unresolved ridge cross-span are excluded from every resolved depth band and reported separately.
- No mask may extend beyond the explicit maximum depth edge merely because map cells remain UNKNOWN.
- No physical site-interior mask, HARD-boundary flood fill, wall repair, doorway classification, or inferred greenhouse polygon participates in the main evaluation chain.
- Existing frozen ground, ray-support, and unique-scan-support arrays are reused; no rosbag replay or ray regeneration is triggered by this work.
- Trusted-ground eligibility is not semantic free space or navigation acceptance.
- All new outputs remain evaluation-only: `automatic_acceptance=false`, `navigation_map_modified=false`, and `semantic_promotion=false`.
- `docs/EXPERIMENTS.md` remains the only canonical experiment record.

---

## File Structure

- Create `src/agt_map_reconstruction/maps/headland_depth_profile.py` — pure geometry builder for finite entry/exit depth masks plus boundary-uncertainty and unresolved masks.
- Create `tools/build_headland_depth_profile.py` — CLI, provenance writer, NPY mask writer, and full-map diagnostic visualization.
- Create `tests/test_headland_depth_profile.py` and `tests/test_headland_depth_profile_cli.py` — geometry and CLI contracts.
- Modify `src/agt_map_reconstruction/maps/structural_endpoint_uncertainty_evidence.py` — expose the existing per-ROI observation summary as a public reusable helper without changing current uncertainty-ROI behavior.
- Create `src/agt_map_reconstruction/maps/headland_depth_evidence.py` — apply the reusable observation summary to every finite depth band.
- Create `tools/evaluate_headland_depth_evidence.py` — JSON + compact depth-response plot from frozen evidence arrays.
- Create `tests/test_headland_depth_evidence.py` and `tests/test_headland_depth_evidence_cli.py`.
- Create `src/agt_map_reconstruction/maps/headland_depth_ground_gate_sweep.py` — per-depth K8/K16 support-distance/disagreement sweep.
- Create `tools/sweep_headland_depth_ground_gate.py` — reproducible sweep CLI and compact sensitivity JSON/plot.
- Create `tests/test_headland_depth_ground_gate_sweep.py` and `tests/test_headland_depth_ground_gate_sweep_cli.py`.
- Create `src/agt_map_reconstruction/maps/headland_depth_reference_comparison.py` — reference-only comparison of finite-depth evidence against the historical unbounded outward diagnostic.
- Create `tools/compare_headland_depth_with_unbounded.py` and `tests/test_headland_depth_reference_comparison.py`.
- Modify `docs/EXPERIMENTS.md` only after the real greenhouse replay fields are measured.

---

### Task 1: Finite Headland Depth Geometry Core

**Files:**
- Create: `src/agt_map_reconstruction/maps/headland_depth_profile.py`
- Test: `tests/test_headland_depth_profile.py`

**Interfaces:**
- Consumes: frozen `structural_endpoint_fused.json` payload, frozen `structural_endpoint_uncertainty_fused.json` payload, `grid_shape_yx`, and explicit depth edges in metres.
- Produces: `build_headland_depth_profile(fused_bundle, uncertainty_payload, *, grid_shape_yx, depth_edges_m=(0.0, 0.5, 1.0, 2.0, 4.0), uncertainty_quantile="p95") -> tuple[dict, dict[str, np.ndarray]]`.
- Mask keys: `entry_boundary_uncertainty`, `exit_boundary_uncertainty`, `entry_depth_0_0p5`, `entry_depth_0p5_1`, `entry_depth_1_2`, `entry_depth_2_4`, corresponding exit keys, and `structurally_unresolved_cross` for default edges. For arbitrary edges, names are deterministically encoded from metres.

- [ ] **Step 1: Write failing tests for finite extent, outward direction, and unresolved exclusion**

```python
def test_default_depth_bands_are_finite_opposite_and_exclude_unresolved():
    fused, uncertainty = synthetic_fused_geometry()
    result, masks = build_headland_depth_profile(
        fused,
        uncertainty,
        grid_shape_yx=(20, 30),
        depth_edges_m=[0.0, 0.5, 1.0, 2.0, 4.0],
        uncertainty_quantile="p95",
    )
    assert result["depth_edges_m"] == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert result["max_outward_depth_m"] == 4.0
    assert not np.any(masks["entry_depth_0_0p5"] & masks["exit_depth_0_0p5"])
    assert not np.any(masks["entry_depth_0_0p5"] & masks["structurally_unresolved_cross"])
    assert not np.any(masks["exit_depth_0_0p5"] & masks["structurally_unresolved_cross"])
```

Also add explicit assertions that the entry band lies on the negative-`u` side of `center-p95`, the exit band lies on the positive-`u` side of `center+p95`, and no cell farther than 4.0 m from those origins is selected.

- [ ] **Step 2: Run the new core test and verify it fails**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_headland_depth_profile.py
```

Expected: FAIL because `headland_depth_profile.py` and `build_headland_depth_profile` do not exist.

- [ ] **Step 3: Implement shared geometry helpers without depending on physical walls**

Implement the following private helpers in `headland_depth_profile.py`:

```python
def _validate_depth_edges(depth_edges_m):
    edges = [float(v) for v in depth_edges_m]
    if len(edges) < 2 or not np.isclose(edges[0], 0.0):
        raise ValueError("depth_edges_m must start at 0.0 and contain at least two edges")
    if any(not np.isfinite(v) or v < 0.0 for v in edges):
        raise ValueError("depth_edges_m must be finite and non-negative")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("depth_edges_m must be strictly increasing")
    return edges
```

Reuse the axis/resolution, cross-domain, unresolved-cross, and side-geometry semantics already present in `structural_endpoint_uncertainty_roi.py`, but keep this module independent of any site-interior mask. Compute `outward_depth_m` as:

```python
entry_depth_m = (center - half - u) * resolution
exit_depth_m = (u - center - half) * resolution
```

A resolved finite-band cell must satisfy `depth_lo <= outward_depth_m < depth_hi`, be inside the structural cross-domain, and not belong to an unresolved ridge cross-span.

- [ ] **Step 4: Add validation tests for bad depth edges and orientation normalization**

Add tests that reject `[0, 1, 0.5]`, `[0, 0.5, 0.5]`, and `[0.5, 1.0]`, and a reversed-source-centerline fixture whose normalized frozen axes produce identical masks.

- [ ] **Step 5: Run core tests and existing uncertainty ROI regression tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_headland_depth_profile.py \
  tests/test_structural_endpoint_uncertainty_roi.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/agt_map_reconstruction/maps/headland_depth_profile.py tests/test_headland_depth_profile.py
git commit -m "feat: add finite headland depth geometry"
```

---

### Task 2: Geometry CLI and Full-Map Audit Visualization

**Files:**
- Create: `tools/build_headland_depth_profile.py`
- Test: `tests/test_headland_depth_profile_cli.py`

**Interfaces:**
- Consumes: `--fused-structural-bundle`, `--fused-uncertainty`, `--depth-edges-m`, `--uncertainty-quantile`, `--output`.
- Produces: `headland_depth_profile.json`, one NPY per mask, and `headland_depth_profile.png`.

- [ ] **Step 1: Write failing CLI test with a small synthetic map and frozen structural payloads**

The test must invoke the real script with:

```text
--depth-edges-m 0 0.5 1 2 4
--uncertainty-quantile p95
```

and assert the JSON contains:

```python
assert payload["method"] == "finite_structural_headland_depth_profile"
assert payload["depth_edges_m"] == [0.0, 0.5, 1.0, 2.0, 4.0]
assert payload["policy"]["physical_site_boundary_required"] is False
assert payload["policy"]["navigation_map_modified"] is False
assert payload["policy"]["semantic_promotion"] is False
```

- [ ] **Step 2: Run the CLI test and verify it fails**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_headland_depth_profile_cli.py
```

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement CLI and deterministic mask filenames**

Use the frozen map path only as the visualization background. Record source paths exactly in JSON. Store mask names in `mask_files`; do not embed mask arrays in JSON.

- [ ] **Step 4: Implement the diagnostic figure**

Render the canonical map with:

- entry boundary-uncertainty band;
- exit boundary-uncertainty band;
- each finite entry/exit depth band with progressively distinct alpha intensity;
- unresolved cross-strip in red;
- legend text stating `depth=0 is outer edge of fused p95 structural uncertainty` and `no physical wall/site boundary used`.

Do not use the failed flood-fill site mask in this figure.

- [ ] **Step 5: Run CLI/core regression tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_headland_depth_profile.py \
  tests/test_headland_depth_profile_cli.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add tools/build_headland_depth_profile.py tests/test_headland_depth_profile_cli.py
git commit -m "feat: add finite headland depth profile CLI"
```

---

### Task 3: Frozen Observation Evidence by Depth

**Files:**
- Modify: `src/agt_map_reconstruction/maps/structural_endpoint_uncertainty_evidence.py`
- Create: `src/agt_map_reconstruction/maps/headland_depth_evidence.py`
- Create: `tools/evaluate_headland_depth_evidence.py`
- Test: `tests/test_headland_depth_evidence.py`
- Test: `tests/test_headland_depth_evidence_cli.py`

**Interfaces:**
- Expose `summarize_observation_sufficiency_roi(base, ground, scan, roi, *, min_repeated_scans, ray=None) -> dict` from the existing uncertainty evidence module; existing `evaluate_uncertainty_roi_observation_sufficiency` must continue to return unchanged fields.
- Produce `evaluate_headland_depth_evidence(base_map, ground_reference, scan_support_count, depth_profile_payload, depth_masks, *, min_repeated_scans=2, ray_support_count=None) -> dict`.

- [ ] **Step 1: Write a regression test for the public single-ROI observation summary**

Use the existing synthetic UNKNOWN/ground/scan/ray fixture and assert exact values for:

```text
trusted_ground_unknown_cell_count
ground_reference_ceiling_fraction_of_unknown
scan_observed_fraction_of_unknown
repeated_scan_fraction_of_unknown
ray_supported_fraction_of_unknown
```

- [ ] **Step 2: Run the regression and verify it fails because the public helper is absent**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_headland_depth_evidence.py
```

- [ ] **Step 3: Promote `_roi_stats` to a public reusable helper without changing current behavior**

Rename or wrap it as:

```python
def summarize_observation_sufficiency_roi(
    base,
    ground,
    scan,
    roi,
    *,
    min_repeated_scans,
    ray=None,
):
    ...
```

Update the existing uncertainty evaluator to call the public helper and run its existing tests.

- [ ] **Step 4: Implement depth evidence aggregation**

The depth evaluator must iterate every entry/exit depth-band mask listed in the geometry payload, plus boundary uncertainty and unresolved cross-strip, and return:

```python
{
    "method": "finite_headland_depth_observation_sufficiency",
    "entry": {"bands": [...]},
    "exit": {"bands": [...]},
    "boundary_uncertainty": {...},
    "structurally_unresolved_cross": {...},
    "policy": {
        "frozen_evidence_reused": True,
        "rosbag_replay_performed": False,
        "ray_evidence_regenerated": False,
        "navigation_map_modified": False,
        "semantic_promotion": False,
    },
}
```

Every band record must retain `depth_min_m` and `depth_max_m` so plotting never depends on parsing a mask name.

- [ ] **Step 5: Add a failing-then-passing CLI test**

The CLI must accept:

```text
--depth-profile headland_depth_profile.json
--ground-reference ground_reference.npy
--scan-support-count scan_free_support_count.npy
--ray-support-count ray_free_support_count.npy
--min-repeated-scans 2
--output <dir>
```

and write `headland_depth_evidence.json` plus `headland_depth_evidence.png`.

- [ ] **Step 6: Implement compact depth-response plotting**

For entry and exit separately, plot against band midpoint:

- ground-reference ceiling fraction of UNKNOWN;
- scan-observed fraction of UNKNOWN;
- repeated-scan fraction of UNKNOWN;
- ray-supported fraction of UNKNOWN.

Do not plot a navigation acceptance curve.

- [ ] **Step 7: Run Task 3 regression tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_headland_depth_evidence.py \
  tests/test_headland_depth_evidence_cli.py \
  tests/test_structural_endpoint_uncertainty_evidence.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  src/agt_map_reconstruction/maps/structural_endpoint_uncertainty_evidence.py \
  src/agt_map_reconstruction/maps/headland_depth_evidence.py \
  tools/evaluate_headland_depth_evidence.py \
  tests/test_headland_depth_evidence.py \
  tests/test_headland_depth_evidence_cli.py
git commit -m "feat: evaluate frozen evidence by headland depth"
```

---

### Task 4: Ground-Reference Gate Sensitivity by Depth

**Files:**
- Create: `src/agt_map_reconstruction/maps/headland_depth_ground_gate_sweep.py`
- Create: `tools/sweep_headland_depth_ground_gate.py`
- Test: `tests/test_headland_depth_ground_gate_sweep.py`
- Test: `tests/test_headland_depth_ground_gate_sweep_cli.py`

**Interfaces:**
- Consumes finite depth masks, canonical UNKNOWN mask, K8/K16 nearest-support-distance grid, and K8/K16 model-disagreement grid.
- Produces one sensitivity grid per finite depth band without choosing a threshold.

- [ ] **Step 1: Write failing core test for per-band gate sweeps**

Use two depth bands with known distance/disagreement values and assert, for each gate pair:

```python
assert record["accepted_unknown_cell_count"] == expected
assert np.isclose(record["accepted_unknown_fraction"], expected / unknown_count)
```

Also assert the result contains `automatic_threshold_selection=False` and rejects overlapping masks.

- [ ] **Step 2: Run the core test and verify failure**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_headland_depth_ground_gate_sweep.py
```

- [ ] **Step 3: Implement depth-specific sweep core**

Reuse the semantics of `structural_endpoint_uncertainty_ground_gate_sweep.py`: finite inputs only, exact gate values recorded, no refit, no selection. Do not import or depend on failed site-clipped masks.

- [ ] **Step 4: Add CLI test and implement CLI**

CLI arguments:

```text
--map
--depth-profile
--reference-a
--reference-b
--max-support-distance-m 0.25 0.50 1.00 2.00 4.00
--max-model-disagreement-m 0.05 0.10 0.20 0.50 1.00
--output
```

Output `headland_depth_ground_gate_sweep.json` and a compact plot showing accepted UNKNOWN fraction versus outward depth for each gate pair.

- [ ] **Step 5: Run Task 4 regression tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_headland_depth_ground_gate_sweep.py \
  tests/test_headland_depth_ground_gate_sweep_cli.py \
  tests/test_structural_endpoint_uncertainty_ground_gate_sweep.py \
  tests/test_structural_endpoint_uncertainty_ground_gate_sweep_cli.py
```

Expected: PASS and historical uncertainty-ROI sweep remains reproducible.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  src/agt_map_reconstruction/maps/headland_depth_ground_gate_sweep.py \
  tools/sweep_headland_depth_ground_gate.py \
  tests/test_headland_depth_ground_gate_sweep.py \
  tests/test_headland_depth_ground_gate_sweep_cli.py
git commit -m "feat: sweep ground confidence by headland depth"
```

---

### Task 5: Reference-Only Comparison with Historical Unbounded Diagnostic

**Files:**
- Create: `src/agt_map_reconstruction/maps/headland_depth_reference_comparison.py`
- Create: `tools/compare_headland_depth_with_unbounded.py`
- Test: `tests/test_headland_depth_reference_comparison.py`

**Interfaces:**
- Consumes new `headland_depth_evidence.json` and old `structural_endpoint_uncertainty_evidence.json`.
- Produces a reference-only comparison; it must never label the two spatial domains equivalent.

- [ ] **Step 1: Write failing test that enforces the comparison warning**

Assert the result contains:

```python
assert result["spatial_domains_equivalent"] is False
assert result["historical_unbounded_metrics_used_for_acceptance"] is False
assert result["finite_depth_profile_is_primary"] is True
```

- [ ] **Step 2: Implement comparison summary**

For entry and exit, report:

- historical unbounded UNKNOWN count;
- historical unbounded trusted-ground ceiling;
- finite 0–4 m aggregate UNKNOWN count;
- finite 0–4 m aggregate trusted-ground ceiling;
- finite per-band evidence curve retained verbatim.

Do not subtract the fractions and call the difference an improvement because the domains differ.

- [ ] **Step 3: Implement CLI and JSON output**

Write `headland_depth_vs_unbounded_reference.json`. A plot is optional; if added, visually label the unbounded value as `historical diagnostic reference` rather than a peer method.

- [ ] **Step 4: Run comparison tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_headland_depth_reference_comparison.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add \
  src/agt_map_reconstruction/maps/headland_depth_reference_comparison.py \
  tools/compare_headland_depth_with_unbounded.py \
  tests/test_headland_depth_reference_comparison.py
git commit -m "feat: compare finite depth with unbounded diagnostic"
```

---

### Task 6: Real Greenhouse Replay Using Frozen Assets

**Files:**
- No source changes in the measurement step.
- Outputs only under `results/P1/greenhouse_01_region_split/...`.

**Interfaces:**
- Consumes the frozen fusion, uncertainty, map, ground, ray, and scan assets already produced in this branch.
- Produces measured finite depth geometry/evidence/sensitivity/reference assets.

- [ ] **Step 1: Run the full new unit/CLI regression set**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_headland_depth_profile.py \
  tests/test_headland_depth_profile_cli.py \
  tests/test_headland_depth_evidence.py \
  tests/test_headland_depth_evidence_cli.py \
  tests/test_headland_depth_ground_gate_sweep.py \
  tests/test_headland_depth_ground_gate_sweep_cli.py \
  tests/test_headland_depth_reference_comparison.py \
  tests/test_structural_endpoint_uncertainty_evidence.py \
  tests/test_structural_endpoint_uncertainty_ground_gate_sweep.py
```

Do not claim PASS unless this command actually passes on the greenhouse workstation.

- [ ] **Step 2: Build finite depth geometry**

```bash
BASE=results/P1/greenhouse_01_region_split
TOPO=$BASE/topology
OBS=$BASE/observation
FUSION=$TOPO/structural_endpoint_fusion_d31
DEPTH=$TOPO/headland_depth_profile_v1

python tools/build_headland_depth_profile.py \
  --fused-structural-bundle "$FUSION/structural_endpoint_fused.json" \
  --fused-uncertainty "$FUSION/structural_endpoint_uncertainty_fused.json" \
  --depth-edges-m 0 0.5 1.0 2.0 4.0 \
  --uncertainty-quantile p95 \
  --output "$DEPTH"
```

Verify visually that no band extends beyond 4.0 m from its uncertainty-band outer edge and that `R_L18_L19` is excluded from every resolved band.

- [ ] **Step 3: Reuse frozen S5/P10 scan/ray evidence**

```bash
EVIDENCE=$OBS/ray_evidence_stream_full_s5_p10_scanv2
DEPTH_EVIDENCE=$OBS/headland_depth_evidence_s5p10_scan2

python tools/evaluate_headland_depth_evidence.py \
  --depth-profile "$DEPTH/headland_depth_profile.json" \
  --ground-reference "$OBS/ground_reference_consensus_d050_a010/ground_reference.npy" \
  --scan-support-count "$EVIDENCE/scan_free_support_count.npy" \
  --ray-support-count "$EVIDENCE/ray_free_support_count.npy" \
  --min-repeated-scans 2 \
  --output "$DEPTH_EVIDENCE"
```

Record the exact per-band fields for entry and exit: `unknown_cell_count`, `ground_reference_ceiling_fraction_of_unknown`, `scan_observed_fraction_of_unknown`, `repeated_scan_fraction_of_unknown`, and `ray_supported_fraction_of_unknown`.

- [ ] **Step 4: Run K8/K16 confidence-gate sensitivity by depth**

```bash
DEPTH_SWEEP=$OBS/headland_depth_ground_gate_sweep_k8_k16

python tools/sweep_headland_depth_ground_gate.py \
  --map "$BASE/navigation/navigation_base_map.pgm" \
  --depth-profile "$DEPTH/headland_depth_profile.json" \
  --reference-a "$OBS/local_ground_reference_k8" \
  --reference-b "$OBS/local_ground_reference_k16" \
  --max-support-distance-m 0.25 0.50 1.00 2.00 4.00 \
  --max-model-disagreement-m 0.05 0.10 0.20 0.50 1.00 \
  --output "$DEPTH_SWEEP"
```

Do not select a gate automatically. Inspect whether any low-distance/low-disagreement plateau persists as depth increases.

- [ ] **Step 5: Build reference-only comparison to the historical unbounded diagnostic**

```bash
python tools/compare_headland_depth_with_unbounded.py \
  --headland-depth-evidence "$DEPTH_EVIDENCE/headland_depth_evidence.json" \
  --unbounded-evidence "$OBS/structural_uncertainty_roi_evidence_s5p10_scan2/structural_endpoint_uncertainty_evidence.json" \
  --output "$OBS/headland_depth_vs_unbounded_reference"
```

Confirm the comparison JSON says the spatial domains are not equivalent and the finite depth profile is primary.

- [ ] **Step 6: Check result provenance before documentation**

Confirm every new JSON records the fused structural source, fused uncertainty source, exact depth edges, p95 quantile, unresolved ridge IDs, map resolution, and frozen evidence source paths. Confirm there is no site-flood-fill path in the new primary geometry/evidence artifacts.

---

### Task 7: Freeze the New Evaluation Interface in the Canonical Experiment Record

**Files:**
- Modify: `docs/EXPERIMENTS.md`

**Interfaces:**
- Consumes exact measured JSON/stdout from Task 6.
- Produces the only canonical written record of the finite-depth evaluation transition.

- [ ] **Step 1: Update the P1 status line**

State that D3.1 structural geometry is frozen, physical HARD-boundary site clipping was rejected by anchor/breach audit, and finite endpoint-relative depth evaluation is the active downstream interface.

- [ ] **Step 2: Record the failed physical-site experiment as a negative diagnostic**

Include the measured breach facts already established:

```text
17 observed-row anchors leaked to the map border
8 unique border exits
11 right / 6 left
path length 17.35–35.95 m
maximum path overlap 6/17 = 35.3%
```

State explicitly that no wall gap was automatically closed and the failed site-clipped metrics are not paper metrics.

- [ ] **Step 3: Record the finite-depth geometry contract**

Document p95 uncertainty-band outer edge as depth zero, exact diagnostic edges `[0, 0.5, 1, 2, 4] m`, row-lattice cross-span restriction, and unresolved `R_L18_L19` exclusion.

- [ ] **Step 4: Record the exact measured per-depth evidence response**

Copy the exact Task 6 JSON values for entry and exit for each band, including UNKNOWN count, trusted-ground ceiling, scan-observed fraction, repeated-scan fraction, and ray-supported fraction. Do not infer or round values that are not in the generated JSON.

- [ ] **Step 5: Record ground-gate sensitivity interpretation**

Summarize whether a low-uncertainty/high-coverage plateau exists by depth. If coverage rises only under large support-distance/model-disagreement gates, record that as evidence against further ground interpolation tuning rather than selecting a looser threshold.

- [ ] **Step 6: Record the historical unbounded comparison correctly**

Keep the old unbounded outward ROI as a historical diagnostic and explicitly state that its spatial domain differs from the finite 0–4 m profile; do not report a fraction difference as an algorithmic improvement.

- [ ] **Step 7: Verify documentation consistency and commit**

Search `docs/EXPERIMENTS.md` for stale statements such as `real trajectory-ray replay pending` and remove/update them using already measured E1/E2 results. Then commit:

```bash
git add docs/EXPERIMENTS.md
git commit -m "docs: freeze finite headland depth evaluation"
```

---

## Final Verification

- [ ] Run the complete feature regression set from Task 6 Step 1.
- [ ] Confirm the working tree contains no accidental generated `results/` files intended for Git unless the repository explicitly tracks them.
- [ ] Confirm old D3, unbounded ROI, flood-fill, and breach-audit outputs remain traceable and are not overwritten.
- [ ] Confirm new primary artifacts contain no physical-wall/site-interior dependency.
- [ ] Confirm no output sets `automatic_acceptance`, `navigation_map_modified`, or `semantic_promotion` to true.
- [ ] Confirm `docs/EXPERIMENTS.md` is the only experiment narrative updated.
