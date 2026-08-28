# P1 Finite Headland Depth Profile Design

## Purpose

Replace the failed physical-site clipping approach with an **endpoint-relative finite headland depth profile** for P1 evidence evaluation.

The new representation answers a narrower and better-supported question:

> As distance increases outward from the structurally recovered row termination, how quickly do trusted-ground, ray, and repeated-scan evidence decay?

It deliberately does **not** try to infer the entire physical greenhouse boundary from the canonical 2D occupancy map.

## Why the Physical-Site Flood-Fill Approach Is Rejected

The canonical navigation PGM does not encode a closed physical greenhouse boundary strongly enough for HARD-boundary border flood fill to define site interior.

Measured breach diagnostics showed:

- 17 observed row-lattice midpoint anchors were evaluated;
- all 17 anchors were exterior-reachable through non-HARD cells;
- border exits occurred at 8 distinct map-border locations;
- 11 leaked to the right map border and 6 to the left;
- leaked-anchor path length ranged from 17.35 m to 35.95 m;
- the strongest common path bottleneck was shared by only 6/17 anchors (35.3%).

Therefore the failure cannot be reduced to one obvious doorway or one isolated wall gap. The 2D HARD layer should not be promoted into a physical greenhouse boundary model.

The previous site-flood-fill and site-clipped ROI results remain as failed diagnostics and are not reused as paper metrics.

## Inputs and Frozen Dependencies

The finite depth profile must reuse the already validated/frozen structural geometry and observation assets:

- fused structural ridge endpoint artifact;
- fused structural endpoint uncertainty artifact;
- row-lattice cross-row span / slot geometry;
- unresolved structural ridge list, currently including `R_L18_L19`;
- canonical navigation map only for UNKNOWN/FREE/OCCUPIED class lookup, never as a physical outer boundary;
- frozen ground-reference grids;
- frozen ray-support grids;
- frozen unique-scan support grids.

No rosbag replay, new raycasting, new structural fitting, or navigation-map mutation is required.

## Coordinate Definition

Use the frozen structural row frame:

- `u`: longitudinal row-axis coordinate;
- `v`: cross-row coordinate.

For entry and exit independently, use the fused structural center trend and its frozen uncertainty summary.

The finite outward distance origin is the **outer edge of the structural uncertainty band**, not the center trend itself:

- entry depth zero begins at `entry_center_trend - entry_p95` on the outward side;
- exit depth zero begins at `exit_center_trend + exit_p95` on the outward side.

This keeps the uncertain structural termination band separate from the outward headland-evidence evaluation.

The center trend is descriptive geometry only and is never promoted to semantic free space.

## Default Diagnostic Depth Bands

The first implementation uses explicit finite bands:

- `0.0–0.5 m`
- `0.5–1.0 m`
- `1.0–2.0 m`
- `2.0–4.0 m`

These bands are **diagnostic bins**, not automatically optimized paper parameters.

The implementation must support arbitrary monotonically increasing user-specified band edges so sensitivity studies can be run without changing code.

No automatic depth-band selection is allowed.

## Cross-Row Domain

Depth bands are evaluated only across the frozen structural row-lattice cross-row domain.

The cross-row domain must come from existing structural/lattice geometry rather than map borders or inferred greenhouse walls.

Cells associated with structurally unresolved ridge cross-spans, including `R_L18_L19`, must be excluded from all resolved headland depth bands and reported separately.

The unresolved strip must never be filled by interpolation from neighboring structural trends.

## Region Partition

For each side, the evaluation partition contains:

1. `boundary_uncertainty` — the fused structural `trend ± p95` band; retained as a separate diagnostic reference region.
2. `depth_0_0p5`
3. `depth_0p5_1`
4. `depth_1_2`
5. `depth_2_4`
6. `structurally_unresolved_cross` — reported separately and excluded from all resolved depth bands.

All produced masks must be mutually exclusive.

No mask may extend beyond the explicitly configured maximum finite depth merely because map cells remain UNKNOWN.

## Evidence Metrics Per Depth Band

For every entry/exit depth band, report at minimum:

- total ROI cells;
- UNKNOWN cells;
- UNKNOWN fraction of ROI;
- trusted-ground UNKNOWN cells;
- trusted-ground ceiling fraction of UNKNOWN;
- UNKNOWN with trusted ground but no scan observation;
- UNKNOWN with single/non-repeated scan support;
- UNKNOWN with repeated scan support;
- scan-observed fraction of trusted-ground UNKNOWN;
- scan-observed fraction of all UNKNOWN;
- repeated-scan fraction of trusted-ground UNKNOWN;
- repeated-scan fraction of all UNKNOWN;
- ray-supported UNKNOWN cells;
- ray-supported fraction of trusted-ground UNKNOWN;
- ray-supported fraction of all UNKNOWN.

The trusted-ground ceiling is an **eligibility ceiling only**. It is not semantic free space, navigation acceptance, or evidence that the robot can turn there.

## Ground-Reference Sensitivity Per Depth Band

The existing K8/K16 nearest-support-distance and model-disagreement grids should be reusable in the new finite depth bands.

The sweep must report, for each depth band and each gate pair:

- accepted UNKNOWN cell count;
- accepted UNKNOWN fraction;
- exact support-distance threshold;
- exact model-disagreement threshold.

No threshold may be automatically selected.

The intended paper-level question is whether there exists a low-uncertainty/high-coverage plateau as outward depth increases. If coverage only rises under large support distance or model disagreement, that should be reported rather than hidden by tuning.

## Interpretation Rules

The method must preserve the distinction between:

- **structure available** — the row termination geometry is supported;
- **ground eligible** — local ground reference is sufficiently supported under a stated gate;
- **observed** — at least one trajectory-aware scan/ray supports the cell;
- **repeatedly observed** — at least the stated number of independent physical scans support the cell;
- **navigation free** — explicitly out of scope for this layer.

A depth band with strong structure but weak ground/observation evidence remains uncertain. It must not be promoted to free.

## Expected Paper Outputs

The primary publication-facing summary should be a depth-response curve rather than a single global headland mask.

For entry and exit separately, plot against outward distance:

- trusted-ground ceiling fraction;
- scan-observed fraction;
- repeated-scan fraction;
- ray-supported fraction.

Optional secondary plots may show:

- ground-gate sensitivity by depth;
- UNKNOWN fraction by depth;
- entry/exit asymmetry.

The visualization should make clear that depth is measured from the outer edge of the structural uncertainty band.

## Outputs

Use a dedicated output directory, for example:

`results/P1/greenhouse_01_region_split/topology/headland_depth_profile_v1/`

Required geometry assets:

- `headland_depth_profile.json`
- `headland_depth_profile.png`
- one boolean `.npy` mask per entry/exit depth band;
- `entry_boundary_uncertainty_mask.npy`;
- `exit_boundary_uncertainty_mask.npy`;
- `structurally_unresolved_cross_mask.npy`.

Observation evaluation should write to the existing observation-results hierarchy, for example:

`results/P1/greenhouse_01_region_split/observation/headland_depth_evidence_s5p10_scan2/`

Required evidence assets:

- `headland_depth_evidence.json`
- `headland_depth_evidence.png` or a compact depth-profile plot.

Ground-gate sensitivity should remain a separate JSON/plot asset so raw evidence evaluation and gate sensitivity are not conflated.

## Provenance

Every depth-profile artifact must record:

- fused structural endpoint source path;
- fused uncertainty source path;
- row-lattice source path;
- unresolved ridge IDs;
- row axis and cross-row axis;
- uncertainty quantile used as the structural-band boundary;
- depth-band edges;
- map resolution;
- evidence grid source paths when evaluating ground/ray/scan support;
- policy flags.

The previous unbounded outward ROI and failed flood-fill/clipped ROI remain traceable but are not silently overwritten.

## Validation Tests

Before real replay, automated tests must cover:

1. **Finite extent:** depth bands stop exactly at the requested maximum depth and never extend to the map border automatically.
2. **Correct outward direction:** entry and exit bands extend in opposite longitudinal directions from the uncertainty-band outer edge.
3. **Mutual exclusivity:** all entry/exit depth masks, uncertainty masks, and unresolved strip are pairwise disjoint where required.
4. **Unresolved exclusion:** a structurally unresolved cross-span is excluded from every resolved depth band.
5. **No map mutation:** input navigation map and frozen structural assets are unchanged.
6. **Metric partition:** UNKNOWN evidence categories sum consistently and trusted-ground ceilings use the correct denominator.
7. **Depth ordering:** user-specified non-monotonic or duplicate depth edges are rejected.
8. **No physical-wall dependency:** the geometry builder works without a site-interior mask or closed HARD boundary.
9. **Orientation invariance:** equivalent reversed source centerline orientation yields the same physical entry/exit depth semantics after normalization.
10. **Evidence reuse:** the evaluator reads existing frozen evidence arrays and records that no replay/raycast regeneration occurred.

## Relationship to D3.1

D3.1 structural geometry remains frozen and authoritative for row termination.

The finite headland depth profile is a **downstream evaluation interface**, not a replacement structural detector.

No lattice, PGM/3D fusion rule, ridge endpoint, p95 structural uncertainty, or unresolved-ridge classification is changed by this work.

## Relationship to Failed Site Flood Fill

The HARD-boundary flood-fill experiment remains valuable as a negative result:

- topology-only flood fill falsely identified small enclosed pockets as site interior;
- trusted row-lattice anchors showed that the main greenhouse body remained connected to the map border;
- outer-boundary breach audit showed dispersed exits rather than one repairable bottleneck.

Therefore physical-site clipping is removed from the main evaluation chain rather than patched with automatic wall closure.

No morphology, wall repair, doorway classification, or inferred physical greenhouse polygon is introduced by the finite-depth method.

## Non-Goals

This work does not:

- infer the physical greenhouse wall polygon;
- repair HARD boundary gaps;
- classify doorways;
- create navigation free space;
- define a globally drivable headland;
- modify the canonical PGM;
- regenerate rosbag/ray evidence;
- change fused structural ridge endpoints;
- automatically choose depth bands or confidence thresholds;
- use UNKNOWN as positive structure or free-space evidence.

## Freeze Boundary

The finite headland depth profile becomes the canonical downstream D3.1 evaluation interface only after:

- all unit and CLI tests pass;
- the real greenhouse visualization shows finite bands beginning at the correct structural uncertainty edges;
- no band leaks indefinitely toward the map border;
- unresolved ridge cross-spans remain visibly excluded;
- frozen evidence replay produces interpretable entry/exit depth-response statistics;
- ground-gate sensitivity is reported without automatic threshold selection.

After these checks, `docs/EXPERIMENTS.md` should record:

- D3.1 structural geometry remains frozen;
- physical-site flood fill was rejected for the canonical PGM;
- finite endpoint-relative depth bands replace global site clipping as the headland evidence-evaluation interface;
- E0/E1/E2 evidence grids were reused rather than regenerated.
