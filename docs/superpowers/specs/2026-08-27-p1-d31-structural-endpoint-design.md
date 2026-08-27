# P1-D3.1 Structural Endpoint Design

## Purpose

P1-D3 currently evaluates endpoint-side evidence outside a common line fitted from recovered row-aisle centerline endpoints. D3.1 introduces a separate **structural row termination** definition for the physical end of greenhouse rows, so that geometric aisle extent, clearance-conditioned handoff, and headland/turn-area semantics are no longer conflated.

The existing P1-D3 result remains a historical geometric-endpoint baseline. D3.1 does not rewrite old experiment outputs and does not silently reinterpret prior metrics.

## Problem Statement

Three endpoint concepts are currently distinct:

1. **Geometric aisle endpoint** — the longitudinal end of the recovered aisle rectangle / centerline.
2. **Clearance handoff** — the end of the currently connected robot-clearance-safe aisle core. This may move inward because of UNKNOWN, HARD obstacles, or safe-component fragmentation.
3. **Structural row termination** — the physical end of the row/ridge structure. This is the boundary that should define a headland / turn-area candidate.

The D3.1 geometry audit showed that most clearance handoffs are close to geometric aisle endpoints, while exit rows A03, A12, and A14 move inward by approximately 4.6–5.8 m. These are safe-core truncations and must not be treated as structural row ends.

## Selected Approach

Use **bilateral structural-support termination detection** around each eligible aisle, followed by robust common-boundary fitting.

For every clearance-width-eligible row aisle:

- Use the frozen row direction from existing row geometry.
- Build two narrow structural strips immediately outside the aisle polygon, one on each side of the aisle.
- Sample structural support along the row axis from the existing frozen navigation evidence / structural occupancy mask.
- Detect the longitudinal interval where left and right structural support are persistently present.
- Define entry and exit structural endpoints from the persistent bilateral support termination, not from free-space reachability.
- If only one side has reliable structure, retain a per-row endpoint candidate but mark it low-confidence / ambiguous rather than hallucinating the missing side.
- Fit common entry and exit structural boundaries robustly across rows so that one genuinely shorter/longer row cannot pull the common boundary several metres.

This design intentionally separates **structure** from **traversability**. Clearance handoffs remain navigation assets; structural endpoints become topology/headland assets.

## Inputs

D3.1 must reuse frozen upstream assets whenever possible:

- `navigation_base_map.pgm` — canonical P1 navigation map, read-only.
- `row_band_regions.json` — recovered row-aisle polygons and centerlines.
- `aisle_handoffs.json` — used only for eligibility/provenance comparison, not as the structural endpoint source.
- Existing frozen grid metadata / resolution.
- Existing structural / occupied evidence already produced by P1; if a dedicated structural mask is required, it must be derived deterministically from an existing frozen P1 asset and recorded in provenance.

No rosbag replay and no ray reconstruction is required for D3.1 geometry recovery.

## Structural Support Profile

For each row aisle, define a local coordinate frame:

- `u`: row-axis longitudinal coordinate.
- `v`: cross-row coordinate.

Construct one structural strip on each side of the aisle polygon. The strip width is an explicit parameter recorded in the output and subjected to sensitivity analysis; it is not automatically optimized.

For each longitudinal bin in `u`, compute at minimum:

- left structural-support fraction,
- right structural-support fraction,
- bilateral support flag,
- direct UNKNOWN fraction,
- direct HARD / structural fraction.

A support profile must preserve missing evidence as missing/uncertain. UNKNOWN cannot be counted as structural support and cannot be converted to free space.

## Endpoint Detection

A row structural endpoint is the termination of a persistent structural-support run, not a one-cell threshold crossing.

The detector must:

- require a minimum longitudinal persistence window;
- tolerate short internal holes without extending through long unsupported gaps;
- expose the selected run and all thresholds in the output;
- report entry and exit independently;
- never replace an ambiguous endpoint with a clearance handoff or raw aisle endpoint automatically.

Per-row endpoint status must be one of:

- `ok_bilateral`
- `ambiguous_single_side`
- `insufficient_structural_support`

No semantic promotion is allowed for ambiguous rows.

## Robust Common Boundary

The common entry/exit boundary is fitted only from per-row structural endpoint candidates with acceptable status.

The fit must be robust to isolated row-length outliers. The output must retain:

- every per-row endpoint,
- residual to the common boundary,
- inlier/outlier classification,
- fit residual summary,
- exact fitting method and thresholds.

Outliers are not deleted from the artifact. They remain visible and traceable.

The robust fit is descriptive only. There is no automatic acceptance of a headland region.

## Relationship to Existing D3 and Handoffs

D3.1 must emit all three endpoint locations side-by-side:

- raw geometric aisle endpoint,
- clearance-conditioned handoff,
- structural endpoint.

This comparison is required because the three concepts answer different questions:

- geometric endpoint: where recovered aisle geometry ends;
- handoff: where the robot can currently remain inside a connected clearance-safe aisle core;
- structural endpoint: where the agricultural row structure physically terminates.

Existing P1-D3 is preserved as the geometric-endpoint baseline and may be described as historical once D3.1 is validated.

## GUI / Manual Confirmation Policy

Manual GUI confirmation is allowed only for rows reported as ambiguous by the automatic structural detector.

There must be one canonical structural endpoint artifact. A manual confirmation updates or annotates the ambiguous row in that artifact with provenance; it must not create a parallel competing semantic map or a second headland definition.

Automatic results and manual overrides must be distinguishable in provenance fields.

## Outputs

Canonical D3.1 outputs live under one experiment directory, for example:

`results/P1/greenhouse_01_region_split/topology/structural_endpoint_d31/`

Required assets:

- `structural_endpoint_profiles.json`
- `structural_endpoint_boundary.json`
- `structural_endpoint_context.png`
- `entry_structural_endpoint_context.png`
- `exit_structural_endpoint_context.png`

The boundary JSON must contain source paths, parameters, row-axis definition, per-row endpoint records, robust fit records, and policy flags.

## Required Visual Context

The context figure must show the full navigation map, not a black-background ROI crop only.

It must distinguish:

- recovered row aisles,
- geometric row endpoints,
- clearance handoffs,
- structural endpoints,
- common structural entry/exit fits,
- ambiguous/outlier rows.

This figure is a geometry audit asset first; publication styling comes later.

## Validation Tests

Before implementation is considered valid, automated tests must cover these behaviors:

1. **Aisle extends beyond ridge:** if free/corridor geometry continues after both side structures end, the structural endpoint must stop at the ridge termination rather than the aisle/map edge.
2. **UNKNOWN truncates clearance core:** if the clearance handoff moves inward by several metres because UNKNOWN breaks the safe component, the structural endpoint must remain at the structural termination.
3. **Single structural-length outlier:** one short/long row must remain visible as an outlier without pulling the robust common boundary substantially.
4. **One side missing:** a row with only one reliable side structure must be marked `ambiguous_single_side`; no bilateral endpoint may be fabricated.
5. **No structural support:** rows without sufficient structure must be marked `insufficient_structural_support`.
6. **Orientation normalization:** reversed source centerlines must produce the same common entry/exit semantics.
7. **No map mutation:** the canonical PGM and prior D3 outputs remain unchanged.

## Reuse of E0/E1/E2 Evidence

After D3.1 geometry is validated, existing ground-reference grids, ray-support grids, unique-scan support grids, and observation-sufficiency assets may be re-evaluated in the D3.1 ROI.

They must not be regenerated solely because the endpoint geometry changed. Any re-evaluation must use the same frozen evidence arrays and record that only the ROI / endpoint geometry changed.

## Experiment Record

`docs/EXPERIMENTS.md` remains the only canonical experiment record.

When D3.1 is replay-validated, the record must state that:

- prior D3 used raw geometric aisle endpoints;
- the D3.1 audit separated geometric endpoints, clearance handoffs, and structural endpoints;
- E0/E1/E2 evidence grids were reused rather than regenerated;
- old D3 results remain traceable and are not silently overwritten.

## Non-Goals

D3.1 does not:

- classify UNKNOWN as FREE;
- alter `navigation_base_map.pgm`;
- define a drivable headland automatically;
- choose rescan targets automatically;
- optimize structural thresholds automatically;
- replace runtime local perception;
- use clearance handoff as a fallback structural endpoint.

## Freeze Boundary

D3.1 becomes the new topology/headland geometry authority only after:

- all required unit tests pass;
- the real greenhouse replay produces interpretable per-row endpoints;
- the full-map context figure confirms that detected structural endpoints correspond to row termination rather than walls or safe-core truncation;
- ambiguous rows are explicitly surfaced rather than automatically repaired.

Until then, existing D3 remains historical/frozen and D3.1 is experimental.