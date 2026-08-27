# Experiment Records

This file is the single entry point for all experiment tracking.

Each experiment records:

- dataset
- algorithm version
- parameters
- output path
- metrics
- failure cases
- conclusion

---

## EXP001 Ground Segmentation Benchmark

Status: completed

Dataset:

```text
FAST-LIVO2 processed.pcd
```

Algorithms:

- height threshold
- morphological PMF

Conclusion:

Binary ground segmentation is only an intermediate representation.
Agricultural navigation requires structure recovery.

Output:

```text
results/EXP001/
```

---

## EXP002 Agricultural Corridor Recovery

Status: completed

Goal:

Recover interpretable agricultural navigation corridors from geometric structure.

Pipeline:

```text
relative elevation
        |
        v
row direction estimation
        |
        v
row-aware corridor extraction
        |
        v
centerline generation
```

Algorithm stages:

### EXP002-A Baseline

- PCA dominant direction estimation
- connected free-space extraction
- centerline export

### EXP002-B Row-aware constraint

- dominant row direction consistency
- geometric filtering of inconsistent regions
- no learned semantic model

Output:

```text
results/EXP002/

├── metadata.yaml
├── height.png
├── relative_height.png
├── traversability.png
├── corridor.png
├── centerline.png
└── centerline.csv
```

Conclusion:

Row-aware corridor recovery provides the geometric prior needed for a navigation-oriented map, but corridor masks alone do not answer whether a robot-sized footprint can traverse every aisle.

Handoff:

- EXP003 owns map-server interface validation and robot-equivalent clearance checks.

---

## EXP003 Navigation Map V2 Interface Validation

Status: completed, superseded by EXP003.1 static obstacle policy correction

Goal:

Convert recovered semantic geometry into a navigation-oriented static-map bundle and quantify aisle connectivity.

Original policy:

```text
aisle prior              -> free
ridge / wall             -> hard occupied
obstacle / step / pillar -> candidate layer
```

The original replay produced 20 / 20 aisles at 0.30 m equivalent clearance, but review of the generated PGM showed that pillar cells inside aisle polygons could be promoted to free. That result is therefore retained only as a pre-correction baseline.

---

## EXP003.1 Static Obstacle Policy Correction

Status: implemented and replay-validated on 2026-08-25

Goal:

Prevent permanent greenhouse structure from being erased by the recovered aisle prior while avoiding a return to overly conservative candidate blocking.

Inputs:

```text
semantic_labels.npy
aisle_rectangles.json
```

Reference dataset:

- grid: 912 x 797
- resolution: 0.05 m/cell
- aisles: 20
- pillar cells: 2181

Corrected semantic priority:

```text
unknown default
      |
      v
aisle prior / semantic aisle -> free
      |
      v
ridge / wall / pillar       -> hard occupied (highest priority)

obstacle_candidate / step_candidate -> advisory candidate_mask only
```

The all-candidate-as-unknown policy was tested and rejected for this dataset because it reduced 0.30 m aisle connectivity to 12 / 20. Pillar-only structural blocking preserves substantially more corridor continuity while fixing the observed safety error.

Static-map grayscale contract:

```text
0   = occupied
205 = unknown
254 = free
```

Nav2 YAML contract:

```text
mode: trinary
occupied_thresh: 0.65
free_thresh: 0.196
```

Implementation:

```text
src/agt_map_reconstruction/maps/navigation_export.py
tools/build_navigation_map.py
tests/test_navigation_export.py
```

Default output:

```text
results/EXP003/navigation-map-v2/

├── navigation_base_map.pgm
├── navigation_base_map.yaml
├── candidate_mask.npy
├── static_obstacle_mask.npy
└── validation.json
```

Reference replay command:

```bash
python tools/build_navigation_map.py \
  --semantic-labels /path/to/semantic_labels.npy \
  --aisles /path/to/aisle_rectangles.json \
  --output results/EXP003/navigation-map-v2 \
  --resolution 0.05
```

Reference replay metrics after the correction:

| clearance radius | equivalent diameter | aisle connectivity |
| ---: | ---: | ---: |
| 0.20 m | 0.40 m | 20 / 20 |
| 0.25 m | 0.50 m | 20 / 20 |
| 0.30 m | 0.60 m | 17 / 20 |
| 0.35 m | 0.70 m | 14 / 20 |
| 0.40 m | 0.80 m | 13 / 20 |
| 0.50 m | 1.00 m | 12 / 20 |

Additional replay metrics:

- canonical gray values: `[0, 205, 254]`
- map-server YAML validation: pass
- pillar cells: 2181
- pillar cells exported as free: 0
- static obstacle semantic validation: pass
- advisory candidate cells: 5829

Failure cases at 0.30 m radius:

- A01
- A15
- A18

Interpretation:

These failures are more informative than the original 20 / 20 result because the corrected map no longer erases structural pillar geometry. They are not yet proof that the physical robot cannot traverse the aisles; circular clearance is only a conservative proxy for the real polygon footprint and vehicle pose.

Conclusion:

EXP003.1 fixes the static-map semantic bug that allowed pillars to become white/free. The corrected map is suitable as the input to robot-scale validation. Obstacle and step candidates remain outside the permanent hard-obstacle layer so local perception can still resolve uncertain geometry at runtime.

Handoff:

- EXP004 owns polygon-footprint and route-level validation.

---

## EXP004 Robot Footprint & Route Validation

### EXP004-A Polygon Footprint Centerline Baseline

Status: implementation complete; replay-validated with measured `mk_mini` footprint on 2026-08-25

Goal:

Replace circular-equivalent clearance with an explicit polygon footprint and determine whether the recovered aisle centerline itself is collision-free for that footprint.

Inputs:

```text
results/EXP003/navigation-map-v2/navigation_base_map.pgm
results/EXP003/navigation-map-v2/navigation_base_map.yaml
results/EXP003/navigation-map-v2/candidate_mask.npy
aisle_rectangles.json
robot_footprint.json
```

Measured replay footprint:

```json
{
  "name": "mk_mini",
  "polygon_xy_m": [
    [0.42, 0.30],
    [0.42, -0.30],
    [-0.42, -0.30],
    [-0.42, 0.30]
  ]
}
```

This is a 0.84 m x 0.60 m base-link-centred footprint. Coordinates are metres in the robot `base_link` frame with `+x` forward and `+y` left.

Validation policy:

```text
occupied            -> fail sampled pose
unknown (default)    -> fail sampled pose
candidate_mask       -> report overlap only
out of map           -> fail sampled pose
```

`--allow-unknown` exists only for diagnostic comparison. EXP004-A samples the footprint strictly along the recovered aisle centerline; it does not search for a laterally shifted collision-free route.

Implementation:

```text
src/agt_map_reconstruction/maps/footprint_validation.py
tools/validate_robot_footprint.py
tests/test_footprint_validation.py
```

Output:

```text
results/EXP004/robot-footprint-v1/
├── aisle_footprint_validation.json
└── aisle_footprint_validation.csv
```

Per-aisle metrics:

- sampled pose count
- occupied collision pose count
- unknown overlap pose count
- advisory candidate overlap pose count
- out-of-bounds pose count
- minimum clearance to the active blocking policy
- first failure reason and first failure pose

Test status:

```text
pytest -q tests/test_footprint_validation.py
6 passed
```

Measured-footprint replay:

- resolution: 0.05 m
- sample spacing: 0.10 m
- `allow_unknown: false`
- centreline pass: 1 / 20
- passing aisle: `A14`
- failing aisles: `A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 A11 A12 A13 A15 A16 A17 A18 A19 A20`

Important cases:

- `A03`, `A08`, and `A12` have zero hard collisions and fail only on unknown overlap.
- `A04`, `A07`, `A10`, `A19`, and `A20` have only 1-3 hard-collision poses on the strict centreline.
- `A14` passes but its minimum blocked-space clearance is only 0.05 m, so it is a geometric pass rather than a robust navigation acceptance result.

Conclusion:

EXP004-A establishes a strict physical-footprint centreline baseline. The measured `mk_mini` result confirms that centreline validity is much stricter than EXP003 corridor connectivity and motivates a route-level lateral search rather than further static-map editing.

---

### EXP004-B1 Constant Lateral Offset Search

Status: implemented, tested, and replay-validated on 2026-08-25

Goal:

Determine whether each aisle contains a collision-free route that stays parallel to the recovered row direction but shifts laterally away from pillars, wall geometry, or unknown boundaries. This stage intentionally avoids A*, Hybrid A*, Nav2, and variable-curvature planning.

Search model:

```text
recovered aisle centreline
          |
          +-- offset -N * step
          +-- ...
          +-- offset 0
          +-- ...
          `-- offset +N * step
                    |
                    v
      polygon footprint sweep
                    |
                    v
     select best passing offset
```

The theoretical offset bounds are computed from each recovered aisle width and the measured footprint lateral extent. Search defaults:

```text
sample_spacing_m = 0.10
offset_step_m    = 0.05
allow_unknown    = false
```

Validation policy is unchanged from EXP004-A:

```text
occupied       -> block
unknown        -> block by default
candidate_mask -> advisory only
out of map     -> block
```

Route selection:

1. every sampled pose must pass;
2. maximize the 10th-percentile blocked-space clearance (`clearance_p10_m`);
3. then maximize minimum clearance;
4. then minimize advisory candidate overlap;
5. then prefer the smaller absolute offset.

`clearance_p10_m` is used before strict minimum clearance because the strict minimum is frequently dominated by the one-cell unknown boundary at aisle entry/exit and is not discriminative between lateral routes.

If no offset passes, `best_attempt_*` records the route with the fewest blocking poses so failures remain quantitatively useful.

Implementation:

```text
src/agt_map_reconstruction/maps/in_aisle_route_search.py
tools/search_in_aisle_offsets.py
tests/test_in_aisle_route_search.py
```

Output:

```text
results/EXP004/in-aisle-route-search-v1/
├── aisle_offset_search.json
├── aisle_offset_search.csv
├── route_overlay.png
└── A05_route_overlay.png
```

Test status:

```text
pytest -q tests/test_in_aisle_route_search.py
6 passed
```

The EXP004-B evaluator was cross-checked at `offset=0` against the measured-footprint EXP004-A output. For all 20 aisles, pose count, occupied collision count, unknown-overlap count, and PASS/FAIL matched exactly.

Reference replay with measured `mk_mini` footprint:

```text
centreline pass:      1 / 20
offset-route pass:    8 / 20
recovered routes:     7
strict failures:     12
```

Recovered by constant lateral offset:

| aisle | best offset | interpretation |
| --- | ---: | --- |
| A04 | +0.05 m | small centreline correction |
| A05 | -0.30 m | large but feasible constant shift |
| A08 | -0.05 m | removes unknown/edge conflict |
| A11 | +0.30 m | large constant shift |
| A12 | +0.05 m | removes unknown/edge conflict |
| A16 | +0.20 m | avoids centreline structural collision |
| A19 | -0.10 m | removes sparse collision |

`A14` remains the only centreline route and is also accepted by EXP004-B1.

Strictly failing after constant-offset search:

```text
A01 A02 A03 A06 A07 A09 A10 A13 A15 A17 A18 A20
```

Useful failure diagnostics:

- `A07`: best attempt `-0.05 m`, only 1 remaining blocking/collision pose.
- `A20`: best attempt `-0.20 m`, only 1 remaining blocking/collision pose.
- `A03`: no hard collision on centreline; strict failure is caused by unknown overlap. Diagnostic `--allow-unknown` increases the total EXP004-B1 pass count from 8 / 20 to 9 / 20, but unknown remains blocking for acceptance.
- `A05`: centreline fails, but a full constant route passes at `-0.30 m`; this validates the purpose of EXP004-B1 and provides the first detailed in-aisle route case.

Conclusion:

Constant lateral search recovers 7 additional measured-footprint routes without modifying the static map. This demonstrates that a substantial portion of the EXP004-A failures were centreline-placement failures rather than aisle-unreachable failures. The remaining 12 aisles should not be forced into PASS by changing the PGM; they require either a spatially varying lateral route or source-geometry review.

---

### EXP004-B2 Smooth Lateral Route

Status: implemented, tested, and replay-validated on 2026-08-25

Goal:

Allow lateral offset to vary gradually along the aisle and determine whether failures that cannot be solved by one constant offset are true in-aisle blockers or are primarily row-entry / row-exit handoff effects.

Search model:

```text
longitudinal control stations (0.50 m default)
                  |
                  v
lateral offset lattice (0.05 m default)
                  |
                  v
bounded offset change between stations
                  |
                  v
continuous swept polygon transition check
                  |
                  v
dynamic-programming route selection
                  |
                  v
0.10 m sampled final route validation
```

Default strict parameters:

```text
sample_spacing_m    = 0.10
control_spacing_m   = 0.50
offset_step_m       = 0.05
max_offset_change_m = 0.10
endpoint_trim_m     = 0.00
allow_unknown       = false
```

The dynamic-programming score rewards blocked-space clearance and penalizes lateral displacement, offset change, curvature proxy, and advisory candidate overlap. The transition checker uses the continuously swept convex polygon footprint so an obstacle cannot be skipped between 0.10 m samples.

No Ackermann steering-angle or minimum-turning-radius constraint is applied in B2. `max_offset_change_m` limits geometric smoothness only; vehicle kinematics remain a later stage.

Implementation:

```text
src/agt_map_reconstruction/maps/smooth_lateral_route.py
tools/search_smooth_lateral_routes.py
tests/test_smooth_lateral_route.py
```

Output:

```text
results/EXP004/smooth-lateral-route-v1/
├── smooth_route_search.json
├── smooth_route_search.csv
├── smooth_route_overlay.png
└── <focus>_smooth_route_overlay.png
```

Overlay legend:

```text
orange -> recovered centreline
green  -> B2 smooth route PASS
red    -> B1 best failed constant route retained for comparison
```

Test status:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_smooth_lateral_route.py
7 passed
```

The plugin-autoload guard is intentional for ROS 2 sourced shells because `launch_testing` can otherwise mix ROS/system Python packages into this pure offline pytest suite.

#### Strict full-length replay

Measured `mk_mini` footprint, `endpoint_trim_m=0.00`, unknown blocking:

```text
B1 baseline pass:      8 / 20
B2 strict pass:        9 / 20
new B2 recoveries:     1
newly recovered aisle: A06
strict failures:      11
```

Strict failure-region classification:

```text
entry:    A09 A10 A18
interior: A03
exit:     A01 A02 A07 A13 A15 A17 A20
```

The main result is therefore **not** that a more complex smooth route solves most remaining aisles. Under strict full-length validation, B2 only adds `A06` beyond B1.

`A07` and `A20`, despite having only one blocking pose in B1, remain B2 failures. Their B1 collision occurs at route distance `29.98 m` in a `30.40 m` aisle, leaving `0.42 m` to the recovered aisle end. This equals the `mk_mini` forward footprint extent (`+0.42 m`), showing that these two failures occur at the terminal footprint pose rather than in the aisle interior.

#### Handoff-boundary sensitivity diagnostic

`endpoint_trim_m` shortens the evaluated in-aisle segment at both ends. It is a diagnostic parameter and must not be reported as strict full-length acceptance.

With `endpoint_trim_m=0.05 m`:

- `A07` still fails at exit;
- `A20` still fails at exit.

With `endpoint_trim_m=0.10 m`, unknown still blocking:

```text
B1 baseline pass:  8 / 20
B2 diagnostic:    17 / 20
recovered:         9
remaining:         A01 A03 A10
```

Recovered relative to B1 under the 0.10 m handoff trim:

```text
A02 A06 A07 A09 A13 A15 A17 A18 A20
```

If `endpoint_trim_m=0.10 m` and `--allow-unknown` are both enabled, the diagnostic result becomes 18 / 20 and additionally recovers `A03`; only `A01` and `A10` remain blocked. This is not an acceptance configuration because unknown is intentionally blocking in the strict policy.

Interpretation:

1. B2 proves that `A06` is a genuine variable-lateral-route recovery under the unchanged strict map policy.
2. Many other B1 failures are dominated by row entry/exit geometry rather than by in-aisle obstacle avoidance.
3. `A07` and `A20` were initially expected to be ideal B2 cases, but the replay disproved that hypothesis: their single blocker is at the exit handoff boundary.
4. `A03` remains the clearest unknown-space policy case.
5. The correct next step is therefore to stop increasing in-aisle planner complexity and explicitly validate the row/headland handoff and vehicle kinematics.

Conclusion:

EXP004-B2 improves the strict measured-footprint result only from 8 / 20 to 9 / 20. The 0.10 m endpoint-trim sensitivity jump to 17 / 20 is stronger evidence that the next bottleneck is the recovered aisle endpoint / headland interface, not the static PGM interior. No map cells were edited to obtain either result.

Next stage: EXP004-C Headland Handoff & Ackermann Transition

1. define explicit in-aisle entry/exit handoff poses rather than forcing the footprint to the recovered rectangle endpoint;
2. keep strict occupied/unknown semantics and measured `mk_mini` footprint;
3. validate `A05 -> headland -> A06` first because A05 is a B1 success and A06 is the first strict B2 recovery;
4. use measured wheelbase / steering geometry from robot configuration for minimum-turning-radius constraints rather than inventing values;
5. separate failure reasons into map collision, unknown overlap, insufficient headland depth, and kinematic infeasibility;
6. only after this offline handoff test is stable, move to Nav2/runtime integration.

---

## P1 Evidence-driven Greenhouse Semantic Reconstruction

Status: P1-A through P1-D replay-validated and frozen on 2026-08-27. P1-E0 observation-ray interface and conservative support accumulator are implemented; real trajectory-ray replay is pending.

Goal:

Reconstruct a conservative, interpretable greenhouse navigation asset from LIO-only map evidence while preserving the distinction between observed free, hard occupied, unknown, row structure, transition geometry, and observation insufficiency.

Reference grid and row axis:

- grid: 912 x 797
- resolution: 0.05 m/cell
- row direction: `[0.9626245859468401, 0.27083926327376306]`
- row angle: approximately 15.72 deg

### P1-A Evidence and row/open-area separation

Measured reconstruction chain:

```text
q90 obstacle regression -> 0 aisles
low-envelope evidence   -> 8 aisles with 90-degree axis ambiguity
occupied-banding resolver -> 20 raw row-aligned bands
width-distribution split -> 17 row aisles + 3 wide row-aligned open areas
```

Evidence counts remained invariant through the downstream rebuild:

```text
unknown:              264863
free_confirmed:       270877
occupied_confirmed:   191117
ground_interpolated:       7
```

The width-outlier threshold is `Q3 + 1.5*IQR = 1.2875 m`. The three wide bands remain `wide_open_area_candidate`; width alone is not headland evidence.

After the candidate aisle-conflict diagnostic policy:

```text
aisle_conflict_candidates: 31331
hard occupied cells:        159786
static free cells:          302208
unknown cells:              264870
hard-as-free cells:              0
```

Measured 17-row clearance:

| clearance radius | equivalent diameter | row-aisle connectivity |
| ---: | ---: | ---: |
| 0.20 m | 0.40 m | 12 / 17 |
| 0.25 m | 0.50 m | 10 / 17 |
| 0.30 m | 0.60 m | 8 / 17 |
| 0.35 m | 0.70 m | 6 / 17 |
| 0.40 m | 0.80 m | 2 / 17 |
| 0.50 m | 1.00 m | 0 / 17 |

Geometry diagnosis:

```text
minimum_width_limited:        A01 A07
minimum_connectivity_limited: A03 A12 A14
unexpected_connectivity:      A03 A10 A12 A14
wide_width_outliers:          none after semantic split
```

P1-A conclusion:

The 20 raw bands must not be used as the aisle denominator. Separating wide row-aligned open bands prevents them from receiving aisle-conditioned obstacle relaxation and preserves a 17-row semantic denominator.

### P1-B Local blocker localization

Measured first unexpected failures:

| aisle | radius | region | causal source | mode | blocker evidence |
| --- | ---: | --- | --- | --- | --- |
| A03 | 0.20 m | exit | unknown | longitudinal_gap | first blocker s/L 0.807-0.817; longest 0.65 m |
| A10 | 0.25 m | exit | unknown | exit_probe_blocked | exit probe unsafe; no full longitudinal blocked cross-section |
| A12 | 0.20 m | exit | unknown | longitudinal_gap | first blocker s/L 0.828-0.838; longest 0.30 m |
| A14 | 0.20 m | exit | unknown | exit_probe_blocked | first blocker s/L 0.849-0.864; longest 0.45 m |

All four unexpected failures localize to the exit side and have causal source `unknown`. No row-interior hard blocker was identified in this set.

### P1-C Clearance-conditioned row core and handoff

At `radius=0.20 m`:

```text
aisles: 17
status ok: 17
no_safe_component: 0
largest_span_fallback: 1
width_clearance_eligible: 15
width_limited: 2 (A01, A07)
```

Causal replay at each aisle's first unexpected radius:

| aisle | radius | width eligible | core fraction | exit transition | causal source | causal mode | exit context | cause/context agree |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| A03 | 0.20 m | yes | 0.798 | 5.85 m | unknown | longitudinal_gap | unknown | yes |
| A10 | 0.25 m | yes | 0.903 | 2.51 m | unknown | exit_probe_blocked | hard | no |
| A12 | 0.20 m | yes | 0.820 | 5.19 m | unknown | longitudinal_gap | hard | no |
| A14 | 0.20 m | yes | 0.840 | 4.60 m | unknown | exit_probe_blocked | unknown | yes |

Interpretation rule:

- causal blocker source comes from the localized connectivity failure;
- transition context source describes the wider transition-zone background;
- `status == ok` only means a safe component exists and is not aisle acceptance;
- a context dominated by hard boundaries does not overwrite an `unknown` causal failure.

P1-C conclusion:

A03/A10/A12/A14 are not width failures at their causal radii. Each retains roughly 80-90% of a safe row core and fails at the exit transition. Static-map editing and additional in-aisle planner complexity are therefore not the next causal fix.

### P1-D1 Global open-area connectivity diagnostic

Using the 15 width-eligible rows at `radius=0.20 m` produced 30 entry/exit handoffs:

```text
strict_connected:    2
unknown_bridge_only: 0
disconnected:       28
```

Only A17 intersects O01 under strict global connected-component reachability; O02/O03 have no connections. This diagnostic has `connectivity_scope=global_component` and must not be interpreted as side-local headland connectivity.

### P1-D2 Geometric audit of O01/O02/O03

Measured geometry:

```text
row_axis_direction: [0.9626245859468401, 0.2708392632737631]
row_cross_span: [-47.0, 514.5]

O01: row_axis_alignment=1.000 cross_row_overlap=0.000 entry_outward=0.000 exit_outward=0.001
O02: row_axis_alignment=1.000 cross_row_overlap=0.000 entry_outward=0.000 exit_outward=0.001
O03: row_axis_alignment=1.000 cross_row_overlap=0.000 entry_outward=0.000 exit_outward=0.002
```

P1-D2 conclusion:

All three regions are row-parallel wide open bands, not cross-row endpoint areas. The hypothesis `wide row-aligned band == headland` is rejected for this dataset. The regions remain `wide_open_area_candidate` and are excluded from headland promotion.

### P1-D3 Endpoint-side evidence envelope

The endpoint envelope is derived from the 15 width-eligible rows rather than from O01/O02/O03.

Measured `radius=0.20 m` evidence:

| metric | entry | exit |
| --- | ---: | ---: |
| strict cross-row coverage | 0.019 | 0.017 |
| strict endpoint median distance | 7.487 m | 7.469 m |
| strict max outward depth | 0.191 m | 0.153 m |
| relaxed cross-row coverage | 1.000 | 1.000 |
| relaxed endpoint median distance | 0.375 m | 0.363 m |
| relaxed max outward depth | 10.205 m | 8.370 m |
| relaxed unknown fraction | 0.998 | 0.997 |
| coverage gain | 0.981 | 0.983 |
| endpoint distance reduction | 7.113 m | 7.106 m |
| outward depth gain | 10.013 m | 8.217 m |
| relaxed observed fraction | 0.002 | 0.003 |

The full JSON records more precise unknown fractions of approximately 0.99799 at entry and 0.99664 at exit.

Interpretation:

`relaxed_unknown_allowed` is a non-hard connectivity upper bound, not observed free space. Coverage rises from about 2% to about 100% only because the relaxed envelope is more than 99.6% unknown. Therefore:

```text
not hard != observed free != navigable
```

P1-D3 conclusion:

The common endpoint-side geometry is plausible, but the LIO-only PCD does not contain enough observed-free evidence to recover a trustworthy headland. The correct result is `HEADLAND NOT RECOVERED: insufficient observation evidence`, not semantic promotion of unknown space.

### P1-E0 Trajectory-aware observation evidence

Status: implementation complete; real rosbag/trajectory-ray replay pending.

Motivation:

An aggregate PCD keeps return coordinates but loses the sensor origin of each return. P1-E restores line-of-sight provenance instead of filling unknown space morphologically.

Frozen ray bundle contract:

```text
observation_rays.npz
├── schema_version = 1
├── frame_id
├── ray_origin_xyz_m      (N,3)
├── ray_endpoint_xyz_m    (N,3)
├── timestamp_s           (optional N)
└── scan_index            (optional N)
```

Implementation:

```text
src/agt_map_reconstruction/maps/observation_ray_bundle.py
src/agt_map_reconstruction/maps/ground_reference_plane.py
src/agt_map_reconstruction/maps/ground_aware_ray_evidence.py
tools/fit_ground_reference_plane.py
tools/build_ground_aware_ray_evidence.py
docs/interfaces/observation_ray_bundle.md
```

Important separation:

- the P1-A semantic `ground_surface.npy` is NaN in unknown cells by design;
- P1-E therefore fits an explicit affine geometry-only `ground_reference.npy` from finite ground-model support;
- affine residual RMSE / median / p95 / max are reported before the reference is used;
- extrapolated ground-reference values are not free-space evidence;
- a 3D ray supports a cell only when its in-cell segment lies in an explicitly configured low-height band above the ground reference;
- the return/hit cell is never marked free;
- output is `ray_free_support_count.npy` and `ray_free_support_mask.npy` only;
- `semantic_promotion=false` remains mandatory for P1-E0.

No paper parameter is frozen yet for the low-height band or minimum ray support count. Those values are explicit CLI inputs and require real-ray sensitivity analysis.

P1-E acceptance question:

Does trajectory-aware ray evidence materially increase strict observed-free endpoint coverage and reduce endpoint distance relative to the frozen P1-D3 PCD-only baseline without promoting unsupported high-canopy rays or hard obstacles?

### P1 freeze boundary

P1-A through P1-D are frozen. Do not change the PGM, candidate policy, row classifier, aisle denominator, or handoff definitions to improve headland appearance. New progress must come from additional observation provenance or new measured data, then be evaluated against the same D3 endpoint metrics.

Primary output roots:

```text
results/P1/greenhouse_01_region_split/
├── navigation/
├── diagnostics/
├── handoffs/
└── topology/
    ├── r020/
    ├── headland_geometry_audit/
    ├── endpoint_envelope_r020/
    └── endpoint_envelope_r020_evidence_gap/
```
