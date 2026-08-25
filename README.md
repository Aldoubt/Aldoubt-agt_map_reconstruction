# agt_map_reconstruction

Agricultural LiDAR map reconstruction benchmark.

## Goal

Convert LIO-only PCD maps (FAST-LIVO2 / FAST-LIO2 outputs) into navigation-oriented 2D grid maps while comparing different ground segmentation algorithms.

Current scope:

- offline PCD processing
- ground segmentation benchmark
- elevation map generation
- traversability grid generation
- agricultural corridor recovery
- navigation-oriented static map export and validation
- polygon-footprint aisle validation
- in-aisle constant lateral-offset route search
- smooth lateral-route geometry validation
- visualization comparison reports

The repository remains independent from the navigation runtime. It produces and validates map assets before they are integrated into planners or Nav2.

## Pipeline

```text
PCD
 |
 +-- preprocessing
 |
 +-- ground segmentation
 |       |- height threshold
 |       |- PMF
 |       |- CSF
 |       |- Patchwork adapter
 |
 +-- elevation / relative elevation
 |
 +-- traversability grid
 |
 +-- row-aware corridor recovery
 |
 +-- semantic geometry
 |       |- aisle prior
 |       |- ridge / wall / pillar static obstacles
 |       `- obstacle / step candidates
 |
 +-- navigation map v2
 |       |- navigation_base_map.pgm
 |       |- navigation_base_map.yaml
 |       |- candidate_mask.npy
 |       |- static_obstacle_mask.npy
 |       `- validation.json
 |
 +-- EXP004-A footprint validation
 |       |- aisle_footprint_validation.json
 |       `- aisle_footprint_validation.csv
 |
 +-- EXP004-B1 constant-offset search
 |       |- aisle_offset_search.json
 |       |- aisle_offset_search.csv
 |       |- route_overlay.png
 |       `- A05_route_overlay.png
 |
 `-- EXP004-B2 smooth lateral route
         |- smooth_route_search.json
         |- smooth_route_search.csv
         |- smooth_route_overlay.png
         |- A05_smooth_route_overlay.png
         |- A07_smooth_route_overlay.png
         `- A20_smooth_route_overlay.png
```

## Navigation map v2

`navigation_base_map` uses an explicit semantic priority for navigation safety:

- aisle geometry is used as a free-space prior;
- ridge, wall, and pillar labels are static hard obstacles and override the aisle prior;
- obstacle and step candidates remain an advisory candidate layer instead of being permanently burned into the static map;
- the exported PGM uses only `0 / 205 / 254` for occupied / unknown / free;
- the YAML uses Nav2 trinary thresholds with `free_thresh < occupied_thresh`;
- `validation.json` reports aisle connectivity at multiple robot-equivalent clearance radii and verifies that no static obstacle is exported as free.

Build a bundle from the semantic reconstruction outputs:

```bash
python tools/build_navigation_map.py \
  --semantic-labels /path/to/semantic_labels.npy \
  --aisles /path/to/aisle_rectangles.json \
  --output results/EXP003/navigation-map-v2 \
  --resolution 0.05
```

Default clearance checks use radii:

```text
0.20 0.25 0.30 0.35 0.40 0.50 m
```

These checks validate static-map corridor connectivity. They do not replace vehicle-footprint, local-costmap, localization, headland-turning, or full Nav2 runtime tests.

## EXP004-A polygon footprint validation

EXP004-A replaces the circular-equivalent clearance proxy with an explicit robot polygon. It is intentionally a **strict aisle-centerline baseline**: the footprint is aligned with each recovered aisle and sampled along that centerline. It does not search for a laterally shifted path around pillars.

Footprint JSON uses metres in the robot `base_link` frame (`+x` forward, `+y` left):

```json
{
  "name": "my_robot",
  "polygon_xy_m": [
    [0.50, 0.30],
    [0.50, -0.30],
    [-0.50, -0.30],
    [-0.50, 0.30]
  ]
}
```

Replace the example coordinates with the measured physical footprint before treating the result as a robot acceptance test.

Run:

```bash
python tools/validate_robot_footprint.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/robot-footprint-v1 \
  --sample-spacing 0.10
```

Policy:

- occupied cells always fail a sampled pose;
- unknown cells fail by default; use `--allow-unknown` only for diagnostic comparison;
- candidate-mask overlap is reported but remains advisory;
- each aisle reports sampled pose count, occupied/unknown/candidate overlap counts, first failure reason and minimum clearance to the active blocking policy.

Outputs:

```text
results/EXP004/robot-footprint-v1/
├── aisle_footprint_validation.json
└── aisle_footprint_validation.csv
```

A centerline failure does not automatically mean the aisle is physically unreachable. EXP004-B tests that distinction explicitly.

## EXP004-B1 constant lateral-offset route search

EXP004-B1 keeps the aisle heading fixed and tests multiple constant cross-track offsets instead of forcing the robot to drive exactly on the recovered centerline. It is deliberately simpler than A*, Hybrid A*, or Nav2: the goal is to determine whether a straight, parallel route exists before introducing a full planner.

The search range is computed from the measured robot polygon and each recovered aisle width. With a centred 0.60 m-wide footprint in a 1.35 m aisle, for example, the theoretical centre offset range is approximately `[-0.375, +0.375] m`. The default search samples that range every `0.05 m` and always includes the geometric limits and zero when feasible.

Run:

```bash
python tools/search_in_aisle_offsets.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/in-aisle-route-search-v1 \
  --sample-spacing 0.10 \
  --offset-step 0.05 \
  --focus-aisle A05
```

Search policy:

- occupied cells always block a route;
- unknown cells block by default; `--allow-unknown` remains diagnostic only;
- candidate-mask overlap is advisory and used only as a late tie-breaker;
- a route must pass every sampled pose;
- among passing offsets, selection first maximizes the 10th-percentile blocked-space clearance (`clearance_p10_m`), then minimum clearance, then candidate overlap and absolute offset;
- if no route passes, `best_attempt_*` reports the offset with the fewest blocking poses for diagnosis.

Outputs:

```text
results/EXP004/in-aisle-route-search-v1/
├── aisle_offset_search.json
├── aisle_offset_search.csv
├── route_overlay.png
└── A05_route_overlay.png
```

`route_overlay.png` shows the recovered centreline and the selected constant-offset route (or the best failed attempt). The focus image crops the selected aisle for closer inspection.

## EXP004-B2 smooth lateral route

EXP004-B2 lets lateral offset change gradually along the aisle. It uses a longitudinal control lattice and dynamic programming over lateral-offset states. Each transition is validated by the continuous swept polygon footprint before the selected path is sampled at the requested resolution.

Defaults:

```text
sample_spacing_m     = 0.10
control_spacing_m    = 0.50
offset_step_m        = 0.05
max_offset_change_m  = 0.10
endpoint_trim_m      = 0.00
allow_unknown        = false
```

The route cost rewards clearance and penalizes lateral displacement, offset change, curvature proxy, and advisory candidate overlap. This is still an offline geometry validator: no Ackermann steering or minimum-turning-radius constraint is applied yet.

Run the strict full-length replay and compare it with B1:

```bash
python tools/search_smooth_lateral_routes.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles results/unified_navigation_comparison_20260824_v4/navigation_map_semantic_v2/aisle_rectangles.json \
  --footprint robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --baseline-b1 results/EXP004/in-aisle-route-search-v1/aisle_offset_search.json \
  --output results/EXP004/smooth-lateral-route-v1 \
  --sample-spacing 0.10 \
  --control-spacing 0.50 \
  --offset-step 0.05 \
  --max-offset-change 0.10 \
  --endpoint-trim 0.0 \
  --focus-aisles A05 A06 A07 A20
```

`failure_region` separates unresolved geometry into `entry`, `interior`, `exit`, or `aisle_geometry`. This is important because many remaining B1 failures occur exactly at aisle entry/exit and belong to the future headland handoff test rather than to in-aisle obstacle avoidance.

`--endpoint-trim` is diagnostic only. For example, `--endpoint-trim 0.10` stops the in-aisle validation 0.10 m earlier at each end and measures sensitivity to the row/headland handoff boundary. A trimmed result must not be reported as strict full-length acceptance.

Outputs:

```text
results/EXP004/smooth-lateral-route-v1/
├── smooth_route_search.json
├── smooth_route_search.csv
├── smooth_route_overlay.png
└── <focus>_smooth_route_overlay.png
```

Overlay legend:

- orange: recovered centreline;
- green: B2 smooth route PASS;
- red: B1 best failed constant-offset route retained for comparison.

## Test environment

This repository is pure offline Python for the EXP003/EXP004 tests. In a shell where ROS 2 Humble is sourced, ROS `launch_testing` may be auto-loaded by pytest. To keep these unit tests isolated, run:

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Using `python -m pytest` also ensures pytest uses the active virtual environment instead of a user/system executable.

## Experiment tracking

`docs/EXPERIMENTS.md` is the single experiment record and points to generated output paths. Generated map assets stay under `results/` and are not treated as source code.

## Design principle

This repository evaluates whether recovered agricultural geometry is interpretable and usable as a navigation-map input. Runtime obstacle perception, localization, planners, controllers, and behavior trees belong outside this repository.
