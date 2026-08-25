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

```
FAST-LIVO2 processed.pcd
```

Algorithms:

- height threshold
- morphological PMF

Conclusion:

Binary ground segmentation is only an intermediate representation.
Agricultural navigation requires structure recovery.

Output:

```
results/EXP001/
```

---

## EXP002 Agricultural Corridor Recovery

Status: completed; synthetic validation passed and authoritative real-PCD run recorded

Goal:

Recover interpretable agricultural navigation corridors from geometric structure.

Pipeline:

```
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

- component-wise PCA row direction estimation
- crop-row direction consistency
- removal of short, compact, or cross-row free-space components
- no learned semantic model

### EXP002-C Geometric corridor constraint

- rotate the map into the recovered crop-row frame
- detect lateral high-row bands and the valley between adjacent rows
- constrain corridor width, longitudinal length, and row-boundary continuity
- rotate accepted corridors back to the source grid

Canonical command:

```bash
python tools/run_corridor_test.py \
  --pcd /data/FAST-LIVO2/processed.pcd \
  --mode all \
  --hash-pcd
```

Each run is immutable. The default run ID is
`<UTC timestamp>_<short git commit>`.

The tree below is produced by the canonical `--mode all` run. A diagnostic
`--mode A`, `B`, or `C` run writes only the selected stage directory and a
single-panel `abc_comparison.png`; it must not be used as the A/B/C acceptance
run.

Output:

```text
results/EXP002/<run_id>/

├── metadata.yaml
├── metrics.yaml
├── height.png
├── relative_height.png
├── traversability.png
├── abc_comparison.png
├── A/
│   ├── corridor.png
│   ├── centerline.png
│   └── centerline.csv
├── B/
│   ├── corridor.png
│   ├── centerline.png
│   └── centerline.csv
└── C/
    ├── corridor.png
    ├── centerline.png
    └── centerline.csv
```

Traceability fields:

- repository and exact Git commit
- dirty-worktree flag
- absolute PCD path and byte size
- optional PCD SHA-256
- point count, UTC time, mode, parameters, row direction, and stage metrics
- grid origin, resolution, and shape; centerline CSV includes both cell and
  PCD-frame metric coordinates at cell centers

Validation evidence:

- synthetic horizontal-row corridor: passed
- component-wise PCA with a cross-row distractor: passed
- width/length/continuity constraints: passed
- rotated row-frame probe at ±15° and 30°: passed
- immutable artifact layout and streaming SHA-256: passed

Real-data status:

- authoritative input: FAST-LIVO2 LIO-only `processed.pcd`
- accepted run: `results/EXP002/20260824T092719Z_d3f7b0d/`
- input SHA-256:
  `c915cba7cb0e4cabd2d35d6e000d82c265f077f38d82121015b32a5f33f06787`
- input points: `85,912,613`
- recovered row angle: `0.3100158217960846` radians
- A corridor cells: `396,064`
- B corridor cells: `951`
- C corridor cells: `26,748`
- C accepted corridors: `4`

Observed failure cases and conclusion:

- A oversegments the global traversable mask and does not isolate useful
  agricultural corridors.
- B collapses the candidate set and removes nearly all corridor area.
- C recovers four plausible local corridors, but recall is insufficient for a
  useful global navigation layer.
- This result closes EXP002 and motivates EXP003: retain explicit measured,
  occupied, interpolated, and unknown evidence before extracting routes.

Residual limitations:

- no reference centerline is available, so centerline accuracy remains unevaluated
- the four C corridors are a local qualitative recovery, not sufficient global
  free-space recall

Decision: do not continue tuning the binary corridor mask. Proceed to EXP003's
conservative ground-evidence representation.

---

## EXP003 Conservative Ground Evidence Map

Status: code and synthetic verification complete; authoritative 85M-point run
and throughput profiling pending

Goal:

Build an explainable global grid that preserves the distinction between
measured free ground, confirmed obstacles, bounded ground interpolation, and
unsupported unknown space.

Evidence schema:

| Value | Label | Meaning |
|---:|---|---|
| `0` | `UNKNOWN` | Insufficient measurement support or an unsupported gap |
| `1` | `FREE_CONFIRMED` | Measured low surface consistent with the local ground model |
| `2` | `OCCUPIED_CONFIRMED` | Measured structure above the obstacle-height threshold |
| `3` | `GROUND_INTERPOLATED` | Bounded hole supported by nearby ground; not measured free space |

Navigation cost schema (`uint8`):

| Value | Meaning |
|---:|---|
| `0` | Confirmed free |
| `1`–`253` | Traversal penalty; default interpolated-ground cost is `64` |
| `254` | Confirmed or inflated obstacle |
| `255` | Unknown |

Canonical command:

```bash
python tools/run_ground_evidence_test.py \
  --pcd /data/FAST-LIVO2/processed.pcd \
  --output results/EXP003 \
  --hash-pcd \
  --resolution 0.05 \
  --chunk-size 1000000 \
  --low-quantile 0.10 \
  --histogram-bins 64 \
  --min-points-per-cell 3 \
  --min-ground-support-cells 2 \
  --ground-window-m 0.50 \
  --ground-percentile 20.0 \
  --ground-seed-percentile 10.0 \
  --max-ground-step-m 0.20 \
  --max-interpolation-gap-m 0.25 \
  --obstacle-height-m 0.15 \
  --obstacle-inflation-radius-m 0.25 \
  --interpolated-ground-cost 64
```

Omitting `--run-id` creates
`<UTC timestamp>_<short git commit>`. Publication is immutable: an existing
run directory is rejected rather than overwritten. `--hash-pcd` adds the
streamed input SHA-256 to `metadata.yaml`. PCD device, inode, byte size, and
nanosecond modification time are snapshotted before loading and verified again
before publication; a changed or replaced input aborts without an output run.

Output:

```text
results/EXP003/<run_id>/
├── metadata.yaml
├── metrics.yaml
├── low_height.npy
├── ground_surface.npy
├── clearance.npy
├── point_count.npy
├── evidence.npy
├── costmap.npy
├── low_height.png
├── ground_surface.png
├── clearance.png
├── evidence.png
└── costmap.png
```

The NPY arrays are the authoritative numeric artifacts. PNG files are
inspection aids. `metadata.yaml` records UTC creation time, repository, exact
Git commit and dirty flag, absolute input path, byte size, optional SHA-256,
input and finite point counts, full configuration, PCD-frame grid origin,
resolution, and grid shape. `metrics.yaml` records measured, free, occupied,
interpolated, unknown, and inflated cell counts.

Configuration:

| Option | Default | Purpose |
|---|---:|---|
| `--resolution` | `0.05` m | Grid cell size and metric basis for all windows/radii |
| `--chunk-size` | `1000000` | Points processed per rasterization chunk; avoids an index array proportional to the 85M-point input |
| `--low-quantile` | `0.10` | Lower cumulative height quantile selected per cell |
| `--histogram-bins` | `64` | Fixed bins used for the robust per-cell low-height estimate |
| `--min-points-per-cell` | `3` | Minimum support for a measured cell |
| `--min-ground-support-cells` | `2` | Minimum distinct neighboring propagated-ground cells, excluding self, required for an independently supported ground model |
| `--ground-window-m` | `0.50` m | NaN-aware local low-percentile ground-estimation window; the target cell is excluded from its own estimate |
| `--ground-percentile` | `20.0` | Local percentile used as the ground surface |
| `--ground-seed-percentile` | `10.0` | Global measured-height percentile that seeds the low-ground envelope; disconnected components do not seed from their own local minima |
| `--max-ground-step-m` | `0.20` m | Maximum adjacent-cell height step allowed while propagating ground support from low-envelope seeds |
| `--max-interpolation-gap-m` | `0.25` m | Maximum distance from measured support for separately labelled interpolation |
| `--obstacle-height-m` | `0.15` m | Clearance above ground at which a measured cell is occupied |
| `--obstacle-inflation-radius-m` | `0.25` m | Euclidean safety inflation radius |
| `--interpolated-ground-cost` | `64` | Nonzero cost assigned to bounded interpolated ground |

Implemented validation and acceptance evidence:

- robust low-height behavior, finite-point filtering, PCD-frame origin, and
  chunk-size invariance: passed
- sparse-noise rejection, low-envelope-seeded and step-constrained ground classification,
  component-local bounded interpolation, sparse/open/oversized-gap unknown
  preservation, obstacle preservation, and metric inflation: passed
- immutable artifact publication, complete metadata/metrics, exact artifact
  layout, stable pre-load input snapshot verification, argument validation,
  and duplicate-run rejection: passed
- synthetic PCD CLI end-to-end smoke: passed
- complete automated suite: passed (`101` tests)

Acceptance boundary and limitations:

- The code and synthetic behavior are accepted.
- The authoritative `85,912,613`-point PCD has not been run through EXP003;
  real-map evidence quality is therefore not accepted yet.
- Acceptance-scale runtime and peak-memory/throughput profiling are pending,
  including the NaN-aware percentile filter and grid-proportional interpolation
  workspace.
- Unknown remains unknown and interpolated ground remains separately penalized;
  neither may be reported as measured free space.

Next action: run the canonical EXP003 command on the authoritative PCD, retain
the complete immutable run directory, profile runtime and peak memory, inspect
the evidence and costmap in the same local regions recovered by EXP002-C, and
record the real-data decision here.

### EXP003 real-data diagnostic iterations

The diagnostic record is archived at
`docs/experiments/EXP003_real_data_diagnostic_2026-08-24.md`. The first two
implementation iterations are deliberately PCD-output-only:

1. `python tools/diagnose_exp003.py results/EXP003/<run_id>` writes robust
   percentiles, negative-clearance counts, evidence counts, an inflation-radius
   scan, and categorical PNG previews without rereading the PCD.
2. Elevation rasterization now retains Q10/Q50/Q90 arrays in memory and EXP003
   uses Q90 as the obstacle statistic, making isolated low-quantile outliers
   less likely to become occupied evidence. The original low-height and
   artifact schemas remain unchanged.

These iterations are implementation baselines, not real-map acceptance claims;
ROI labels and row-direction metrics remain follow-up work.

The executed PMF, EXP003, and MK-mini envelope comparison is recorded in
[`docs/experiments/EXP003_pmf_envelope_comparison_2026-08-24.md`](experiments/EXP003_pmf_envelope_comparison_2026-08-24.md).

### MK-mini offline envelope v0

The fixed PCD preview may use the bare MK-mini body envelope without waiting for
the mounted `base_footprint` transform:

```text
length = 0.840 m
width = 0.600 m
reference = geometric center
payload = excluded
```

`VehicleEnvelopeConfig` and `build_vehicle_navigation_layers` apply this
rectangle to a row-aligned evidence map. Unknown cells remain non-traversable;
the result is an offline navigation candidate, not a vehicle-ready route asset.
The 1.5 m turning-radius constraint remains a separate connector-planning gate.

The first PCD-only row/channel extraction is recorded in
[`docs/experiments/EXP003_row_structure_channel_analysis_2026-08-24.md`](experiments/EXP003_row_structure_channel_analysis_2026-08-24.md).

---

## EXP004 Rosbag Raycasting Feasibility Probe

Status: deferred feasibility probe only; not implemented and not a baseline

If EXP003 real-data inspection shows that endpoint-only evidence is
insufficient, a later probe may replay a short rosbag segment using
time-aligned raw LiDAR frames and FAST-LIVO2 poses for raycasting and log-odds
fusion. It must compare the same local regions against EXP003 and demonstrate
useful gains without pose-error obstacle carving before any baseline is
proposed. No EXP004 implementation or acceptance claim exists today.
