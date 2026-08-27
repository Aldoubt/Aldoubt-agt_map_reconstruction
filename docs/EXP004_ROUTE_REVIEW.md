# EXP004 route visual review

This review records manual inspection of the route overlay projected onto the
processed PCD. It is a human-review layer and does not silently replace the
strict footprint validator.

| Aisles | Manual result | Cause / action |
| --- | --- | --- |
| A01 | blocked | Confirmed obstruction |
| A02 | pass | Map debris false positive |
| A03 | pass | False ridge near upper end |
| A04, A05, A08, A11, A12, A14 | pass | Visually clear |
| A06 | blocked | Pillar present |
| A07, A09, A10, A20 | pass | Rectangle extends outside wall |
| A13, A15 | pass | Vegetation falsely projected to ground |
| A16 | pass with warning | Pillar visible, requires local review |
| A17 | pass | Local step/pillar false positive; cleared only near the affected stations |
| A18 | pass | Local pillar false positive; cleared only near the affected stations |

The reviewed-v2 correction keeps hard obstacles globally. It clears only the
explicitly reviewed local regions for A17 and A18, then reruns the strict
measured-footprint smooth-route search. Both aisles pass after this correction.

The resulting strict route result is **17 / 20**. Remaining failures are A01
(manual confirmed obstruction), A06 (manual confirmed pillar), and A03
(no feasible transition at interior station 7). A03 is not force-marked as
pass: its remaining failure is consistent with the robot footprint touching
occupied cells at the narrow aisle boundary and needs a geometry/PCD review.

Artifacts:

```text
results/EXP004/navigation-map-reviewed-v2/
results/EXP004/smooth-lateral-route-reviewed-v2/
results/EXP004/pcd-route-review-reviewed-v2/
```

The next correction should focus on A03 aisle width/boundary geometry. A06
remains hard-blocked by the pillar and A01 remains blocked.

The reviewed-v3 export also clips ridge end caps to the interior greenhouse
scene boundary. This prevents overlong ridge polygons from occupying the
headland/turning area. A11 remains passable in the strict route replay. The
diesel-generator-side narrow gap is intentionally not auto-cleared yet: the
engine is retained as a hard obstacle and the gap should be drawn as an
explicit reviewed corridor after its exact width is measured in the PCD.

Both the interactive 3D viewer and the top-view PNG now show the map origin
and X/Y axes. The map origin is `(-4.5383229, -28.9657173, 0.0)` and the map
resolution is `0.05 m/cell`.
