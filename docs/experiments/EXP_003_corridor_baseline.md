# EXP_003 Corridor Extraction Baseline

## Goal

Move from ground segmentation to navigation-oriented agricultural corridor recovery.

## Input

FAST-LIVO2 LIO-only PCD map.

## Pipeline

PCD

-> height grid

-> traversability grid

-> connected region filtering

-> corridor mask

-> skeleton centerline

## Current implementation

- Simple geometric corridor extraction
- Small region removal
- Optional skeletonization

## Known limitations

- Does not yet use row direction
- Does not distinguish ridge from vegetation
- No centerline smoothing

## Next experiment

Add agricultural row-aware extraction:

- orientation estimation
- parallel row detection
- gap filling
- waypoint export
