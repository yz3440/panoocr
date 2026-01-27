# Deduplication

Removes duplicate text detections when the same text appears in multiple perspective views.

## Algorithm

OCR results are deduplicated using spatial overlap and text similarity:

1. For each result pair, check both text similarity and region overlap
2. If texts are similar (Levenshtein) or overlapping, and regions intersect sufficiently, mark as duplicate
3. Keep the result with longer text, or higher confidence if equal length

## Adaptive Processing Strategy

PanoOCR automatically selects the optimal deduplication strategy based on perspective arrangement:

- **Sequential pairwise (fast)**: Used when perspectives form a simple horizontal ring (same pitch, sorted by yaw). Compares only adjacent perspective pairs plus wrap-around.
- **Incremental master list (thorough)**: Used for arbitrary perspective arrangements (multiple pitch levels, custom arrangements). Compares each result against all previous results.

This is handled automatically - no configuration needed.

## SphereOCRDuplicationDetectionEngine

::: panoocr.dedup.detection.SphereOCRDuplicationDetectionEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - check_duplication
        - remove_duplication_for_two_lists
        - deduplicate_frames

## Data Classes

::: panoocr.dedup.detection.RegionIntersection
    options:
      show_root_heading: true
