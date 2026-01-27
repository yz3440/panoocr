# Deduplication

Removes duplicate text detections when the same text appears in multiple perspective views.

## Algorithm

OCR results are deduplicated using spatial overlap and text similarity:

1. Process perspectives sequentially (adjacent pairs)
2. For each result pair, check both text similarity and region overlap
3. If texts are similar (Levenshtein) or overlapping, and regions intersect sufficiently, mark as duplicate
4. Keep the result with longer text, or higher confidence if equal length

## SphereOCRDuplicationDetectionEngine

::: panoocr.dedup.detection.SphereOCRDuplicationDetectionEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - check_duplication
        - remove_duplication_for_two_lists

## Data Classes

::: panoocr.dedup.detection.RegionIntersection
    options:
      show_root_heading: true
