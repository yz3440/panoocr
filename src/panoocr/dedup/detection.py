"""Duplication detection for spherical OCR results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import Polygon
import textdistance

from ..ocr.models import SphereOCRResult


# Default thresholds for duplication detection
DEFAULT_MIN_TEXT_SIMILARITY = 0.5
DEFAULT_MIN_INTERSECTION_RATIO_FOR_SIMILAR_TEXT = 0.5
DEFAULT_MIN_TEXT_OVERLAP = 0.5
DEFAULT_MIN_INTERSECTION_RATIO_FOR_OVERLAPPING_TEXT = 0.15
DEFAULT_MIN_INTERSECTION_RATIO = 0.1


@dataclass
class RegionIntersection:
    """Result of intersecting two regions.

    Attributes:
        region_1_area: Area of the first region.
        region_2_area: Area of the second region.
        intersection_area: Area of the intersection.
        intersection_ratio: Intersection area divided by minimum region area.
    """

    region_1_area: float
    region_2_area: float
    intersection_area: float
    intersection_ratio: float


class SphereOCRDuplicationDetectionEngine:
    """Engine for detecting and removing duplicate OCR results.

    Duplicates are identified by comparing both spatial overlap and text
    similarity between results from overlapping perspective views.

    Attributes:
        min_text_similarity: Minimum Levenshtein similarity threshold.
        min_intersection_ratio_for_similar_text: Required overlap for similar texts.
        min_text_overlap: Minimum overlap similarity threshold.
        min_intersection_ratio_for_overlapping_text: Required overlap for overlapping texts.
        min_intersection_ratio: Minimum intersection ratio to consider.
    """

    def __init__(
        self,
        min_text_similarity: float = DEFAULT_MIN_TEXT_SIMILARITY,
        min_intersection_ratio_for_similar_text: float = DEFAULT_MIN_INTERSECTION_RATIO_FOR_SIMILAR_TEXT,
        min_text_overlap: float = DEFAULT_MIN_TEXT_OVERLAP,
        min_intersection_ratio_for_overlapping_text: float = DEFAULT_MIN_INTERSECTION_RATIO_FOR_OVERLAPPING_TEXT,
        min_intersection_ratio: float = DEFAULT_MIN_INTERSECTION_RATIO,
    ):
        """Initialize the deduplication engine.

        Args:
            min_text_similarity: Minimum Levenshtein normalized similarity (0-1).
            min_intersection_ratio_for_similar_text: Minimum region overlap for
                text matches based on similarity.
            min_text_overlap: Minimum overlap normalized similarity (0-1).
            min_intersection_ratio_for_overlapping_text: Minimum region overlap
                for text matches based on overlap.
            min_intersection_ratio: Absolute minimum intersection ratio to
                consider any match.
        """
        self.min_text_similarity = min_text_similarity
        self.min_intersection_ratio_for_similar_text = (
            min_intersection_ratio_for_similar_text
        )
        self.min_text_overlap = min_text_overlap
        self.min_intersection_ratio_for_overlapping_text = (
            min_intersection_ratio_for_overlapping_text
        )
        self.min_intersection_ratio = min_intersection_ratio

    def _has_valid_coordinates(self, sphere_ocr: SphereOCRResult) -> bool:
        """Check if a sphere OCR result has valid (finite) coordinates.

        Args:
            sphere_ocr: The OCR result to validate.

        Returns:
            True if all coordinates are finite, False if any are NaN or Inf.
        """
        values = [sphere_ocr.yaw, sphere_ocr.pitch, sphere_ocr.width, sphere_ocr.height]
        return all(math.isfinite(v) for v in values)

    def _sphere_ocr_to_polygon(self, sphere_ocr: SphereOCRResult) -> Optional[Polygon]:
        """Convert a sphere OCR result to a polygon in yaw/pitch space.

        Args:
            sphere_ocr: The OCR result to convert.

        Returns:
            A Polygon in yaw/pitch space, or None if coordinates are invalid.
        """
        if not self._has_valid_coordinates(sphere_ocr):
            return None

        half_width = sphere_ocr.width / 2
        half_height = sphere_ocr.height / 2

        left = sphere_ocr.yaw - half_width
        right = sphere_ocr.yaw + half_width
        top = sphere_ocr.pitch + half_height
        bottom = sphere_ocr.pitch - half_height

        points = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]

        return Polygon(points)

    def _polygon_to_gdf(self, polygon: Polygon) -> gpd.GeoDataFrame:
        """Convert a polygon to a GeoDataFrame in projected CRS."""
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon])
        gdf.to_crs(3857, inplace=True)
        return gdf

    def _sphere_ocr_to_gdf(
        self, sphere_ocr: SphereOCRResult
    ) -> Optional[gpd.GeoDataFrame]:
        """Convert a sphere OCR result to a GeoDataFrame.

        Args:
            sphere_ocr: The OCR result to convert.

        Returns:
            A GeoDataFrame, or None if coordinates are invalid.
        """
        polygon = self._sphere_ocr_to_polygon(sphere_ocr)
        if polygon is None:
            return None
        return self._polygon_to_gdf(polygon)

    def _get_intersection(
        self, gdf_1: gpd.GeoDataFrame, gdf_2: gpd.GeoDataFrame
    ) -> Optional[RegionIntersection]:
        """Calculate the intersection between two GeoDataFrames."""
        gdf_intersection = gpd.overlay(gdf_1, gdf_2, how="intersection")

        if gdf_intersection.empty:
            return None

        region_1_area = gdf_1.area.values[0]
        region_2_area = gdf_2.area.values[0]
        intersection_area = gdf_intersection.area.values[0]

        # Ratio relative to the smaller region
        intersection_ratio = intersection_area / min(region_1_area, region_2_area)

        return RegionIntersection(
            region_1_area=region_1_area,
            region_2_area=region_2_area,
            intersection_area=intersection_area,
            intersection_ratio=intersection_ratio,
        )

    def _intersect_ocr_results(
        self,
        ocr_result_1: SphereOCRResult,
        ocr_result_2: SphereOCRResult,
    ) -> Optional[RegionIntersection]:
        """Calculate the intersection between two OCR results.

        Args:
            ocr_result_1: First OCR result.
            ocr_result_2: Second OCR result.

        Returns:
            RegionIntersection if both results have valid coordinates, None otherwise.
        """
        gdf_1 = self._sphere_ocr_to_gdf(ocr_result_1)
        gdf_2 = self._sphere_ocr_to_gdf(ocr_result_2)

        # If either result has invalid coordinates, cannot compute intersection
        if gdf_1 is None or gdf_2 is None:
            return None

        return self._get_intersection(gdf_1, gdf_2)

    def _get_texts_similarity(self, text_1: str, text_2: str) -> float:
        """Calculate Levenshtein normalized similarity between texts."""
        return textdistance.levenshtein.normalized_similarity(text_1, text_2)

    def _get_texts_overlap(self, text_1: str, text_2: str) -> float:
        """Calculate overlap normalized similarity between texts."""
        return textdistance.overlap.normalized_similarity(text_1, text_2)

    def check_duplication(
        self, ocr_result_1: SphereOCRResult, ocr_result_2: SphereOCRResult
    ) -> bool:
        """Check if two OCR results are duplicates.

        Args:
            ocr_result_1: First OCR result.
            ocr_result_2: Second OCR result.

        Returns:
            True if the results are considered duplicates.
        """
        text_similarity = self._get_texts_similarity(
            ocr_result_1.text, ocr_result_2.text
        )
        text_overlap = self._get_texts_overlap(ocr_result_1.text, ocr_result_2.text)

        # If texts are neither similar nor overlapping, not duplicates
        if (text_similarity < self.min_text_similarity) and (
            text_overlap < self.min_text_overlap
        ):
            return False

        # Check spatial intersection
        intersection = self._intersect_ocr_results(ocr_result_1, ocr_result_2)
        if intersection is None:
            return False

        if intersection.intersection_ratio < self.min_intersection_ratio:
            return False

        # Check if texts overlap and regions overlap sufficiently
        if (
            text_overlap >= self.min_text_overlap
            and intersection.intersection_ratio
            >= self.min_intersection_ratio_for_overlapping_text
        ):
            return True

        # Check if texts are similar and regions overlap sufficiently
        if (
            text_similarity >= self.min_text_similarity
            and intersection.intersection_ratio
            >= self.min_intersection_ratio_for_similar_text
        ):
            return True

        return False

    def remove_duplication_for_two_lists(
        self,
        ocr_results_0: List[SphereOCRResult],
        ocr_results_1: List[SphereOCRResult],
    ) -> Tuple[List[SphereOCRResult], List[SphereOCRResult]]:
        """Remove duplicates between two lists of OCR results.

        When duplicates are found, keeps the result with longer text,
        or higher confidence if texts are equal length.

        Args:
            ocr_results_0: First list of OCR results (modified in place).
            ocr_results_1: Second list of OCR results (modified in place).

        Returns:
            Tuple of the two lists with duplicates removed.
        """
        # Find all duplicate pairs
        duplications = []
        for i, ocr_result_0 in enumerate(ocr_results_0):
            for j, ocr_result_1 in enumerate(ocr_results_1):
                if self.check_duplication(ocr_result_0, ocr_result_1):
                    duplications.append((i, j))

        # Determine which to remove from each list
        indices_to_remove_from_0: List[int] = []
        indices_to_remove_from_1: List[int] = []

        for i, j in duplications:
            candidate_0 = ocr_results_0[i]
            candidate_1 = ocr_results_1[j]

            if len(candidate_0.text) == len(candidate_1.text):
                # Equal length: prefer higher confidence
                if candidate_0.confidence < candidate_1.confidence:
                    indices_to_remove_from_0.append(i)
                else:
                    indices_to_remove_from_1.append(j)
            elif len(candidate_0.text) > len(candidate_1.text):
                # Prefer longer text
                indices_to_remove_from_1.append(j)
            else:
                indices_to_remove_from_0.append(i)

        # Remove duplicates (in reverse order to preserve indices)
        indices_to_remove_from_0 = sorted(set(indices_to_remove_from_0), reverse=True)
        indices_to_remove_from_1 = sorted(set(indices_to_remove_from_1), reverse=True)

        for index in indices_to_remove_from_0:
            ocr_results_0.pop(index)

        for index in indices_to_remove_from_1:
            ocr_results_1.pop(index)

        return ocr_results_0, ocr_results_1

    def _select_best_result(
        self,
        result_1: SphereOCRResult,
        result_2: SphereOCRResult,
    ) -> SphereOCRResult:
        """Select the best result from two duplicates.

        Prefers longer text, or higher confidence if equal length.

        Args:
            result_1: First OCR result.
            result_2: Second OCR result.

        Returns:
            The better of the two results.
        """
        if len(result_1.text) == len(result_2.text):
            # Equal length: prefer higher confidence
            return result_1 if result_1.confidence >= result_2.confidence else result_2
        elif len(result_1.text) > len(result_2.text):
            return result_1
        else:
            return result_2

    def deduplicate_frames(
        self,
        frames: List[List[SphereOCRResult]],
    ) -> List[SphereOCRResult]:
        """Deduplicate OCR results across multiple frames using incremental merging.

        Processes frames one by one, maintaining a master list. Each result from
        a new frame is compared against the entire master list. When duplicates
        are found, keeps the result with longer text or higher confidence.

        This approach is slower than pairwise deduplication but handles arbitrary
        perspective arrangements correctly (e.g., multiple pitch levels, custom
        arrangements).

        Args:
            frames: List of frames, each containing a list of OCR results.

        Returns:
            Deduplicated list of sphere OCR results.
        """
        if not frames:
            return []

        # Start with the first frame as the master list
        master_list: List[SphereOCRResult] = list(frames[0])

        # Process each subsequent frame
        for frame_results in frames[1:]:
            for new_result in frame_results:
                # Check for overlaps with existing master list
                duplicate_indices = []

                for i, master_result in enumerate(master_list):
                    if self.check_duplication(new_result, master_result):
                        duplicate_indices.append(i)

                if not duplicate_indices:
                    # No duplicates found, add to master list
                    master_list.append(new_result)
                else:
                    # Found duplicates - select best result among all candidates
                    candidates = [new_result] + [
                        master_list[i] for i in duplicate_indices
                    ]

                    # Find the best result
                    best_result = candidates[0]
                    for candidate in candidates[1:]:
                        best_result = self._select_best_result(best_result, candidate)

                    # Remove all duplicates from master list (in reverse order)
                    for i in sorted(duplicate_indices, reverse=True):
                        master_list.pop(i)

                    # Add the best result
                    master_list.append(best_result)

        return master_list
