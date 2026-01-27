"""Duplication detection for spherical OCR results."""

from __future__ import annotations

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

    def _sphere_ocr_to_polygon(self, sphere_ocr: SphereOCRResult) -> Polygon:
        """Convert a sphere OCR result to a polygon in yaw/pitch space."""
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

    def _sphere_ocr_to_gdf(self, sphere_ocr: SphereOCRResult) -> gpd.GeoDataFrame:
        """Convert a sphere OCR result to a GeoDataFrame."""
        return self._polygon_to_gdf(self._sphere_ocr_to_polygon(sphere_ocr))

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
        """Calculate the intersection between two OCR results."""
        gdf_1 = self._sphere_ocr_to_gdf(ocr_result_1)
        gdf_2 = self._sphere_ocr_to_gdf(ocr_result_2)

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
