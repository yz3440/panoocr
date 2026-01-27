"""PanoOCR pipeline-first API client."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from PIL import Image
from tqdm import tqdm

from ..image.models import PanoramaImage, PerspectiveMetadata
from ..image.perspectives import (
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
    combine_perspectives,
)
from ..ocr.models import FlatOCRResult, SphereOCRResult
from ..dedup.detection import SphereOCRDuplicationDetectionEngine

from .models import (
    OCREngine,
    OCROptions,
    DedupOptions,
    PerspectivePreset,
    OCRResult,
)


def _get_perspectives_for_preset(preset: PerspectivePreset) -> List[PerspectiveMetadata]:
    """Get perspective list for a preset."""
    if preset == PerspectivePreset.DEFAULT:
        return DEFAULT_IMAGE_PERSPECTIVES
    elif preset == PerspectivePreset.ZOOMED_IN:
        return ZOOMED_IN_IMAGE_PERSPECTIVES
    elif preset == PerspectivePreset.ZOOMED_OUT:
        return ZOOMED_OUT_IMAGE_PERSPECTIVES
    else:
        raise ValueError(f"Unknown perspective preset: {preset}")


class PanoOCR:
    """Pipeline-first API for panorama OCR.

    This class provides a high-level interface for running OCR on
    equirectangular panorama images with automatic perspective projection
    and deduplication.

    Example:
        >>> from panoocr import PanoOCR
        >>> from panoocr.engines.macocr import MacOCREngine
        >>>
        >>> engine = MacOCREngine()
        >>> pano = PanoOCR(engine)
        >>> result = pano.recognize("panorama.jpg")
        >>> result.save_json("results.json")

    Attributes:
        engine: The OCR engine to use for text recognition.
        perspectives: List of perspective configurations.
        dedup_options: Deduplication options.
    """

    def __init__(
        self,
        engine: OCREngine,
        perspectives: Optional[Union[PerspectivePreset, List[PerspectiveMetadata]]] = None,
        dedup_options: Optional[DedupOptions] = None,
    ):
        """Initialize PanoOCR.

        Args:
            engine: OCR engine implementing the OCREngine protocol.
            perspectives: Perspective configuration - either a preset name or
                custom list of PerspectiveMetadata. Defaults to DEFAULT.
            dedup_options: Deduplication options. Uses defaults if not provided.
        """
        self.engine = engine

        # Set up perspectives
        if perspectives is None:
            self.perspectives = DEFAULT_IMAGE_PERSPECTIVES
            self._preset_name = PerspectivePreset.DEFAULT.value
        elif isinstance(perspectives, PerspectivePreset):
            self.perspectives = _get_perspectives_for_preset(perspectives)
            self._preset_name = perspectives.value
        else:
            self.perspectives = perspectives
            self._preset_name = "custom"

        # Set up deduplication
        self.dedup_options = dedup_options or DedupOptions()
        self._dedup_engine = SphereOCRDuplicationDetectionEngine(
            min_text_similarity=self.dedup_options.min_text_similarity,
            min_intersection_ratio_for_similar_text=self.dedup_options.min_intersection_ratio_for_similar_text,
            min_text_overlap=self.dedup_options.min_text_overlap,
            min_intersection_ratio_for_overlapping_text=self.dedup_options.min_intersection_ratio_for_overlapping_text,
            min_intersection_ratio=self.dedup_options.min_intersection_ratio,
        )

    def recognize(
        self,
        image: Union[str, Image.Image],
        panorama_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Run OCR on a panorama image.

        Args:
            image: Path to panorama image or PIL Image.
            panorama_id: Optional identifier for the panorama.
            show_progress: Whether to show a progress bar.

        Returns:
            OCRResult containing deduplicated sphere OCR results.
        """
        # Get image path for result metadata
        image_path = image if isinstance(image, str) else None
        if panorama_id is None:
            panorama_id = image_path or "panorama"

        # Load panorama
        pano = PanoramaImage(panorama_id=panorama_id, image=image)

        # Run OCR on each perspective
        all_sphere_results: List[List[SphereOCRResult]] = []

        perspective_iter = self.perspectives
        if show_progress:
            perspective_iter = tqdm(
                self.perspectives,
                desc="Processing perspectives",
                unit="perspective",
            )

        for perspective in perspective_iter:
            # Generate perspective view
            persp_image = pano.generate_perspective_image(perspective)

            # Run OCR
            flat_results = self.engine.recognize(persp_image.get_perspective_image())

            # Convert to sphere coordinates
            sphere_results = [
                result.to_sphere(
                    horizontal_fov=perspective.horizontal_fov,
                    vertical_fov=perspective.vertical_fov,
                    yaw_offset=perspective.yaw_offset,
                    pitch_offset=perspective.pitch_offset,
                )
                for result in flat_results
            ]

            all_sphere_results.append(sphere_results)

        # Deduplicate across adjacent perspectives
        deduplicated = self._deduplicate_results(all_sphere_results)

        return OCRResult(
            results=deduplicated,
            image_path=image_path,
            perspective_preset=self._preset_name,
        )

    def recognize_multi(
        self,
        image: Union[str, Image.Image],
        presets: Sequence[PerspectivePreset],
        panorama_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Run OCR on a panorama using multiple perspective presets.

        Useful for multi-scale detection to catch both small and large text.

        Args:
            image: Path to panorama image or PIL Image.
            presets: List of perspective presets to use.
            panorama_id: Optional identifier for the panorama.
            show_progress: Whether to show a progress bar.

        Returns:
            OCRResult containing deduplicated sphere OCR results.
        """
        # Get image path for result metadata
        image_path = image if isinstance(image, str) else None
        if panorama_id is None:
            panorama_id = image_path or "panorama"

        # Combine perspectives from all presets
        combined_perspectives = combine_perspectives(
            *[_get_perspectives_for_preset(preset) for preset in presets]
        )

        # Temporarily swap perspectives
        original_perspectives = self.perspectives
        original_preset_name = self._preset_name

        self.perspectives = combined_perspectives
        self._preset_name = None

        try:
            result = self.recognize(
                image=image,
                panorama_id=panorama_id,
                show_progress=show_progress,
            )
            # Update result with multi-preset info
            return OCRResult(
                results=result.results,
                image_path=image_path,
                perspective_preset=None,
                perspective_presets=[preset.value for preset in presets],
            )
        finally:
            # Restore original perspectives
            self.perspectives = original_perspectives
            self._preset_name = original_preset_name

    def _deduplicate_results(
        self,
        all_results: List[List[SphereOCRResult]],
    ) -> List[SphereOCRResult]:
        """Deduplicate OCR results across perspectives.

        Compares results from adjacent perspectives and removes duplicates,
        keeping the result with longer text or higher confidence.

        Args:
            all_results: List of result lists, one per perspective.

        Returns:
            Deduplicated list of sphere OCR results.
        """
        if not all_results:
            return []

        # Deduplicate between adjacent perspective pairs
        for i in range(len(all_results) - 1):
            all_results[i], all_results[i + 1] = (
                self._dedup_engine.remove_duplication_for_two_lists(
                    all_results[i], all_results[i + 1]
                )
            )

        # Handle wrap-around (last and first perspectives are adjacent)
        if len(all_results) > 1:
            all_results[-1], all_results[0] = (
                self._dedup_engine.remove_duplication_for_two_lists(
                    all_results[-1], all_results[0]
                )
            )

        # Flatten results
        deduplicated = []
        for results in all_results:
            deduplicated.extend(results)

        return deduplicated
