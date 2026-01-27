"""Public, pipeline-first API models.

These types are designed for ergonomic library usage and stable serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence, runtime_checkable

from PIL import Image

from ..ocr.models import FlatOCRResult, SphereOCRResult


class PerspectivePreset(str, Enum):
    """Pre-defined perspective configurations for common text scales."""

    DEFAULT = "default"
    ZOOMED_IN = "zoomed_in"
    ZOOMED_OUT = "zoomed_out"
    WIDEANGLE = "wideangle"


@dataclass(frozen=True)
class OCROptions:
    """Options passed to the underlying OCR engine.

    Attributes:
        config: Engine-specific configuration dictionary.
    """

    config: dict | None = None


@dataclass(frozen=True)
class DedupOptions:
    """Deduplication options applied after multi-view OCR.

    Attributes:
        min_text_similarity: Minimum Levenshtein similarity for text comparison.
        min_intersection_ratio_for_similar_text: Minimum region overlap for similar texts.
        min_text_overlap: Minimum overlap similarity for text comparison.
        min_intersection_ratio_for_overlapping_text: Minimum region overlap for overlapping texts.
        min_intersection_ratio: Minimum region intersection ratio threshold.
    """

    min_text_similarity: float = 0.5
    min_intersection_ratio_for_similar_text: float = 0.5
    min_text_overlap: float = 0.5
    min_intersection_ratio_for_overlapping_text: float = 0.15
    min_intersection_ratio: float = 0.1


@runtime_checkable
class OCREngine(Protocol):
    """Protocol for OCR engines (structural typing).

    Any class with a matching `recognize()` method can be used.
    No inheritance required.
    """

    def recognize(self, image: Image.Image) -> list[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult objects with normalized bounding boxes (0-1 range).
        """
        ...


@dataclass(frozen=True)
class OCRResult:
    """OCR output plus metadata, with preview-tool-friendly JSON export.

    Attributes:
        results: List of deduplicated sphere OCR results.
        image_path: Optional path to the source image.
        perspective_preset: Name of the perspective preset used.
        perspective_presets: List of perspective preset names if multiple were used.
    """

    results: Sequence[SphereOCRResult]
    image_path: Optional[str] = None
    perspective_preset: Optional[str] = None
    perspective_presets: Optional[Sequence[str]] = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "perspective_preset": self.perspective_preset,
            "perspective_presets": list(self.perspective_presets)
            if self.perspective_presets is not None
            else None,
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: str) -> None:
        """Save OCR results in a JSON file suitable for the preview tool.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "OCRResult":
        """Create an OCRResult from a dictionary.

        Args:
            data: Dictionary with OCR result data.

        Returns:
            OCRResult instance.
        """
        results = [SphereOCRResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            image_path=data.get("image_path"),
            perspective_preset=data.get("perspective_preset"),
            perspective_presets=data.get("perspective_presets"),
        )

    @classmethod
    def load_json(cls, path: str) -> "OCRResult":
        """Load OCR results from a JSON file.

        Args:
            path: Input file path.

        Returns:
            OCRResult instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
