"""MacOCR Engine using Apple Vision Framework.

This module provides OCR using macOS's built-in Vision Framework.
Requires macOS and the ocrmac package.

Install with: pip install "panoocr[macocr]"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult


def _check_macocr_dependencies():
    """Check if MacOCR dependencies are installed."""
    try:
        import ocrmac.ocrmac
    except ImportError:
        raise ImportError(
            "MacOCR dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[macocr]'\n\n"
            "Note: Requires macOS with Apple Vision Framework."
        )


class MacOCRRecognitionLevel(Enum):
    """Recognition accuracy level for MacOCR."""

    FAST = "fast"
    ACCURATE = "accurate"


class MacOCRLanguageCode(Enum):
    """Supported language codes for MacOCR."""

    ENGLISH_US = "en-US"
    FRENCH_FR = "fr-FR"
    ITALIAN_IT = "it-IT"
    GERMAN_DE = "de-DE"
    SPANISH_ES = "es-ES"
    PORTUGUESE_BR = "pt-BR"
    CHINESE_SIMPLIFIED = "zh-Hans"
    CHINESE_TRADITIONAL = "zh-Hant"
    CHINESE_YUE_SIMPLIFIED = "yue-Hans"
    CHINESE_YUE_TRADITIONAL = "yue-Hant"
    KOREAN_KR = "ko-KR"
    JAPANESE_JP = "ja-JP"
    RUSSIAN_RU = "ru-RU"
    UKRAINIAN_UA = "uk-UA"
    THAI_TH = "th-TH"
    VIETNAMESE_VT = "vi-VT"


DEFAULT_LANGUAGE_PREFERENCE = [MacOCRLanguageCode.ENGLISH_US]
DEFAULT_RECOGNITION_LEVEL = MacOCRRecognitionLevel.ACCURATE


@dataclass
class MacOCRResult:
    """Raw result from MacOCR."""

    text: str
    bounding_box: Tuple[float, float, float, float]
    confidence: float

    def to_flat(self) -> FlatOCRResult:
        """Convert to FlatOCRResult with normalized coordinates."""
        # MacOCR returns (x, y, width, height) where y is from bottom
        left = self.bounding_box[0]
        right = self.bounding_box[0] + self.bounding_box[2]
        top = 1 - self.bounding_box[1] - self.bounding_box[3]
        bottom = 1 - self.bounding_box[1]

        return FlatOCRResult(
            text=self.text,
            confidence=self.confidence,
            bounding_box=BoundingBox(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                width=right - left,
                height=bottom - top,
            ),
            engine="APPLE_VISION_FRAMEWORK",
        )


class MacOCREngine:
    """OCR engine using Apple Vision Framework via ocrmac.

    This engine uses macOS's built-in Vision Framework for text recognition.
    It provides excellent accuracy for many languages on Apple Silicon.

    Attributes:
        language_preference: List of language codes to use for recognition.
        recognition_level: Recognition accuracy level ("fast" or "accurate").

    Example:
        >>> from panoocr.engines.macocr import MacOCREngine, MacOCRLanguageCode
        >>>
        >>> engine = MacOCREngine(config={
        ...     "language_preference": [MacOCRLanguageCode.ENGLISH_US],
        ...     "recognition_level": MacOCRRecognitionLevel.ACCURATE,
        ... })
        >>> results = engine.recognize(image)

    Note:
        Requires macOS and the ocrmac package.
        Install with: pip install "panoocr[macocr]"
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the MacOCR engine.

        Args:
            config: Configuration dictionary with optional keys:
                - language_preference: List of MacOCRLanguageCode values.
                - recognition_level: MacOCRRecognitionLevel value.

        Raises:
            ImportError: If ocrmac is not installed.
            ValueError: If configuration values are invalid.
        """
        # Check dependencies first
        _check_macocr_dependencies()

        config = config or {}

        # Parse language preference
        language_preference = config.get(
            "language_preference", DEFAULT_LANGUAGE_PREFERENCE
        )
        try:
            self.language_preference = [
                lang.value if isinstance(lang, MacOCRLanguageCode) else lang
                for lang in language_preference
            ]
        except (KeyError, AttributeError):
            raise ValueError("Invalid language code in language_preference")

        # Parse recognition level
        recognition_level = config.get("recognition_level", DEFAULT_RECOGNITION_LEVEL)
        try:
            self.recognition_level = (
                recognition_level.value
                if isinstance(recognition_level, MacOCRRecognitionLevel)
                else recognition_level
            )
        except (KeyError, AttributeError):
            raise ValueError("Invalid recognition level")

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        import ocrmac.ocrmac

        annotations = ocrmac.ocrmac.OCR(
            image,
            recognition_level=self.recognition_level,
            language_preference=self.language_preference,
        ).recognize()

        mac_ocr_results = [
            MacOCRResult(
                text=annotation[0],
                confidence=annotation[1],
                bounding_box=annotation[2],
            )
            for annotation in annotations
        ]

        return [result.to_flat() for result in mac_ocr_results]
