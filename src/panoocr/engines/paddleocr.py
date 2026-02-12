"""PaddleOCR Engine using the PaddleOCR library.

This module provides OCR using PaddlePaddle's PaddleOCR library.
Supports multiple languages and runs on CPU or GPU.

Install with: pip install "panoocr[paddleocr]"

Note:
    PaddleOCR 3.x has a significantly different API from 2.x.
    This engine supports PaddleOCR >= 3.0.0. For older versions,
    please upgrade with: pip install --upgrade paddleocr
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult


def _check_paddleocr_dependencies():
    """Check if PaddleOCR dependencies are installed."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError(
            "PaddleOCR dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[paddleocr]'\n\n"
            "You also need PaddlePaddle:\n"
            "  pip install paddlepaddle\n"
            "For GPU support, install paddlepaddle-gpu instead."
        )

    try:
        import paddle
    except ImportError:
        raise ImportError(
            "PaddlePaddle framework not installed.\n\n"
            "Install with:\n"
            "  pip install paddlepaddle\n\n"
            "For GPU support:\n"
            "  pip install paddlepaddle-gpu"
        )


class PaddleOCRLanguageCode(Enum):
    """Supported language codes for PaddleOCR."""

    ENGLISH = "en"
    CHINESE = "ch"
    FRENCH = "french"
    GERMAN = "german"
    KOREAN = "korean"
    JAPANESE = "japan"


class PaddleOCRVersion(Enum):
    """Supported PaddleOCR model versions."""

    PP_OCRV3 = "PP-OCRv3"
    PP_OCRV4 = "PP-OCRv4"
    PP_OCRV5 = "PP-OCRv5"


DEFAULT_LANGUAGE = PaddleOCRLanguageCode.ENGLISH
DEFAULT_RECOGNIZE_UPSIDE_DOWN = False


@dataclass
class PaddleOCRResult:
    """Raw result from PaddleOCR."""

    text: str
    bounding_box: List[List[float]]  # 4 corner points
    confidence: float
    image_width: int
    image_height: int

    def to_flat(self) -> FlatOCRResult:
        """Convert to FlatOCRResult with normalized coordinates."""
        left = min(p[0] for p in self.bounding_box)
        right = max(p[0] for p in self.bounding_box)
        top = min(p[1] for p in self.bounding_box)
        bottom = max(p[1] for p in self.bounding_box)

        return FlatOCRResult(
            text=self.text,
            confidence=self.confidence,
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="PADDLE_OCR",
        )


class PaddleOCREngine:
    """OCR engine using PaddleOCR library (v3.x).

    PaddleOCR is developed by PaddlePaddle and supports multiple languages.
    It provides good accuracy with automatic model management.

    Attributes:
        language_preference: Language code for recognition.
        recognize_upside_down: Whether to use textline orientation classifier.
        ocr_version: The PP-OCR model version to use.

    Example:
        >>> from panoocr.engines.paddleocr import PaddleOCREngine, PaddleOCRLanguageCode
        >>>
        >>> engine = PaddleOCREngine(config={
        ...     "language_preference": PaddleOCRLanguageCode.CHINESE,
        ... })
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[paddleocr]" paddlepaddle
        For GPU support, install paddlepaddle-gpu instead.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the PaddleOCR engine.

        Args:
            config: Configuration dictionary with optional keys:
                - language_preference: PaddleOCRLanguageCode value.
                - recognize_upside_down: Enable textline orientation classifier
                  (default: False).
                - ocr_version: PaddleOCRVersion or string like "PP-OCRv5"
                  (default: auto-selected by PaddleOCR based on language).
                - text_detection_model_name: Override detection model name.
                - text_recognition_model_name: Override recognition model name.
                - text_det_limit_side_len: Max side length for detection input.
                - text_rec_score_thresh: Minimum recognition score threshold.

        Raises:
            ImportError: If paddleocr or paddlepaddle is not installed.
            ValueError: If configuration values are invalid.
        """
        # Check dependencies first
        _check_paddleocr_dependencies()

        from paddleocr import PaddleOCR

        config = config or {}

        # Parse language preference
        language = config.get("language_preference", DEFAULT_LANGUAGE)
        try:
            self.language_preference = (
                language.value if isinstance(language, PaddleOCRLanguageCode) else language
            )
        except (KeyError, AttributeError):
            raise ValueError("Invalid language code")

        # Parse textline orientation (replaces use_angle_cls from v2)
        self.recognize_upside_down = config.get(
            "recognize_upside_down", DEFAULT_RECOGNIZE_UPSIDE_DOWN
        )
        if not isinstance(self.recognize_upside_down, bool):
            raise ValueError("recognize_upside_down must be a boolean")

        # Parse OCR version
        ocr_version = config.get("ocr_version", None)
        if isinstance(ocr_version, PaddleOCRVersion):
            ocr_version = ocr_version.value
        self.ocr_version = ocr_version

        # Handle deprecated use_v4_server config
        if config.get("use_v4_server", False):
            warnings.warn(
                "use_v4_server is deprecated in PaddleOCR 3.x. "
                "Use ocr_version='PP-OCRv4' instead. "
                "Model management is now handled automatically by PaddleOCR.",
                DeprecationWarning,
                stacklevel=2,
            )
            if ocr_version is None:
                ocr_version = "PP-OCRv4"

        # Handle deprecated use_gpu config
        if "use_gpu" in config:
            warnings.warn(
                "use_gpu is deprecated in PaddleOCR 3.x. "
                "GPU/CPU device selection is now automatic.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Build PaddleOCR kwargs
        ocr_kwargs: Dict[str, Any] = {
            "lang": self.language_preference,
            "use_textline_orientation": self.recognize_upside_down,
            # Disable document preprocessing for perspective images
            # (these are designed for scanned documents, not perspective crops)
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }

        if ocr_version is not None:
            ocr_kwargs["ocr_version"] = ocr_version

        # Allow custom model overrides
        for key in (
            "text_detection_model_name",
            "text_detection_model_dir",
            "text_recognition_model_name",
            "text_recognition_model_dir",
            "text_det_limit_side_len",
            "text_rec_score_thresh",
        ):
            if key in config:
                ocr_kwargs[key] = config[key]

        # Suppress connectivity check if env var is not set
        if "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" not in os.environ:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        self.ocr = PaddleOCR(**ocr_kwargs)

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        image_array = np.array(image)

        # Use the predict API (PaddleOCR 3.x)
        results_iter = self.ocr.predict(image_array)

        paddle_results = []
        for result in results_iter:
            # PaddleOCR 3.x returns OCRResult objects with:
            # - rec_texts: list of recognized text strings
            # - rec_scores: list of confidence scores
            # - dt_polys: list of detection polygons (numpy arrays)
            texts = result.get("rec_texts", []) if hasattr(result, "get") else getattr(result, "rec_texts", [])
            scores = result.get("rec_scores", []) if hasattr(result, "get") else getattr(result, "rec_scores", [])
            polys = result.get("dt_polys", []) if hasattr(result, "get") else getattr(result, "dt_polys", [])

            for text, score, poly in zip(texts, scores, polys):
                # Skip empty or very low confidence results
                if not text or not text.strip():
                    continue

                # Convert numpy array polygon to list of [x, y] points
                if hasattr(poly, "tolist"):
                    bounding_box = poly.tolist()
                else:
                    bounding_box = list(poly)

                paddle_results.append(
                    PaddleOCRResult(
                        text=text,
                        confidence=float(score),
                        bounding_box=bounding_box,
                        image_width=image.width,
                        image_height=image.height,
                    )
                )

        return [result.to_flat() for result in paddle_results]
