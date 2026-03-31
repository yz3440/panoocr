"""RapidOCR Engine using PP-OCRv4/v5 via ONNX Runtime.

This module provides OCR using PaddleOCR's models converted to ONNX,
running via ONNX Runtime. Bypasses PaddlePaddle framework entirely,
avoiding documented freeze/crash issues on Apple Silicon.

Supports both PP-OCRv4 (default) and PP-OCRv5 via config.

Install with: pip install "panoocr[rapidocr]"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult


def _check_rapidocr_dependencies():
    try:
        import rapidocr
    except ImportError:
        raise ImportError(
            "RapidOCR dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[rapidocr]'\n\n"
            "Also requires onnxruntime:\n"
            "  pip install onnxruntime"
        )


@dataclass
class RapidOCRResult:
    """Raw result from RapidOCR."""

    text: str
    bounding_box: np.ndarray  # shape (4, 2) — 4-point polygon in pixel coords
    confidence: float
    image_width: int
    image_height: int

    def to_flat(self, engine_tag: str) -> FlatOCRResult:
        """Convert to FlatOCRResult with normalized coordinates."""
        left = float(self.bounding_box[:, 0].min())
        right = float(self.bounding_box[:, 0].max())
        top = float(self.bounding_box[:, 1].min())
        bottom = float(self.bounding_box[:, 1].max())

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
            engine=engine_tag,
        )


class RapidOCREngine:
    """OCR engine using RapidOCR (PP-OCRv4/v5 via ONNX Runtime).

    Attributes:
        ocr_version: Which PP-OCR model version to use ("PP-OCRv4" or "PP-OCRv5").

    Example:
        >>> from panoocr.engines.rapidocr_engine import RapidOCREngine
        >>>
        >>> engine_v4 = RapidOCREngine()
        >>> engine_v5 = RapidOCREngine(config={"ocr_version": "PP-OCRv5"})
        >>> results = engine_v4.recognize(image)

    Note:
        Install with: pip install "panoocr[rapidocr]" onnxruntime
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _check_rapidocr_dependencies()

        from rapidocr import RapidOCR, OCRVersion

        config = config or {}
        version_str = config.get("ocr_version", "PP-OCRv4")

        params: Dict[str, Any] = {}

        if version_str == "PP-OCRv5":
            params["Rec.ocr_version"] = OCRVersion.PPOCRV5
            self._engine_tag = "RAPID_OCR_V5"
        else:
            self._engine_tag = "RAPID_OCR_V4"

        self.ocr = RapidOCR(params=params if params else None)
        self.ocr_version = version_str

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        image_array = np.array(image)
        result = self.ocr(image_array)

        if not result or len(result) == 0:
            return []

        rapid_results = []
        for box, txt, score in zip(result.boxes, result.txts, result.scores):
            if not txt or not txt.strip():
                continue

            rapid_results.append(
                RapidOCRResult(
                    text=txt,
                    bounding_box=box,
                    confidence=float(score),
                    image_width=image.width,
                    image_height=image.height,
                )
            )

        return [r.to_flat(self._engine_tag) for r in rapid_results]
