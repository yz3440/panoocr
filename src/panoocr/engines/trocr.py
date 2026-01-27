"""TrOCR Engine using Microsoft's TrOCR model.

This module provides OCR using Microsoft's TrOCR (Transformer-based OCR) model.
NOTE: This engine is experimental and does not return bounding boxes.

Install with: pip install "panoocr[trocr]"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, TYPE_CHECKING

from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult

if TYPE_CHECKING:
    import torch


def _check_trocr_dependencies():
    """Check if TrOCR dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        missing.append("transformers")

    if missing:
        raise ImportError(
            f"TrOCR dependencies not installed: {', '.join(missing)}\n\n"
            "Install with:\n"
            "  pip install 'panoocr[trocr]'\n\n"
            "For GPU support, install PyTorch with CUDA."
        )


class TrOCRModel(Enum):
    """Available TrOCR models."""

    MICROSOFT_TROCR_LARGE_PRINTED = "microsoft/trocr-large-printed"
    MICROSOFT_TROCR_BASE_HANDWRITTEN = "microsoft/trocr-base-handwritten"
    MICROSOFT_TROCR_SMALL_PRINTED = "microsoft/trocr-small-printed"


DEFAULT_MODEL = TrOCRModel.MICROSOFT_TROCR_BASE_HANDWRITTEN


class TrOCREngine:
    """OCR engine using Microsoft's TrOCR model.

    TrOCR is a transformer-based OCR model that excels at single-line text
    recognition. It does NOT provide bounding boxes - it reads the entire
    image as a single text line.

    WARNING: This engine is experimental and may not work well for panorama
    OCR since it doesn't detect text regions. Consider using Florence2OCREngine
    or other engines that provide region detection.

    Attributes:
        model: The TrOCR model.
        processor: The TrOCR processor.

    Example:
        >>> from panoocr.engines.trocr import TrOCREngine, TrOCRModel
        >>>
        >>> engine = TrOCREngine(config={
        ...     "model": TrOCRModel.MICROSOFT_TROCR_LARGE_PRINTED,
        ... })
        >>> # Note: Returns single result for entire image
        >>> results = engine.recognize(cropped_text_image)

    Note:
        Install with: pip install "panoocr[trocr]"
        For GPU support, install PyTorch with CUDA.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the TrOCR engine.

        Args:
            config: Configuration dictionary with optional keys:
                - model: TrOCRModel enum value or model ID string.
                - device: Device to use ("cuda", "mps", "cpu", or None for auto).

        Raises:
            ImportError: If dependencies are not installed.
            ValueError: If configuration values are invalid.
        """
        # Check dependencies first
        _check_trocr_dependencies()

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        config = config or {}

        # Parse model
        model = config.get("model", DEFAULT_MODEL)
        try:
            model_id = model.value if isinstance(model, TrOCRModel) else model
        except (KeyError, AttributeError):
            raise ValueError("Invalid model specified")

        # Auto-detect device
        device = config.get("device")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        print(f"Loading TrOCR model {model_id} on {device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
        print("TrOCR model loaded successfully.")

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        NOTE: TrOCR treats the entire image as a single text line and does
        not provide bounding boxes. This makes it unsuitable for most panorama
        OCR use cases. The result will have a bounding box covering the entire
        image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List with single FlatOCRResult covering the entire image, or empty
            list if no text is recognized.
        """
        import torch

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # TrOCR doesn't provide bounding boxes - return full image bbox
        if generated_text.strip():
            return [
                FlatOCRResult(
                    text=generated_text.strip(),
                    confidence=1.0,  # TrOCR doesn't provide confidence
                    bounding_box=BoundingBox(
                        left=0.0,
                        top=0.0,
                        right=1.0,
                        bottom=1.0,
                        width=1.0,
                        height=1.0,
                    ),
                    engine="TROCR",
                )
            ]

        return []
