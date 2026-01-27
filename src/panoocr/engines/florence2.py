"""Florence-2 OCR Engine using Microsoft's Florence-2 model.

This module provides OCR using Microsoft's Florence-2 vision-language model.
Requires transformers and torch.

Install with: pip install "panoocr[florence2]"
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult

if TYPE_CHECKING:
    import torch


def _check_florence2_dependencies():
    """Check if Florence-2 dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        missing.append("transformers")

    try:
        import einops
    except ImportError:
        missing.append("einops")

    try:
        import timm
    except ImportError:
        missing.append("timm")

    if missing:
        raise ImportError(
            f"Florence-2 dependencies not installed: {', '.join(missing)}\n\n"
            "Install with:\n"
            "  pip install 'panoocr[florence2]'\n\n"
            "For GPU support, install PyTorch with CUDA."
        )


@dataclass
class Florence2OCRResult:
    """Raw result from Florence-2."""

    text: str
    bounding_box: List[List[float]]  # 4 corner points from quad_boxes
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
            confidence=1.0,  # Florence-2 doesn't provide confidence scores
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="FLORENCE_2",
        )


class Florence2OCREngine:
    """OCR engine using Microsoft's Florence-2 model.

    Florence-2 is a vision-language model that can perform OCR with region
    detection. It provides good accuracy across many languages and can
    detect text in various orientations.

    Attributes:
        device: Device to run inference on (cuda, mps, or cpu).
        model: The Florence-2 model.
        processor: The Florence-2 processor.

    Example:
        >>> from panoocr.engines.florence2 import Florence2OCREngine
        >>>
        >>> engine = Florence2OCREngine(config={
        ...     "model_id": "microsoft/Florence-2-large",
        ... })
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[florence2]"
        For GPU support, install PyTorch with CUDA.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the Florence-2 engine.

        Args:
            config: Configuration dictionary with optional keys:
                - model_id: HuggingFace model ID (default: "microsoft/Florence-2-large").
                - device: Device to use ("cuda", "mps", "cpu", or None for auto).

        Raises:
            ImportError: If dependencies are not installed.
        """
        # Check dependencies first
        _check_florence2_dependencies()

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        config = config or {}

        model_id = config.get("model_id", "microsoft/Florence-2-large")

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

        # Select dtype based on device
        if torch.cuda.is_available() and device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        print(f"Loading Florence-2 model on {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.dtype, trust_remote_code=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Florence-2 model loaded successfully.")

        self.prompt = "<OCR_WITH_REGION>"

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        import torch

        inputs = self.processor(
            text=self.prompt, images=image, return_tensors="pt"
        ).to(self.device, self.dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task="<OCR_WITH_REGION>",
            image_size=(image.width, image.height),
        )

        florence2_results = []
        try:
            ocr_data = parsed_answer.get("<OCR_WITH_REGION>", {})
            quad_boxes = ocr_data.get("quad_boxes", [])
            labels = ocr_data.get("labels", [])

            for quad_box, label in zip(quad_boxes, labels):
                # Clean up text
                label = label.replace("</s>", "").replace("<s>", "")

                # Convert quad_box [x1,y1,x2,y2,x3,y3,x4,y4] to corner points
                bounding_box = [
                    [quad_box[0], quad_box[1]],
                    [quad_box[2], quad_box[3]],
                    [quad_box[4], quad_box[5]],
                    [quad_box[6], quad_box[7]],
                ]

                florence2_results.append(
                    Florence2OCRResult(
                        text=label,
                        bounding_box=bounding_box,
                        image_width=image.width,
                        image_height=image.height,
                    )
                )
        except KeyError:
            print("Error parsing OCR results, returning empty list")

        # Clean up to prevent memory leak
        del inputs
        del generated_ids
        del generated_text
        del parsed_answer
        gc.collect()

        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

        return [result.to_flat() for result in florence2_results]
