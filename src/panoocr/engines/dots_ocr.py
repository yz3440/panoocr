"""DOTS.OCR Engine using mlx-vlm.

Document-focused VLM (2.9B params) from RedNote. Included as a second
document-domain-mismatch data point alongside GLM-OCR.
Returns unstructured text (no bounding boxes).

Install with: pip install "panoocr[mlx-vlm]" torch
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List

from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult

_FULL_IMAGE_BBOX = BoundingBox(
    left=0.0, top=0.0, right=1.0, bottom=1.0, width=1.0, height=1.0
)

DEFAULT_PROMPT = (
    "Extract all visible text from this image. "
    "Return each text fragment on its own line, exactly as it appears."
)


def _check_dots_ocr_dependencies():
    try:
        import mlx_vlm
    except ImportError:
        raise ImportError(
            "DOTS.OCR dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[mlx-vlm]' torch"
        )


class DotsOCREngine:
    """OCR engine using DOTS.OCR (2.9B) via mlx-vlm.

    This is an unstructured engine -- it returns text without bounding boxes.
    Each text line gets a full-image bounding box for crop-level attribution.

    Example:
        >>> from panoocr.engines.dots_ocr import DotsOCREngine
        >>> engine = DotsOCREngine()
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[mlx-vlm]" torch
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _check_dots_ocr_dependencies()

        from mlx_vlm import load

        config = config or {}
        model_id = config.get("model_id", "mlx-community/dots.ocr-4bit")
        self.max_tokens = config.get("max_tokens", 512)
        self.prompt = config.get("prompt", DEFAULT_PROMPT)

        print(f"Loading DOTS.OCR model: {model_id}...")
        self.model, self.processor = load(model_id)
        print("DOTS.OCR model loaded.")

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        from mlx_vlm import generate

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f, format="JPEG")
            tmp_path = f.name

        try:
            result = generate(
                self.model,
                self.processor,
                image=tmp_path,
                prompt=self.prompt,
                max_tokens=self.max_tokens,
                verbose=False,
            )
        finally:
            import os
            os.unlink(tmp_path)

        output_text = result.text if hasattr(result, "text") else str(result)

        results = []
        for line in output_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("```") or line.startswith("---"):
                continue
            results.append(
                FlatOCRResult(
                    text=line,
                    confidence=1.0,
                    bounding_box=_FULL_IMAGE_BBOX,
                    engine="DOTS_OCR",
                )
            )

        return results
