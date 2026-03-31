"""GLM-OCR Engine using mlx-vlm.

Document-focused VLM (0.9B params) from Zhipu AI. Included as a
domain-mismatch control -- trained on documents, not scene text.
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


def _check_glm_ocr_dependencies():
    try:
        import mlx_vlm
    except ImportError:
        raise ImportError(
            "GLM-OCR dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[mlx-vlm]' torch"
        )


class GlmOCREngine:
    """OCR engine using GLM-OCR (0.9B) via mlx-vlm.

    This is an unstructured engine -- it returns text without bounding boxes.
    Each text line gets a full-image bounding box for crop-level attribution.

    Example:
        >>> from panoocr.engines.glm_ocr import GlmOCREngine
        >>> engine = GlmOCREngine()
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[mlx-vlm]" torch
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _check_glm_ocr_dependencies()

        from mlx_vlm import load

        config = config or {}
        model_id = config.get("model_id", "mlx-community/GLM-OCR-4bit")
        self.max_tokens = config.get("max_tokens", 512)
        self.prompt = config.get("prompt", DEFAULT_PROMPT)

        print(f"Loading GLM-OCR model: {model_id}...")
        self.model, self.processor = load(model_id)
        print("GLM-OCR model loaded.")

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
            # Skip markdown artifacts from VLM output
            if line.startswith("```") or line.startswith("---"):
                continue
            results.append(
                FlatOCRResult(
                    text=line,
                    confidence=1.0,
                    bounding_box=_FULL_IMAGE_BBOX,
                    engine="GLM_OCR",
                )
            )

        return results
