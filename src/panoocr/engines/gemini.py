"""Gemini API Engine using google-genai.

Frontier VLM for OCR accuracy ceiling measurement.
Returns structured text with bounding boxes via spatial grounding.

Install with: pip install "panoocr[gemini]"
Requires GOOGLE_GEMINI_API_KEY environment variable.
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Dict, List

from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult

_FULL_IMAGE_BBOX = BoundingBox(
    left=0.0, top=0.0, right=1.0, bottom=1.0, width=1.0, height=1.0
)

BBOX_PROMPT = (
    "For each piece of text visible in this image, return a JSON array "
    "where each element has:\n"
    '- "text": the text string exactly as it appears\n'
    '- "box": [y1, x1, y2, x2] normalized bounding box coordinates (0-1000)\n'
    "\n"
    "Include partial, blurry, or small text. Do not interpret or correct spelling.\n"
    "Return ONLY the JSON array, no other text."
)

PLAIN_PROMPT = (
    "Extract all visible text from this street-level image. "
    "Return each text fragment on its own line, exactly as it appears. "
    "Include partial, blurry, or small text. Do not interpret or correct spelling."
)


def _check_gemini_dependencies():
    try:
        import google.genai
    except ImportError:
        raise ImportError(
            "Gemini dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[gemini]'\n\n"
            "Also set GOOGLE_GEMINI_API_KEY environment variable."
        )


def _parse_bbox_response(text: str, engine_tag: str) -> List[FlatOCRResult]:
    """Parse Gemini JSON bbox response into FlatOCRResults."""
    text = text.strip()
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if not json_match:
        return []

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    results = []
    for item in items:
        txt = item.get("text", "").strip()
        box = item.get("box", [])
        if not txt or len(box) != 4:
            continue

        # Gemini returns [y1, x1, y2, x2] in 0-1000 range
        y1, x1, y2, x2 = [v / 1000.0 for v in box]
        if y1 > y2:
            y1, y2 = y2, y1
        if x1 > x2:
            x1, x2 = x2, x1

        results.append(FlatOCRResult(
            text=txt,
            confidence=1.0,
            bounding_box=BoundingBox(
                left=x1, top=y1, right=x2, bottom=y2,
                width=x2 - x1, height=y2 - y1,
            ),
            engine=engine_tag,
        ))

    return results


class GeminiEngine:
    """OCR engine using Gemini API with spatial grounding.

    Returns structured text with per-detection bounding boxes by default.
    Set config "use_bbox" to False for plain text mode (crop-level attribution).

    Example:
        >>> from panoocr.engines.gemini import GeminiEngine
        >>> engine = GeminiEngine(config={"model": "gemini-2.5-flash"})
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[gemini]"
        Requires GOOGLE_GEMINI_API_KEY environment variable.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _check_gemini_dependencies()

        from google import genai

        config = config or {}

        self._load_dotenv()

        api_key = config.get(
            "api_key",
            os.environ.get("GOOGLE_GEMINI_API_KEY",
                os.environ.get("GEMINI_API_KEY",
                    os.environ.get("GOOGLE_API_KEY"))),
        )
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_GEMINI_API_KEY, "
                "GEMINI_API_KEY, or GOOGLE_API_KEY environment variable, "
                "or pass api_key in config."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = config.get("model", "gemini-2.5-pro")
        self.use_bbox = config.get("use_bbox", True)

        if self.use_bbox:
            self.prompt = config.get("prompt", BBOX_PROMPT)
        else:
            self.prompt = config.get("prompt", PLAIN_PROMPT)

        if "flash" in self.model_name:
            self._engine_tag = "GEMINI_2_5_FLASH"
        elif "pro" in self.model_name:
            self._engine_tag = "GEMINI_2_5_PRO"
        else:
            self._engine_tag = "GEMINI"

    @staticmethod
    def _load_dotenv():
        try:
            from pathlib import Path

            env_path = Path(".env")
            if not env_path.exists():
                env_path = Path(__file__).resolve().parents[3] / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            pass

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        from google.genai import types

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        image_bytes = buf.getvalue()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=self.prompt),
                    ]
                )
            ],
        )

        output_text = response.text or ""

        if self.use_bbox:
            return _parse_bbox_response(output_text, self._engine_tag)

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
                    engine=self._engine_tag,
                )
            )

        return results
