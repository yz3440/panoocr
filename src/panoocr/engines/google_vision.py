"""Google Cloud Vision API Engine.

Uses the REST API with an API key for TEXT_DETECTION (scene text).
Requires GOOGLE_VISION_API_KEY environment variable or passed via config.

Install with: pip install "panoocr[google-vision]"
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def _check_google_vision_dependencies():
    try:
        import requests
    except ImportError:
        raise ImportError(
            "Google Vision dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[google-vision]'\n\n"
            "Also set GOOGLE_VISION_API_KEY environment variable."
        )


@dataclass
class GoogleVisionResult:
    """Raw result from Google Cloud Vision TEXT_DETECTION."""

    text: str
    vertices: List[Dict[str, int]]  # [{"x": int, "y": int}, ...]
    image_width: int
    image_height: int

    def to_flat(self) -> FlatOCRResult:
        """Convert to FlatOCRResult with normalized coordinates."""
        xs = [v.get("x", 0) for v in self.vertices]
        ys = [v.get("y", 0) for v in self.vertices]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)

        return FlatOCRResult(
            text=self.text,
            confidence=1.0,  # TEXT_DETECTION doesn't return per-word confidence by default
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="GOOGLE_CLOUD_VISION",
        )


class GoogleVisionEngine:
    """OCR engine using Google Cloud Vision API (TEXT_DETECTION).

    Uses the REST API with an API key. Set GOOGLE_VISION_API_KEY
    in environment or .env file, or pass via config.

    Example:
        >>> from panoocr.engines.google_vision import GoogleVisionEngine
        >>>
        >>> engine = GoogleVisionEngine()
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[google-vision]"
        Requires GOOGLE_VISION_API_KEY environment variable.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _check_google_vision_dependencies()

        config = config or {}

        self._load_dotenv()

        self.api_key = config.get(
            "api_key",
            os.environ.get("GOOGLE_VISION_API_KEY"),
        )
        if not self.api_key:
            raise ValueError(
                "Google Vision API key not found. Set GOOGLE_VISION_API_KEY "
                "environment variable or pass api_key in config."
            )

    @staticmethod
    def _load_dotenv():
        """Try to load .env file from project root."""
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
        """Recognize text in an image via Google Cloud Vision API.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        import requests

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {"content": image_b64},
                    "features": [{"type": "TEXT_DETECTION"}],
                }
            ]
        }

        response = requests.post(
            VISION_API_URL,
            params={"key": self.api_key},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        annotations = (
            data.get("responses", [{}])[0].get("textAnnotations", [])
        )

        if len(annotations) <= 1:
            return []

        # annotations[0] is the full text block; [1:] are individual words
        results = []
        for ann in annotations[1:]:
            text = ann.get("description", "")
            vertices = ann.get("boundingPoly", {}).get("vertices", [])
            if not text.strip() or len(vertices) < 3:
                continue

            results.append(
                GoogleVisionResult(
                    text=text,
                    vertices=vertices,
                    image_width=image.width,
                    image_height=image.height,
                )
            )

        return [r.to_flat() for r in results]
