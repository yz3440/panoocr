"""PaddleOCR Engine using the PaddleOCR library.

This module provides OCR using PaddlePaddle's PaddleOCR library.
Supports multiple languages and runs on CPU or GPU.

Install with: pip install "panoocr[paddleocr]"
"""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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
            "For GPU support, install paddlepaddle-gpu instead of paddlepaddle."
        )


class PaddleOCRLanguageCode(Enum):
    """Supported language codes for PaddleOCR."""

    ENGLISH = "en"
    CHINESE = "ch"
    FRENCH = "french"
    GERMAN = "german"
    KOREAN = "korean"
    JAPANESE = "japan"


DEFAULT_LANGUAGE = PaddleOCRLanguageCode.ENGLISH
DEFAULT_RECOGNIZE_UPSIDE_DOWN = False

# PP-OCR V4 Server model URLs
PP_OCR_V4_SERVER = {
    "detection_model": "https://paddleocr.bj.bcebos.com/models/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
    "detection_yml": "https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml",
    "recognition_model": "https://paddleocr.bj.bcebos.com/models/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar",
    "recognition_yml": "https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml",
    "cls_model": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar",
}


@dataclass
class PaddleOCRResult:
    """Raw result from PaddleOCR."""

    text: str
    bounding_box: List[List[float]]  # 4 corner points
    confidence: float
    image_width: int
    image_height: int
    use_v4_server: bool

    def to_flat(self) -> FlatOCRResult:
        """Convert to FlatOCRResult with normalized coordinates."""
        left = min(p[0] for p in self.bounding_box)
        right = max(p[0] for p in self.bounding_box)
        top = min(p[1] for p in self.bounding_box)
        bottom = max(p[1] for p in self.bounding_box)

        engine_name = "PADDLE_OCR_SERVER_V4" if self.use_v4_server else "PADDLE_OCR"

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
            engine=engine_name,
        )


class PaddleOCREngine:
    """OCR engine using PaddleOCR library.

    PaddleOCR is developed by PaddlePaddle and supports multiple languages.
    It provides good accuracy and can optionally use the V4 server model
    for better results on Chinese text.

    Attributes:
        language_preference: Language code for recognition.
        recognize_upside_down: Whether to use angle classifier.
        use_v4_server: Whether to use the V4 server model.

    Example:
        >>> from panoocr.engines.paddleocr import PaddleOCREngine, PaddleOCRLanguageCode
        >>>
        >>> engine = PaddleOCREngine(config={
        ...     "language_preference": PaddleOCRLanguageCode.CHINESE,
        ...     "use_gpu": True,
        ... })
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[paddleocr]"
        For GPU support, install paddlepaddle-gpu.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the PaddleOCR engine.

        Args:
            config: Configuration dictionary with optional keys:
                - language_preference: PaddleOCRLanguageCode value.
                - recognize_upside_down: Enable angle classifier (default: False).
                - use_v4_server: Use V4 server model for better Chinese OCR.
                - use_gpu: Whether to use GPU (default: True).
                - model_dir: Custom directory for V4 server models.

        Raises:
            ImportError: If paddleocr is not installed.
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

        # Parse other settings
        self.recognize_upside_down = config.get(
            "recognize_upside_down", DEFAULT_RECOGNIZE_UPSIDE_DOWN
        )
        if not isinstance(self.recognize_upside_down, bool):
            raise ValueError("recognize_upside_down must be a boolean")

        self.use_v4_server = config.get("use_v4_server", False)
        if not isinstance(self.use_v4_server, bool):
            raise ValueError("use_v4_server must be a boolean")

        use_gpu = config.get("use_gpu", True)
        self.model_dir = config.get("model_dir", "./models")

        # Initialize OCR engine
        if not self.use_v4_server:
            self.ocr = PaddleOCR(
                use_angle_cls=self.recognize_upside_down,
                lang=self.language_preference,
                use_gpu=use_gpu,
            )
        else:
            # Download and setup V4 server models
            self._download_v4_server_models()

            model_base = Path(self.model_dir) / "PP-OCRv4" / "chinese"
            self.ocr = PaddleOCR(
                use_angle_cls=self.recognize_upside_down,
                det_model_dir=str(model_base / "ch_PP-OCRv4_det_server_infer"),
                det_algorithm="DB",
                rec_model_dir=str(model_base / "ch_PP-OCRv4_rec_server_infer"),
                rec_algorithm="CRNN",
                cls_model_dir=str(model_base / "ch_ppocr_mobile_v2.0_cls_slim_infer"),
                use_gpu=use_gpu,
            )

    def _download_v4_server_models(self) -> None:
        """Download PP-OCR V4 server models if not present."""
        import requests

        model_base = Path(self.model_dir) / "PP-OCRv4" / "chinese"
        model_base.mkdir(parents=True, exist_ok=True)

        models_to_download = [
            ("detection_model", "ch_PP-OCRv4_det_server_infer"),
            ("recognition_model", "ch_PP-OCRv4_rec_server_infer"),
            ("cls_model", "ch_ppocr_mobile_v2.0_cls_slim_infer"),
        ]

        for key, folder_name in models_to_download:
            tar_path = model_base / f"{folder_name}.tar"
            folder_path = model_base / folder_name

            if not folder_path.exists():
                if not tar_path.exists():
                    print(f"Downloading {folder_name}...")
                    r = requests.get(PP_OCR_V4_SERVER[key], allow_redirects=True)
                    tar_path.write_bytes(r.content)

                print(f"Extracting {folder_name}...")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(model_base)

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        image_array = np.array(image)

        # Use slicing for large images
        slice_config = {
            "horizontal_stride": 300,
            "vertical_stride": 500,
            "merge_x_thres": 50,
            "merge_y_thres": 35,
        }

        annotations = self.ocr.ocr(image_array, cls=True, slice=slice_config)

        paddle_results = []
        for annotation in annotations:
            if not isinstance(annotation, list):
                continue

            for res in annotation:
                bounding_box = res[0]
                text = res[1][0]
                confidence = res[1][1]

                paddle_results.append(
                    PaddleOCRResult(
                        text=text,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        image_width=image.width,
                        image_height=image.height,
                        use_v4_server=self.use_v4_server,
                    )
                )

        return [result.to_flat() for result in paddle_results]
