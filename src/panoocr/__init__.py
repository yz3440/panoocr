"""PanoOCR: OCR for equirectangular panorama images.

PanoOCR is a Python library for performing Optical Character Recognition (OCR)
on equirectangular panorama images with automatic perspective projection and
deduplication.

Example:
    >>> from panoocr import PanoOCR
    >>> from panoocr.engines.macocr import MacOCREngine
    >>>
    >>> engine = MacOCREngine()
    >>> pano = PanoOCR(engine)
    >>> result = pano.recognize("panorama.jpg")
    >>> result.save_json("results.json")

Install OCR engine dependencies:
    - MacOCR (macOS): pip install "panoocr[macocr]"
    - EasyOCR: pip install "panoocr[easyocr]"
    - PaddleOCR: pip install "panoocr[paddleocr]"
    - Florence-2: pip install "panoocr[florence2]"
    - All engines: pip install "panoocr[full]"
"""

__version__ = "0.2.0"

# Pipeline-first public API
from .api import (
    PanoOCR,
    OCREngine,
    OCROptions,
    DedupOptions,
    PerspectivePreset,
    OCRResult,
)

# Image module
from .image.models import PanoramaImage, PerspectiveImage, PerspectiveMetadata
from .image.perspectives import (
    generate_perspectives,
    combine_perspectives,
    DEFAULT_IMAGE_PERSPECTIVES,
    ZOOMED_IN_IMAGE_PERSPECTIVES,
    ZOOMED_OUT_IMAGE_PERSPECTIVES,
)

# OCR models
from .ocr.models import BoundingBox, FlatOCRResult, SphereOCRResult

# Deduplication
from .dedup.detection import SphereOCRDuplicationDetectionEngine

# Visualization utilities
from .ocr.utils import visualize_ocr_results, visualize_sphere_ocr_results

# Geometry utilities
from .geometry import (
    uv_to_yaw_pitch,
    yaw_pitch_to_uv,
    normalize_yaw,
    yaw_to_equirectangular_x,
    pitch_to_equirectangular_y,
    equirectangular_x_to_yaw,
    equirectangular_y_to_pitch,
)

__all__ = [
    # Version
    "__version__",
    # Pipeline-first public API
    "PanoOCR",
    "OCREngine",
    "OCROptions",
    "DedupOptions",
    "PerspectivePreset",
    "OCRResult",
    # Image module
    "PanoramaImage",
    "PerspectiveImage",
    "PerspectiveMetadata",
    # Perspective generation API
    "generate_perspectives",
    "combine_perspectives",
    # Pre-defined perspective sets
    "DEFAULT_IMAGE_PERSPECTIVES",
    "ZOOMED_IN_IMAGE_PERSPECTIVES",
    "ZOOMED_OUT_IMAGE_PERSPECTIVES",
    # OCR models
    "BoundingBox",
    "FlatOCRResult",
    "SphereOCRResult",
    # Deduplication module
    "SphereOCRDuplicationDetectionEngine",
    # Visualization utilities
    "visualize_ocr_results",
    "visualize_sphere_ocr_results",
    # Geometry utilities
    "uv_to_yaw_pitch",
    "yaw_pitch_to_uv",
    "normalize_yaw",
    "yaw_to_equirectangular_x",
    "pitch_to_equirectangular_y",
    "equirectangular_x_to_yaw",
    "equirectangular_y_to_pitch",
]
