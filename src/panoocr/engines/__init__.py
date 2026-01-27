"""OCR engine implementations.

Each engine is in its own module to enable lazy loading and
optional dependency management.

Available engines:
- MacOCREngine: Uses Apple Vision Framework (macOS only)
- EasyOCREngine: Uses EasyOCR library
- PaddleOCREngine: Uses PaddleOCR library
- Florence2OCREngine: Uses Microsoft Florence-2 model
- TrOCREngine: Uses Microsoft TrOCR model (experimental)

Install the required dependencies for each engine:
- MacOCR: pip install "panoocr[macocr]"
- EasyOCR: pip install "panoocr[easyocr]"
- PaddleOCR: pip install "panoocr[paddleocr]"
- Florence-2: pip install "panoocr[florence2]"
- TrOCR: pip install "panoocr[trocr]"
"""

# Engines are imported lazily to avoid requiring all dependencies
__all__ = [
    "MacOCREngine",
    "EasyOCREngine",
    "PaddleOCREngine",
    "Florence2OCREngine",
    "TrOCREngine",
]


def __getattr__(name: str):
    """Lazy import engines to avoid loading unnecessary dependencies."""
    if name == "MacOCREngine":
        from .macocr import MacOCREngine
        return MacOCREngine
    elif name == "EasyOCREngine":
        from .easyocr import EasyOCREngine
        return EasyOCREngine
    elif name == "PaddleOCREngine":
        from .paddleocr import PaddleOCREngine
        return PaddleOCREngine
    elif name == "Florence2OCREngine":
        from .florence2 import Florence2OCREngine
        return Florence2OCREngine
    elif name == "TrOCREngine":
        from .trocr import TrOCREngine
        return TrOCREngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
