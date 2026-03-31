"""OCR engine implementations.

Each engine is in its own module to enable lazy loading and
optional dependency management.

Available engines:
- MacOCREngine: Uses Apple Vision Framework (macOS only)
- EasyOCREngine: Uses EasyOCR library
- PaddleOCREngine: Uses PaddleOCR library
- Florence2OCREngine: Uses Microsoft Florence-2 model (transformers + torch)
- TrOCREngine: Uses Microsoft TrOCR model (experimental)
- RapidOCREngine: Uses PP-OCRv4/v5 via ONNX Runtime
- GoogleVisionEngine: Uses Google Cloud Vision API (TEXT_DETECTION)
- Florence2MLXEngine: Uses Florence-2 via mlx-vlm (structured, with bboxes)
- GlmOCREngine: Uses GLM-OCR via mlx-vlm (unstructured)
- DotsOCREngine: Uses DOTS.OCR via mlx-vlm (unstructured)
- GeminiEngine: Uses Gemini API (unstructured)

Install the required dependencies for each engine:
- MacOCR: uv sync --extra macocr
- EasyOCR: uv sync --extra easyocr
- PaddleOCR: uv sync --extra paddleocr
- Florence-2 (torch): uv sync --extra florence2
- TrOCR: uv sync --extra trocr
- RapidOCR: uv sync --extra rapidocr
- Google Vision: uv sync --extra google-vision
- Florence-2 MLX / GLM-OCR / DOTS.OCR: uv sync --extra mlx-vlm
- Gemini: uv sync --extra gemini
"""

__all__ = [
    "MacOCREngine",
    "EasyOCREngine",
    "PaddleOCREngine",
    "Florence2OCREngine",
    "TrOCREngine",
    "RapidOCREngine",
    "GoogleVisionEngine",
    "Florence2MLXEngine",
    "GlmOCREngine",
    "DotsOCREngine",
    "GeminiEngine",
]

_LAZY_IMPORTS = {
    "MacOCREngine": (".macocr", "MacOCREngine"),
    "EasyOCREngine": (".easyocr", "EasyOCREngine"),
    "PaddleOCREngine": (".paddleocr", "PaddleOCREngine"),
    "Florence2OCREngine": (".florence2", "Florence2OCREngine"),
    "TrOCREngine": (".trocr", "TrOCREngine"),
    "RapidOCREngine": (".rapidocr_engine", "RapidOCREngine"),
    "GoogleVisionEngine": (".google_vision", "GoogleVisionEngine"),
    "Florence2MLXEngine": (".florence2_mlx", "Florence2MLXEngine"),
    "GlmOCREngine": (".glm_ocr", "GlmOCREngine"),
    "DotsOCREngine": (".dots_ocr", "DotsOCREngine"),
    "GeminiEngine": (".gemini", "GeminiEngine"),
}


def __getattr__(name: str):
    """Lazy import engines to avoid loading unnecessary dependencies."""
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
