# Engines

PanoOCR uses dependency injection for OCR engines. Provide any object with a matching `recognize()` method.

## OCREngine Protocol

::: panoocr.api.models.OCREngine
    options:
      show_root_heading: true

## Structured Engines

Structured engines return per-word bounding boxes, enabling geographic text indexing.

### MacOCREngine

Uses Apple's Vision Framework for fast, accurate OCR on macOS. Requires the `[macocr]` extra.

```bash
pip install "panoocr[macocr]"
```

::: panoocr.engines.macocr.MacOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### RapidOCREngine

PaddleOCR PP-OCRv4/v5 models via ONNX Runtime. Bypasses PaddlePaddle framework (avoids Apple Silicon issues). Supports v4 and v5 model versions. Requires the `[rapidocr]` extra.

```bash
pip install "panoocr[rapidocr]"
```

::: panoocr.engines.rapidocr_engine.RapidOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### EasyOCREngine

Cross-platform OCR supporting 80+ languages. Requires the `[easyocr]` extra.

```bash
pip install "panoocr[easyocr]"
```

::: panoocr.engines.easyocr.EasyOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### PaddleOCREngine

PaddlePaddle-based OCR supporting multiple languages with automatic model management. Requires the `[paddleocr]` extra (includes both `paddleocr` and `paddlepaddle`).

```bash
pip install "panoocr[paddleocr]"
```

::: panoocr.engines.paddleocr.PaddleOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### GoogleVisionEngine

Google Cloud Vision API (`TEXT_DETECTION`). Uses REST API with an API key. Requires the `[google-vision]` extra and `GOOGLE_VISION_API_KEY` environment variable.

```bash
pip install "panoocr[google-vision]"
```

::: panoocr.engines.google_vision.GoogleVisionEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### Florence2OCREngine

Microsoft's Florence-2 vision-language model via transformers + torch. Requires the `[florence2]` extra.

```bash
pip install "panoocr[florence2]"
```

::: panoocr.engines.florence2.Florence2OCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### Florence2MLXEngine

Florence-2 via mlx-vlm with `<OCR_WITH_REGION>` structured output. The only VLM engine that returns per-word bounding boxes. Requires macOS Apple Silicon with the `[mlx-vlm]` extra, plus `torch` and `torchvision`.

```bash
pip install "panoocr[mlx-vlm]" torch torchvision
```

::: panoocr.engines.florence2_mlx.Florence2MLXEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

## Unstructured Engines

Unstructured engines return text without bounding boxes. Each detection gets a full-image bounding box for crop-level attribution in the panoocr pipeline.

### GeminiEngine

Google Gemini API (Gemini 2.5 Flash / Pro). Requires the `[gemini]` extra and `GOOGLE_GEMINI_API_KEY` environment variable.

```bash
pip install "panoocr[gemini]"
```

::: panoocr.engines.gemini.GeminiEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### GlmOCREngine

GLM-OCR (0.9B) via mlx-vlm. Document-focused VLM with limited effectiveness on scene text. Requires macOS Apple Silicon with the `[mlx-vlm]` extra.

```bash
pip install "panoocr[mlx-vlm]" torch
```

::: panoocr.engines.glm_ocr.GlmOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### DotsOCREngine

DOTS.OCR (2.9B) via mlx-vlm. Document layout parser with limited effectiveness on scene text. Requires macOS Apple Silicon with the `[mlx-vlm]` extra.

```bash
pip install "panoocr[mlx-vlm]" torch
```

::: panoocr.engines.dots_ocr.DotsOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

### TrOCREngine (experimental)

Microsoft's TrOCR transformer-based OCR. Requires the `[trocr]` extra.

```bash
pip install "panoocr[trocr]"
```

::: panoocr.engines.trocr.TrOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

## Custom Engines

Any class with a compatible `recognize()` method works:

```python
from panoocr import PanoOCR, FlatOCRResult, BoundingBox
from PIL import Image

class MyEngine:
    def recognize(self, image: Image.Image) -> list[FlatOCRResult]:
        # Return list of FlatOCRResult with normalized bounding boxes (0-1)
        return [
            FlatOCRResult(
                text="Hello",
                confidence=0.95,
                bounding_box=BoundingBox(
                    left=0.1, top=0.2, right=0.4, bottom=0.3,
                    width=0.3, height=0.1
                ),
                engine="my_engine",
            )
        ]

pano = PanoOCR(engine=MyEngine())
```
