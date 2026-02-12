# Engines

PanoOCR uses dependency injection for OCR engines. Provide any object with a matching `recognize()` method.

## OCREngine Protocol

::: panoocr.api.models.OCREngine
    options:
      show_root_heading: true

## MacOCREngine

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

## EasyOCREngine

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

## PaddleOCREngine

PaddlePaddle-based OCR supporting multiple languages with automatic model management. Uses PP-OCRv5 by default. Requires the `[paddleocr]` extra (includes both `paddleocr` and `paddlepaddle`).

```bash
pip install "panoocr[paddleocr]"
```

::: panoocr.engines.paddleocr.PaddleOCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

## Florence2OCREngine

Microsoft's Florence-2 vision-language model for OCR. Requires the `[florence2]` extra.

```bash
pip install "panoocr[florence2]"
```

::: panoocr.engines.florence2.Florence2OCREngine
    options:
      show_root_heading: true
      members:
        - __init__
        - recognize

## TrOCREngine

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
