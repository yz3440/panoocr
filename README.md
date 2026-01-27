# PanoOCR

PanoOCR is a Python library for performing Optical Character Recognition (OCR) on equirectangular panorama images with automatic perspective projection and deduplication.

https://github.com/user-attachments/assets/57507c48-ec88-4d4a-bf68-067eefc9d42f

## Features

- **Multiple OCR Engines**: Support for MacOCR (Apple Vision), EasyOCR, PaddleOCR, Florence-2, and TrOCR
- **Automatic Perspective Projection**: Converts equirectangular panoramas to multiple perspective views for better OCR accuracy
- **Deduplication**: Automatically removes duplicate text detections across overlapping perspective views
- **Spherical Coordinates**: Returns OCR results in yaw/pitch coordinates that map directly to the panorama
- **Preview Tool**: Interactive 3D preview of OCR results on the panorama

## Installation

Install the base package:

```bash
pip install panoocr
```

Install with OCR engine dependencies:

```bash
# macOS (Apple Vision Framework)
pip install "panoocr[macocr]"

# EasyOCR (cross-platform)
pip install "panoocr[easyocr]"

# PaddleOCR (cross-platform)
pip install "panoocr[paddleocr]"

# Florence-2 (requires GPU recommended)
pip install "panoocr[florence2]"

# All engines (excluding platform-specific macocr)
pip install "panoocr[full]"
```

Using uv (recommended):

```bash
uv add panoocr
uv add "panoocr[macocr]"  # or other extras
```

## Quick Start

```python
from panoocr import PanoOCR
from panoocr.engines.macocr import MacOCREngine  # or other engines

# Create an OCR engine
engine = MacOCREngine()

# Create the PanoOCR pipeline
pano = PanoOCR(engine)

# Run OCR on a panorama
result = pano.recognize("panorama.jpg")

# Save results as JSON
result.save_json("results.json")

# Access individual results
for r in result.results:
    print(f"Text: {r.text}")
    print(f"Position: yaw={r.yaw}°, pitch={r.pitch}°")
    print(f"Confidence: {r.confidence}")
```

## Available OCR Engines

### MacOCREngine (macOS only)

Uses Apple's Vision Framework for fast, accurate OCR on macOS.

```python
from panoocr.engines.macocr import MacOCREngine, MacOCRLanguageCode

engine = MacOCREngine(config={
    "language_preference": [MacOCRLanguageCode.ENGLISH_US],
})
```

### EasyOCREngine

Cross-platform OCR supporting 80+ languages.

```python
from panoocr.engines.easyocr import EasyOCREngine, EasyOCRLanguageCode

engine = EasyOCREngine(config={
    "language_preference": [EasyOCRLanguageCode.ENGLISH],
    "gpu": True,
})
```

### PaddleOCREngine

PaddlePaddle-based OCR with optional V4 server model for Chinese text.

```python
from panoocr.engines.paddleocr import PaddleOCREngine, PaddleOCRLanguageCode

engine = PaddleOCREngine(config={
    "language_preference": PaddleOCRLanguageCode.CHINESE,
    "use_v4_server": True,
})
```

### Florence2OCREngine

Microsoft's Florence-2 vision-language model for OCR.

```python
from panoocr.engines.florence2 import Florence2OCREngine

engine = Florence2OCREngine(config={
    "model_id": "microsoft/Florence-2-large",
})
```

## Advanced Usage

### Custom Perspectives

```python
from panoocr import PanoOCR, PerspectivePreset, generate_perspectives

# Use a preset
pano = PanoOCR(engine, perspectives=PerspectivePreset.ZOOMED_IN)

# Or create custom perspectives
custom_perspectives = generate_perspectives(
    pixel_size=1024,
    horizontal_fov=30,
    vertical_fov=30,
    pitch_offsets=[0, 15, -15],  # Multiple rows
)
pano = PanoOCR(engine, perspectives=custom_perspectives)
```

### Multi-Scale Detection

```python
from panoocr import PanoOCR, PerspectivePreset

pano = PanoOCR(engine)

# Run OCR at multiple scales to catch both small and large text
result = pano.recognize_multi(
    "panorama.jpg",
    presets=[
        PerspectivePreset.ZOOMED_IN,
        PerspectivePreset.DEFAULT,
    ],
)
```

### Custom Deduplication Settings

```python
from panoocr import PanoOCR, DedupOptions

pano = PanoOCR(
    engine,
    dedup_options=DedupOptions(
        min_text_similarity=0.6,
        min_intersection_ratio=0.2,
    ),
)
```

### Using the Protocol for Custom Engines

You can create your own OCR engine by implementing the `OCREngine` protocol:

```python
from panoocr import OCREngine, FlatOCRResult
from PIL import Image

class MyCustomEngine:
    def recognize(self, image: Image.Image) -> list[FlatOCRResult]:
        # Your OCR implementation here
        # Return results with normalized bounding boxes (0-1 range)
        ...

# No inheritance required - just implement the method
engine = MyCustomEngine()
pano = PanoOCR(engine)
```

## Preview Tool

The package includes an interactive HTML preview tool for visualizing OCR results on the panorama. Open `preview/index.html` in a browser and drag & drop your panorama image and JSON results file.

## Output Format

OCR results are returned as `SphereOCRResult` objects with spherical coordinates:

```json
{
  "results": [
    {
      "text": "HELLO WORLD",
      "confidence": 0.95,
      "yaw": 45.0,
      "pitch": 0.0,
      "width": 10.5,
      "height": 3.2,
      "engine": "APPLE_VISION_FRAMEWORK"
    }
  ],
  "image_path": "panorama.jpg",
  "perspective_preset": "default"
}
```

- `yaw`: Horizontal angle in degrees (-180 to 180)
- `pitch`: Vertical angle in degrees (-90 to 90)
- `width`, `height`: Angular dimensions in degrees

## License

MIT License - see [LICENSE](LICENSE) for details.
