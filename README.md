# PanoOCR

PanoOCR is a Python library for performing Optical Character Recognition (OCR) on equirectangular panorama images with automatic perspective projection and deduplication.

https://github.com/user-attachments/assets/57507c48-ec88-4d4a-bf68-067eefc9d42f

## Features

- **Multiple OCR Engines**: Support for MacOCR (Apple Vision), RapidOCR, EasyOCR, PaddleOCR, Florence-2, Google Cloud Vision, Gemini, and more
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

# RapidOCR (PP-OCRv4/v5 via ONNX Runtime, cross-platform)
pip install "panoocr[rapidocr]"

# EasyOCR (cross-platform)
pip install "panoocr[easyocr]"

# PaddleOCR (cross-platform)
pip install "panoocr[paddleocr]"

# Florence-2 via transformers + torch (requires GPU recommended)
pip install "panoocr[florence2]"

# MLX VLM engines: Florence-2 MLX, GLM-OCR, DOTS.OCR (macOS Apple Silicon)
pip install "panoocr[mlx-vlm]"

# Google Cloud Vision API (requires API key)
pip install "panoocr[google-vision]"

# Gemini API (requires API key)
pip install "panoocr[gemini]"

# Cross-platform local engines + visualization
# (excludes macOS-only macocr, Apple Silicon mlx-vlm, cloud APIs, and experimental trocr)
pip install "panoocr[full]"
```

Using uv (recommended):

```bash
uv add panoocr
uv sync --extra macocr      # or other extras
uv sync --extra rapidocr
uv sync --extra mlx-vlm     # Florence-2 MLX, GLM-OCR, DOTS.OCR
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

### Structured engines (return per-word bounding boxes)

#### MacOCREngine (macOS only)

Uses Apple's Vision Framework for fast, accurate OCR on macOS.

```python
from panoocr.engines.macocr import MacOCREngine

engine = MacOCREngine()
```

#### RapidOCREngine

PaddleOCR PP-OCRv4/v5 models via ONNX Runtime. Supports both v4 (2023) and v5 (2025) models, multilingual including CJK.

```python
from panoocr.engines.rapidocr_engine import RapidOCREngine

engine_v4 = RapidOCREngine()                                     # default: PP-OCRv4
engine_v5 = RapidOCREngine(config={"ocr_version": "PP-OCRv5"})   # PP-OCRv5
```

#### EasyOCREngine

Cross-platform OCR supporting 80+ languages.

```python
from panoocr.engines.easyocr import EasyOCREngine

engine = EasyOCREngine(config={"language_preference": ["en"], "gpu": True})
```

#### PaddleOCREngine

PaddlePaddle-based OCR supporting multiple languages with automatic model management.

```python
from panoocr.engines.paddleocr import PaddleOCREngine

engine = PaddleOCREngine()
```

#### GoogleVisionEngine

Google Cloud Vision API (`TEXT_DETECTION`). Requires `GOOGLE_VISION_API_KEY` in environment or `.env`.

```python
from panoocr.engines.google_vision import GoogleVisionEngine

engine = GoogleVisionEngine()
```

#### Florence2OCREngine (transformers + torch)

Microsoft's Florence-2 vision-language model via transformers.

```python
from panoocr.engines.florence2 import Florence2OCREngine

engine = Florence2OCREngine()
```

#### Florence2MLXEngine (mlx-vlm, macOS Apple Silicon)

Florence-2 via mlx-vlm with `<OCR_WITH_REGION>` for structured quad-box output. The only VLM engine that returns per-word bounding boxes.

```python
from panoocr.engines.florence2_mlx import Florence2MLXEngine

engine = Florence2MLXEngine()
```

### Unstructured engines (return text without bounding boxes)

These engines return text only. Each detection gets a full-image bounding box for crop-level attribution in the panoocr pipeline.

#### GeminiEngine

Google Gemini API. Supports multiple model variants. Requires `GOOGLE_GEMINI_API_KEY` in environment or `.env`.

```python
from panoocr.engines.gemini import GeminiEngine

engine_flash = GeminiEngine(config={"model": "gemini-2.5-flash"})
engine_pro = GeminiEngine(config={"model": "gemini-2.5-pro"})
```

#### GlmOCREngine (mlx-vlm, macOS Apple Silicon)

GLM-OCR (0.9B) via mlx-vlm. Document-focused VLM -- limited effectiveness on scene text.

```python
from panoocr.engines.glm_ocr import GlmOCREngine

engine = GlmOCREngine()
```

#### DotsOCREngine (mlx-vlm, macOS Apple Silicon)

DOTS.OCR (2.9B) via mlx-vlm. Document layout parser -- limited effectiveness on scene text.

```python
from panoocr.engines.dots_ocr import DotsOCREngine

engine = DotsOCREngine()
```

#### TrOCREngine (experimental)

Microsoft's TrOCR transformer-based single-line OCR. Does not detect text regions -- treats the entire image as one text line. Experimental; consider other engines for panorama OCR.

```python
from panoocr.engines.trocr import TrOCREngine

engine = TrOCREngine()
```

## Advanced Usage

### Custom Perspectives

```python
from panoocr import PanoOCR, PerspectivePreset, generate_perspectives

# Use a preset
pano = PanoOCR(engine, perspectives=PerspectivePreset.ZOOMED_IN)

# Or create custom perspectives
custom_perspectives = generate_perspectives(
    fov=30,              # Horizontal FOV in degrees
    resolution=1024,     # Pixel width/height
    overlap=0.5,         # 50% overlap between adjacent views
    pitch_angles=[0, 15, -15],  # Multiple rows
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
