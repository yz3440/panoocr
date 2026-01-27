# PanoOCR

OCR for equirectangular panorama images. Handles perspective projection, coordinate conversion, and text deduplication.

## Installation

```bash
# Lightweight (bring your own engine)
pip install "panoocr @ git+https://github.com/yz3440/panoocr.git"

# With MacOCR (macOS only)
pip install "panoocr[macocr]"

# With EasyOCR (cross-platform)
pip install "panoocr[easyocr]"

# With all engines
pip install "panoocr[full] @ git+https://github.com/yz3440/panoocr.git"
```

## Quick Start

```python
from panoocr import PanoOCR
from panoocr.engines.macocr import MacOCREngine

engine = MacOCREngine()
pano = PanoOCR(engine)
result = pano.recognize("panorama.jpg")
result.save_json("results.ocr.json")
```

## Demo

<video controls width="100%">
  <source src="https://github.com/user-attachments/assets/57507c48-ec88-4d4a-bf68-067eefc9d42f" type="video/mp4">
</video>

The preview tool visualizes OCR results on an interactive 3D sphere.

```bash
cd preview && python -m http.server
```

Open `http://localhost:8000` and drag in the JSON result file and panorama image.

## Next

- [Examples](examples.md) - Working scripts
- [API Reference](api/index.md) - Full documentation
