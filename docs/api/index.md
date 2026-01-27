# API Reference

## Client API

The `PanoOCR` class is the main entry point.

::: panoocr.api.client.PanoOCR
    options:
      show_root_heading: true

::: panoocr.api.models.OCRResult
    options:
      show_root_heading: true

::: panoocr.api.models.PerspectivePreset
    options:
      show_root_heading: true

::: panoocr.api.models.OCROptions
    options:
      show_root_heading: true

::: panoocr.api.models.DedupOptions
    options:
      show_root_heading: true

## Module Structure

```text
panoocr/
├── api/              # Client API
│   ├── client.py     # PanoOCR
│   └── models.py     # OCRResult, options, OCREngine protocol
├── engines/          # OCR engines (lazily imported)
│   ├── macocr.py     # MacOCREngine (requires [macocr])
│   ├── easyocr.py    # EasyOCREngine (requires [easyocr])
│   ├── paddleocr.py  # PaddleOCREngine (requires [paddleocr])
│   ├── florence2.py  # Florence2OCREngine (requires [florence2])
│   └── trocr.py      # TrOCREngine (requires [trocr])
├── ocr/              # OCR result models
│   ├── models.py     # FlatOCRResult, SphereOCRResult
│   └── utils.py      # Visualization (requires [viz])
├── dedup/            # Deduplication
│   └── detection.py  # SphereOCRDuplicationDetectionEngine
├── image/            # Panorama handling
│   ├── models.py     # PanoramaImage, PerspectiveMetadata
│   └── perspectives.py  # Presets, generate_perspectives()
└── geometry.py       # Coordinate conversion utilities
```

## Submodules

- [Engines](engines.md) - `OCREngine` protocol and built-in engines
- [Image](image.md) - Panorama and perspective classes
- [OCR Models](ocr.md) - OCR result types
- [Deduplication](dedup.md) - Text deduplication
- [Geometry](geometry.md) - Coordinate conversion
- [Visualization](visualization.md) - OCR visualization (requires `[viz]`)
