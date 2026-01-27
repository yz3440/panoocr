# PanoOCR Examples

This directory contains example scripts demonstrating how to use PanoOCR.

## Prerequisites

Install panoocr with your preferred OCR engine:

```bash
# For macOS
pip install "panoocr[macocr]"

# For cross-platform
pip install "panoocr[easyocr]"

# For all engines
pip install "panoocr[full]"
```

## Examples

### Basic Usage (`basic_usage.py`)

Demonstrates the simplest way to run OCR on a panorama image.

```bash
python basic_usage.py path/to/panorama.jpg
```

### Multi-Engine Comparison (`multi_engine.py`)

Compares results from different OCR engines on the same panorama.

```bash
python multi_engine.py path/to/panorama.jpg
```

## Output

Results are saved as JSON files that can be viewed with the preview tool
at `preview/index.html`. Drag and drop both the panorama image and the
JSON results file to visualize the OCR detections.
