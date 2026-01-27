#!/usr/bin/env python3
"""Multi-engine comparison example for PanoOCR.

This script compares OCR results from different engines on the same panorama.

Usage:
    python multi_engine.py path/to/panorama.jpg

Prerequisites:
    pip install "panoocr[full]"  # Install all engines
"""

import sys
from pathlib import Path
from typing import Dict, Any


def get_available_engines() -> Dict[str, Any]:
    """Get all available OCR engines."""
    engines = {}

    try:
        from panoocr.engines.macocr import MacOCREngine

        engines["macocr"] = MacOCREngine()
    except ImportError:
        print("MacOCR not available (requires macOS)")

    try:
        from panoocr.engines.easyocr import EasyOCREngine

        engines["easyocr"] = EasyOCREngine()
    except ImportError:
        print("EasyOCR not available (pip install 'panoocr[easyocr]')")

    try:
        from panoocr.engines.paddleocr import PaddleOCREngine

        engines["paddleocr"] = PaddleOCREngine()
    except ImportError:
        print("PaddleOCR not available (pip install 'panoocr[paddleocr]')")

    try:
        from panoocr.engines.florence2 import Florence2OCREngine

        engines["florence2"] = Florence2OCREngine()
    except ImportError:
        print("Florence2 not available (pip install 'panoocr[florence2]')")

    return engines


def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_engine.py <panorama_image>")
        print("\nExample:")
        print("  python multi_engine.py panorama.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    from panoocr import PanoOCR, PerspectivePreset

    # Get available engines
    print("Detecting available OCR engines...")
    engines = get_available_engines()

    if not engines:
        print("\nError: No OCR engines available.")
        print("Install engines with: pip install 'panoocr[full]'")
        sys.exit(1)

    print(f"\nAvailable engines: {', '.join(engines.keys())}")
    print("=" * 60)

    # Run OCR with each engine
    for name, engine in engines.items():
        print(f"\n[{name.upper()}]")
        print("-" * 40)

        pano = PanoOCR(
            engine,
            perspectives=PerspectivePreset.DEFAULT,
        )

        result = pano.recognize(str(image_path), show_progress=True)

        print(f"Found {len(result.results)} text regions")

        # Show first 5 results
        for i, r in enumerate(result.results[:5], 1):
            text_preview = r.text[:40] + "..." if len(r.text) > 40 else r.text
            print(f"  {i}. {text_preview} (conf: {r.confidence:.2f})")

        if len(result.results) > 5:
            print(f"  ... and {len(result.results) - 5} more")

        # Save results
        output_path = image_path.with_suffix(f".{name}.json")
        result.save_json(str(output_path))
        print(f"Saved: {output_path}")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("\nView results with: preview/index.html")
    print("Drag and drop the panorama image + any JSON results file")


if __name__ == "__main__":
    main()
