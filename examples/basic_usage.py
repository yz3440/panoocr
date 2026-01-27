#!/usr/bin/env python3
"""Basic usage example for PanoOCR.

This script demonstrates the simplest way to run OCR on a panorama image.

Usage:
    python basic_usage.py path/to/panorama.jpg

Prerequisites:
    pip install "panoocr[macocr]"  # For macOS
    # or
    pip install "panoocr[easyocr]"  # For cross-platform
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <panorama_image>")
        print("\nExample:")
        print("  python basic_usage.py panorama.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    # Import panoocr
    from panoocr import PanoOCR, PerspectivePreset

    # Try to import an OCR engine (prefer MacOCR on macOS, fall back to EasyOCR)
    engine = None

    try:
        from panoocr.engines.macocr import MacOCREngine

        print("Using MacOCR engine (Apple Vision Framework)")
        engine = MacOCREngine()
    except ImportError:
        pass

    if engine is None:
        try:
            from panoocr.engines.easyocr import EasyOCREngine

            print("Using EasyOCR engine")
            engine = EasyOCREngine()
        except ImportError:
            pass

    if engine is None:
        print("Error: No OCR engine available.")
        print("\nInstall an OCR engine with:")
        print("  pip install 'panoocr[macocr]'  # For macOS")
        print("  pip install 'panoocr[easyocr]'  # For cross-platform")
        sys.exit(1)

    # Create the PanoOCR pipeline
    pano = PanoOCR(
        engine,
        perspectives=PerspectivePreset.DEFAULT,
    )

    # Run OCR
    print(f"\nProcessing: {image_path}")
    result = pano.recognize(str(image_path))

    # Print results
    print(f"\nFound {len(result.results)} text regions:")
    print("-" * 60)

    for i, r in enumerate(result.results, 1):
        print(f"\n[{i}] {r.text}")
        print(f"    Position: yaw={r.yaw:.1f}°, pitch={r.pitch:.1f}°")
        print(f"    Size: {r.width:.1f}° x {r.height:.1f}°")
        print(f"    Confidence: {r.confidence:.2f}")

    # Save results
    output_path = image_path.with_suffix(".ocr.json")
    result.save_json(str(output_path))
    print(f"\nResults saved to: {output_path}")
    print(f"View with: preview/index.html (drag and drop image + JSON)")


if __name__ == "__main__":
    main()
