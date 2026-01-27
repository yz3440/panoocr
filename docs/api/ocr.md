# OCR Models

Data classes for OCR results in different coordinate systems.

## BoundingBox

::: panoocr.ocr.models.BoundingBox
    options:
      show_root_heading: true

## FlatOCRResult

OCR result from a flat (perspective) image with normalized bounding box coordinates.

::: panoocr.ocr.models.FlatOCRResult
    options:
      show_root_heading: true
      members:
        - text
        - confidence
        - bounding_box
        - engine
        - to_dict
        - from_dict
        - to_sphere

## SphereOCRResult

OCR result in spherical (panorama) coordinates.

::: panoocr.ocr.models.SphereOCRResult
    options:
      show_root_heading: true
      members:
        - text
        - confidence
        - yaw
        - pitch
        - width
        - height
        - engine
        - to_dict
        - from_dict
