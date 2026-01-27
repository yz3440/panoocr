"""Visualization utilities for OCR results.

This module provides functions for visualizing OCR results on images.
Some functions require optional dependencies (opencv-python, scipy).

Install visualization dependencies with: pip install "panoocr[viz]"
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from .models import FlatOCRResult, SphereOCRResult


def _check_viz_dependencies():
    """Check if visualization dependencies are installed."""
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        missing.append("scipy")

    if missing:
        raise ImportError(
            f"Visualization dependencies not installed: {', '.join(missing)}\n\n"
            "Install with:\n"
            "  pip install 'panoocr[viz]'"
        )


def visualize_ocr_results(
    image: Image.Image,
    ocr_results: List[FlatOCRResult],
    font_size: int = 16,
    highlight_color: str = "red",
    stroke_width: int = 2,
) -> Image.Image:
    """Visualize flat OCR results on an image.

    Draws bounding boxes and labels on a copy of the image.

    Args:
        image: The image to visualize OCR results on.
        ocr_results: List of FlatOCRResult objects to visualize.
        font_size: Font size for labels.
        highlight_color: Color for boxes and text.
        stroke_width: Width of bounding box lines.

    Returns:
        Copy of the image with OCR results visualized.
    """
    # Make a copy to avoid modifying the original
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    width, height = result_image.size

    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        # Older Pillow versions don't support size parameter
        font = ImageFont.load_default()

    for ocr_result in ocr_results:
        # Draw bounding box
        bbox = ocr_result.bounding_box
        draw.rectangle(
            [
                bbox.left * width,
                bbox.top * height,
                bbox.right * width,
                bbox.bottom * height,
            ],
            outline=highlight_color,
            width=3,
        )

        # Draw text label
        draw.text(
            (
                bbox.left * width,
                bbox.top * height - font_size - stroke_width,
            ),
            ocr_result.text,
            fill=highlight_color,
            stroke_fill="white",
            stroke_width=stroke_width,
            font=font,
        )

        # Draw confidence score
        draw.text(
            (
                bbox.left * width,
                bbox.bottom * height + stroke_width,
            ),
            f"{ocr_result.confidence:.2f}",
            fill=highlight_color,
            stroke_fill="white",
            stroke_width=stroke_width,
            font=font,
        )

    return result_image


def visualize_sphere_ocr_results(
    image: Image.Image,
    ocr_results: List[SphereOCRResult],
    font_size: int = 16,
    highlight_color: str = "red",
    stroke_width: int = 2,
    inplace: bool = False,
) -> Image.Image:
    """Visualize spherical OCR results on an equirectangular image.

    Projects OCR result labels back onto the panorama image.
    This is SLOW and should only be used for debugging purposes.

    Args:
        image: The equirectangular panorama image.
        ocr_results: List of SphereOCRResult objects to visualize.
        font_size: Font size for labels (unused, size is automatic).
        highlight_color: Color for boxes and text.
        stroke_width: Width of text stroke.
        inplace: If True, modify the input image directly (faster).

    Returns:
        Image with OCR results visualized.

    Raises:
        ImportError: If visualization dependencies are not installed.
    """
    _check_viz_dependencies()

    import numpy as np
    from scipy.ndimage import map_coordinates

    # Convert image to RGBA for alpha compositing
    image = image.convert("RGBA")

    def get_ocr_result_image(ocr_result: SphereOCRResult) -> Image.Image:
        """Create an image for a single OCR result."""
        PIXEL_PER_DEGREE = 300

        text_image = Image.new(
            "RGBA",
            (
                int(ocr_result.width * PIXEL_PER_DEGREE),
                int(ocr_result.height * PIXEL_PER_DEGREE),
            ),
            (255, 255, 255, 0),
        )

        draw = ImageDraw.Draw(text_image)

        try:
            font = ImageFont.load_default(
                size=int(ocr_result.height * PIXEL_PER_DEGREE * 0.2)
            )
        except TypeError:
            font = ImageFont.load_default()

        # Draw bounding box
        draw.rectangle(
            [0, 0, text_image.width, text_image.height],
            outline=highlight_color,
            width=3,
            fill=(255, 255, 255, 0),
        )

        # Draw text
        draw.text(
            (text_image.width / 2, text_image.height / 2),
            ocr_result.text,
            fill=highlight_color,
            anchor="mm",
            stroke_fill="white",
            stroke_width=stroke_width,
            font=font,
        )

        return text_image

    def place_ocr_result_on_panorama(
        panorama_array: np.ndarray, ocr_result: SphereOCRResult
    ) -> np.ndarray:
        """Project an OCR result onto the panorama."""
        ocr_result_image = np.array(get_ocr_result_image(ocr_result))

        pano_height, pano_width = panorama_array.shape[:2]
        ocr_height, ocr_width = ocr_result_image.shape[:2]

        yaw_rad = np.radians(-ocr_result.yaw)
        pitch_rad = np.radians(ocr_result.pitch)
        width_rad = np.radians(ocr_result.width)
        height_rad = np.radians(ocr_result.height)

        # Create coordinate mappings for the panorama
        y_pano, x_pano = np.mgrid[0:pano_height, 0:pano_width]

        # Convert panorama coordinates to spherical coordinates
        lon = (x_pano / pano_width - 0.5) * 2 * np.pi
        lat = (0.5 - y_pano / pano_height) * np.pi

        # Calculate 3D coordinates on the unit sphere
        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = np.cos(lat) * np.cos(lon)

        # Combine rotation matrices
        sin_yaw, cos_yaw = np.sin(yaw_rad), np.cos(yaw_rad)
        sin_pitch, cos_pitch = np.sin(pitch_rad), np.cos(pitch_rad)

        # Apply rotation
        x_rot = cos_yaw * x + sin_yaw * z
        y_rot = sin_pitch * sin_yaw * x + cos_pitch * y - sin_pitch * cos_yaw * z
        z_rot = -cos_pitch * sin_yaw * x + sin_pitch * y + cos_pitch * cos_yaw * z

        # Project onto the plane
        epsilon = 1e-8
        x_proj = x_rot / (z_rot + epsilon)
        y_proj = y_rot / (z_rot + epsilon)

        # Scale and shift to image coordinates
        x_img = (x_proj / np.tan(width_rad / 2) + 1) * ocr_width / 2
        y_img = (-y_proj / np.tan(height_rad / 2) + 1) * ocr_height / 2

        # Create mask for valid coordinates
        mask = (
            (x_img >= 0)
            & (x_img < ocr_width)
            & (y_img >= 0)
            & (y_img < ocr_height)
            & (z_rot > 0)
        )

        # Sample from the OCR result image
        warped_channels = []
        for channel in range(ocr_result_image.shape[2]):
            warped_channel = map_coordinates(
                ocr_result_image[:, :, channel],
                [y_img, x_img],
                order=1,
                mode="constant",
                cval=0,
            )
            warped_channels.append(warped_channel)

        warped_image = np.stack(warped_channels, axis=-1)

        if not inplace:
            result = panorama_array.copy()
        else:
            result = panorama_array

        # Apply alpha compositing
        alpha = warped_image[:, :, 3] / 255.0
        for c in range(3):  # RGB channels
            result[:, :, c] = (
                result[:, :, c] * (1 - alpha * mask)
                + warped_image[:, :, c] * (alpha * mask)
            )

        # Update alpha channel
        result[:, :, 3] = np.maximum(result[:, :, 3], warped_image[:, :, 3] * mask)

        return result

    import numpy as np

    new_image = np.array(image)

    for ocr_result in ocr_results:
        if inplace:
            place_ocr_result_on_panorama(new_image, ocr_result)
        else:
            new_image = place_ocr_result_on_panorama(new_image, ocr_result)

    new_image = Image.fromarray(new_image)

    # Convert back to RGB
    new_image = new_image.convert("RGB")

    return new_image
