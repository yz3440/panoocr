"""OCR result models with coordinate transformation support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from panoocr.geometry import perspective_to_sphere


@dataclass
class BoundingBox:
    """Normalized bounding box with coordinates in 0-1 range.

    Coordinates are relative to image dimensions:
    - (0, 0) is top-left
    - (1, 1) is bottom-right

    Attributes:
        left: Distance from left edge (0-1).
        top: Distance from top edge (0-1).
        right: Distance from left edge to right side (0-1).
        bottom: Distance from top edge to bottom side (0-1).
        width: Box width (0-1).
        height: Box height (0-1).
    """

    left: float
    top: float
    right: float
    bottom: float
    width: float
    height: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoundingBox":
        """Create from dictionary."""
        return cls(
            left=data["left"],
            top=data["top"],
            right=data["right"],
            bottom=data["bottom"],
            width=data["width"],
            height=data["height"],
        )


@dataclass
class FlatOCRResult:
    """OCR result from a flat (perspective) image.

    Attributes:
        text: Recognized text content.
        confidence: Recognition confidence (0-1).
        bounding_box: Normalized bounding box in image coordinates.
        engine: Name of the OCR engine used.
    """

    text: str
    confidence: float
    bounding_box: BoundingBox
    engine: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict(),
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlatOCRResult":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            bounding_box=BoundingBox.from_dict(data["bounding_box"]),
            engine=data.get("engine"),
        )

    def to_sphere(
        self,
        horizontal_fov: float,
        vertical_fov: float,
        yaw_offset: float,
        pitch_offset: float,
    ) -> "SphereOCRResult":
        """Convert to spherical OCR result using camera parameters.

        Uses proper 3D rotation via perspective_to_sphere() to correctly
        transform bounding box coordinates from perspective image space to
        world spherical coordinates. This accounts for the coupling between
        yaw and pitch that occurs when the camera has a non-zero pitch offset.

        All parameters are in degrees.

        Args:
            horizontal_fov: Horizontal field of view of the camera.
            vertical_fov: Vertical field of view of the camera.
            yaw_offset: Horizontal offset of the camera.
            pitch_offset: Vertical offset of the camera.

        Returns:
            SphereOCRResult with spherical coordinates.
        """
        if horizontal_fov <= 0 or vertical_fov <= 0:
            raise ValueError("FOV must be positive")

        # Convert center point using proper 3D rotation
        center_x = (self.bounding_box.left + self.bounding_box.right) * 0.5
        center_y = (self.bounding_box.top + self.bounding_box.bottom) * 0.5

        center_yaw, center_pitch = perspective_to_sphere(
            center_x, center_y,
            horizontal_fov, vertical_fov,
            yaw_offset, pitch_offset,
        )

        # Convert all four corners using proper 3D rotation
        tl_yaw, tl_pitch = perspective_to_sphere(
            self.bounding_box.left, self.bounding_box.top,
            horizontal_fov, vertical_fov, yaw_offset, pitch_offset,
        )
        tr_yaw, tr_pitch = perspective_to_sphere(
            self.bounding_box.right, self.bounding_box.top,
            horizontal_fov, vertical_fov, yaw_offset, pitch_offset,
        )
        bl_yaw, bl_pitch = perspective_to_sphere(
            self.bounding_box.left, self.bounding_box.bottom,
            horizontal_fov, vertical_fov, yaw_offset, pitch_offset,
        )
        br_yaw, br_pitch = perspective_to_sphere(
            self.bounding_box.right, self.bounding_box.bottom,
            horizontal_fov, vertical_fov, yaw_offset, pitch_offset,
        )

        # Compute angular width and height from world-space corners
        corner_yaws = [tl_yaw, tr_yaw, bl_yaw, br_yaw]
        corner_pitches = [tl_pitch, tr_pitch, bl_pitch, br_pitch]

        # Handle yaw wrap-around at ±180° boundary
        yaw_range = max(corner_yaws) - min(corner_yaws)
        if yaw_range > 180:
            shifted_yaws = [y + 360 if y < 0 else y for y in corner_yaws]
            width = max(shifted_yaws) - min(shifted_yaws)
        else:
            width = yaw_range

        height = max(corner_pitches) - min(corner_pitches)

        return SphereOCRResult(
            text=self.text,
            confidence=self.confidence,
            yaw=center_yaw,
            pitch=center_pitch,
            width=width,
            height=height,
            engine=self.engine,
        )


@dataclass
class SphereOCRResult:
    """OCR result in spherical (panorama) coordinates.

    Attributes:
        text: Recognized text content.
        confidence: Recognition confidence (0-1).
        yaw: Horizontal angle in degrees (-180 to 180).
        pitch: Vertical angle in degrees (-90 to 90).
        width: Angular width in degrees.
        height: Angular height in degrees.
        engine: Name of the OCR engine used.
    """

    text: str
    confidence: float
    yaw: float
    pitch: float
    width: float
    height: float
    engine: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "width": self.width,
            "height": self.height,
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SphereOCRResult":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            yaw=data["yaw"],
            pitch=data["pitch"],
            width=data["width"],
            height=data["height"],
            engine=data.get("engine"),
        )
