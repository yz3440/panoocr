"""OCR result models with coordinate transformation support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


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

    def _uv_to_yaw_pitch(
        self, horizontal_fov: float, vertical_fov: float, u: float, v: float
    ) -> tuple[float, float]:
        """Convert UV coordinate to yaw and pitch using camera parameters.

        All parameters are in degrees.

        Args:
            horizontal_fov: Horizontal field of view of the camera.
            vertical_fov: Vertical field of view of the camera.
            u: Horizontal coordinate on flat image (0-1).
            v: Vertical coordinate on flat image (0-1).

        Returns:
            Tuple of (yaw, pitch) in degrees.
        """
        if horizontal_fov <= 0 or vertical_fov <= 0:
            raise ValueError("FOV must be positive")

        # Translate origin to center of image
        u = u - 0.5
        v = 0.5 - v

        yaw = math.atan2(2 * u * math.tan(math.radians(horizontal_fov) / 2), 1)
        pitch = math.atan2(2 * v * math.tan(math.radians(vertical_fov) / 2), 1)

        return math.degrees(yaw), math.degrees(pitch)

    def to_sphere(
        self,
        horizontal_fov: float,
        vertical_fov: float,
        yaw_offset: float,
        pitch_offset: float,
    ) -> "SphereOCRResult":
        """Convert to spherical OCR result using camera parameters.

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

        # Calculate center point
        center_x = (self.bounding_box.left + self.bounding_box.right) * 0.5
        center_y = (self.bounding_box.top + self.bounding_box.bottom) * 0.5

        center_yaw, center_pitch = self._uv_to_yaw_pitch(
            horizontal_fov, vertical_fov, center_x, center_y
        )

        # Calculate corners for width/height
        left_yaw, top_pitch = self._uv_to_yaw_pitch(
            horizontal_fov, vertical_fov, self.bounding_box.left, self.bounding_box.top
        )

        right_yaw, bottom_pitch = self._uv_to_yaw_pitch(
            horizontal_fov,
            vertical_fov,
            self.bounding_box.right,
            self.bounding_box.bottom,
        )

        width = right_yaw - left_yaw
        height = top_pitch - bottom_pitch

        return SphereOCRResult(
            text=self.text,
            confidence=self.confidence,
            yaw=center_yaw + yaw_offset,
            pitch=center_pitch + pitch_offset,
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
