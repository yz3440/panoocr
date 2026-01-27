"""Panorama and perspective image models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image
import py360convert


@dataclass
class PerspectiveMetadata:
    """Metadata for a perspective projection from an equirectangular panorama.

    Attributes:
        pixel_width: Width of the perspective image in pixels.
        pixel_height: Height of the perspective image in pixels.
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.
        yaw_offset: Horizontal rotation offset in degrees (-180 to 180).
        pitch_offset: Vertical rotation offset in degrees (-90 to 90).
    """

    pixel_width: int
    pixel_height: int
    horizontal_fov: float
    vertical_fov: float
    yaw_offset: float
    pitch_offset: float

    def __str__(self) -> str:
        return (
            f"PerspectiveMetadata(pixel_width={self.pixel_width}, "
            f"pixel_height={self.pixel_height}, "
            f"horizontal_fov={self.horizontal_fov}, "
            f"vertical_fov={self.vertical_fov}, "
            f"yaw_offset={self.yaw_offset}, "
            f"pitch_offset={self.pitch_offset})"
        )

    def to_file_suffix(self) -> str:
        """Generate a unique file suffix for this perspective configuration."""
        return (
            f"{self.pixel_width}_{self.pixel_height}_"
            f"{self.horizontal_fov}_{self.vertical_fov}_"
            f"{self.yaw_offset}_{self.pitch_offset}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "horizontal_fov": self.horizontal_fov,
            "vertical_fov": self.vertical_fov,
            "yaw_offset": self.yaw_offset,
            "pitch_offset": self.pitch_offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerspectiveMetadata":
        """Create from dictionary."""
        return cls(
            pixel_width=data["pixel_width"],
            pixel_height=data["pixel_height"],
            horizontal_fov=data["horizontal_fov"],
            vertical_fov=data["vertical_fov"],
            yaw_offset=data["yaw_offset"],
            pitch_offset=data["pitch_offset"],
        )


class PerspectiveImage:
    """A perspective projection from an equirectangular panorama.

    Attributes:
        panorama_id: Identifier for the source panorama.
        perspective_metadata: Configuration for the perspective projection.
        perspective_image: The generated perspective image as PIL Image.
        perspective_image_array: The generated perspective image as numpy array.
    """

    def __init__(
        self,
        panorama_id: str,
        source_panorama_image_array: np.ndarray,
        perspective_metadata: PerspectiveMetadata,
    ):
        """Initialize a perspective image from a panorama.

        Args:
            panorama_id: Identifier for the source panorama.
            source_panorama_image_array: The equirectangular panorama as numpy array.
            perspective_metadata: Configuration for the perspective projection.
        """
        self.source_panorama_image_array = source_panorama_image_array
        self.panorama_id = panorama_id
        self.perspective_metadata = perspective_metadata

        # Generate perspective projection using py360convert
        self.perspective_image_array = py360convert.e2p(
            e_img=self.source_panorama_image_array,
            fov_deg=(
                self.perspective_metadata.horizontal_fov,
                self.perspective_metadata.vertical_fov,
            ),
            u_deg=self.perspective_metadata.yaw_offset,
            v_deg=self.perspective_metadata.pitch_offset,
            out_hw=(
                self.perspective_metadata.pixel_height,
                self.perspective_metadata.pixel_width,
            ),
            in_rot_deg=0,
            mode="bilinear",
        )

        self.perspective_image = Image.fromarray(self.perspective_image_array)

    def get_perspective_metadata(self) -> PerspectiveMetadata:
        """Get the perspective metadata."""
        return self.perspective_metadata

    def get_perspective_image_array(self) -> np.ndarray:
        """Get the perspective image as numpy array."""
        return self.perspective_image_array

    def get_perspective_image(self) -> Image.Image:
        """Get the perspective image as PIL Image."""
        return self.perspective_image


class PanoramaImage:
    """An equirectangular panorama image.

    Attributes:
        panorama_id: Identifier for this panorama.
        loaded_image: The panorama as PIL Image.
        loaded_image_array: The panorama as numpy array.
    """

    def __init__(
        self,
        panorama_id: str,
        image: Union[str, Image.Image, np.ndarray],
    ):
        """Initialize a panorama image.

        Args:
            panorama_id: Identifier for this panorama.
            image: Path to image file, PIL Image, or numpy array.

        Raises:
            ValueError: If image type is not supported.
        """
        self.panorama_id = panorama_id

        if isinstance(image, str):
            self.loaded_image = Image.open(image)
            self.loaded_image_array = np.array(self.loaded_image)
        elif isinstance(image, Image.Image):
            self.loaded_image = image
            self.loaded_image_array = np.array(self.loaded_image)
        elif isinstance(image, np.ndarray):
            self.loaded_image_array = image
            self.loaded_image = Image.fromarray(self.loaded_image_array)
        else:
            raise ValueError(
                "Input image must be a path string, PIL Image, or numpy array"
            )

    def generate_perspective_image(
        self, perspective: PerspectiveMetadata
    ) -> PerspectiveImage:
        """Generate a perspective projection from this panorama.

        Args:
            perspective: Configuration for the perspective projection.

        Returns:
            PerspectiveImage with the generated projection.

        Raises:
            ValueError: If the panorama image has not been loaded.
        """
        if self.loaded_image is None:
            raise ValueError("Image has not been loaded")

        perspective_image = PerspectiveImage(
            source_panorama_image_array=self.loaded_image_array,
            panorama_id=self.panorama_id,
            perspective_metadata=perspective,
        )
        return perspective_image

    def get_image(self) -> Image.Image:
        """Get the panorama as PIL Image."""
        return self.loaded_image

    def get_image_array(self) -> np.ndarray:
        """Get the panorama as numpy array."""
        return self.loaded_image_array
