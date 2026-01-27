"""Geometry utilities for coordinate conversion.

This module provides utility functions for converting between different
coordinate systems used in panorama image processing.
"""

from __future__ import annotations

import math
from typing import Tuple


def uv_to_yaw_pitch(
    u: float,
    v: float,
    horizontal_fov: float,
    vertical_fov: float,
) -> Tuple[float, float]:
    """Convert UV coordinates to yaw and pitch angles.

    Converts normalized image coordinates (0-1 range) to spherical
    coordinates using the camera field of view parameters.

    Args:
        u: Horizontal coordinate (0-1, left to right).
        v: Vertical coordinate (0-1, top to bottom).
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.

    Returns:
        Tuple of (yaw, pitch) in degrees.
        - yaw: Horizontal angle (-fov/2 to fov/2)
        - pitch: Vertical angle (-fov/2 to fov/2)

    Raises:
        ValueError: If FOV values are not positive.
    """
    if horizontal_fov <= 0 or vertical_fov <= 0:
        raise ValueError("FOV must be positive")

    # Translate origin to center of image
    u_centered = u - 0.5
    v_centered = 0.5 - v  # Flip vertical axis

    # Convert to angles using perspective projection
    yaw = math.atan2(
        2 * u_centered * math.tan(math.radians(horizontal_fov) / 2), 1
    )
    pitch = math.atan2(
        2 * v_centered * math.tan(math.radians(vertical_fov) / 2), 1
    )

    return math.degrees(yaw), math.degrees(pitch)


def yaw_pitch_to_uv(
    yaw: float,
    pitch: float,
    horizontal_fov: float,
    vertical_fov: float,
) -> Tuple[float, float]:
    """Convert yaw and pitch angles to UV coordinates.

    Converts spherical coordinates to normalized image coordinates
    using the camera field of view parameters.

    Args:
        yaw: Horizontal angle in degrees.
        pitch: Vertical angle in degrees.
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.

    Returns:
        Tuple of (u, v) in 0-1 range.
        - u: Horizontal coordinate (0 = left, 1 = right)
        - v: Vertical coordinate (0 = top, 1 = bottom)

    Raises:
        ValueError: If FOV values are not positive.
    """
    if horizontal_fov <= 0 or vertical_fov <= 0:
        raise ValueError("FOV must be positive")

    # Convert angles to centered coordinates
    u_centered = math.tan(math.radians(yaw)) / (
        2 * math.tan(math.radians(horizontal_fov) / 2)
    )
    v_centered = math.tan(math.radians(pitch)) / (
        2 * math.tan(math.radians(vertical_fov) / 2)
    )

    # Translate back to image coordinates
    u = u_centered + 0.5
    v = 0.5 - v_centered  # Flip vertical axis

    return u, v


def normalize_yaw(yaw: float) -> float:
    """Normalize yaw angle to -180 to 180 range.

    Args:
        yaw: Yaw angle in degrees.

    Returns:
        Normalized yaw in -180 to 180 range.
    """
    while yaw > 180:
        yaw -= 360
    while yaw < -180:
        yaw += 360
    return yaw


def yaw_to_equirectangular_x(yaw: float, image_width: int) -> float:
    """Convert yaw angle to equirectangular image x coordinate.

    Args:
        yaw: Yaw angle in degrees (-180 to 180).
        image_width: Width of the equirectangular image.

    Returns:
        X coordinate in pixels.
    """
    normalized_yaw = normalize_yaw(yaw)
    return (normalized_yaw + 180) / 360 * image_width


def pitch_to_equirectangular_y(pitch: float, image_height: int) -> float:
    """Convert pitch angle to equirectangular image y coordinate.

    Args:
        pitch: Pitch angle in degrees (-90 to 90).
        image_height: Height of the equirectangular image.

    Returns:
        Y coordinate in pixels.
    """
    # Clamp pitch to valid range
    pitch = max(-90, min(90, pitch))
    return (90 - pitch) / 180 * image_height


def equirectangular_x_to_yaw(x: float, image_width: int) -> float:
    """Convert equirectangular image x coordinate to yaw angle.

    Args:
        x: X coordinate in pixels.
        image_width: Width of the equirectangular image.

    Returns:
        Yaw angle in degrees (-180 to 180).
    """
    return (x / image_width) * 360 - 180


def equirectangular_y_to_pitch(y: float, image_height: int) -> float:
    """Convert equirectangular image y coordinate to pitch angle.

    Args:
        y: Y coordinate in pixels.
        image_height: Height of the equirectangular image.

    Returns:
        Pitch angle in degrees (-90 to 90).
    """
    return 90 - (y / image_height) * 180
