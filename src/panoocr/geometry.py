"""Geometry utilities for coordinate conversion.

This module provides utility functions for converting between different
coordinate systems used in panorama image processing.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


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


def perspective_to_sphere(
    u: float,
    v: float,
    horizontal_fov: float,
    vertical_fov: float,
    yaw_offset: float,
    pitch_offset: float,
) -> Tuple[float, float]:
    """Convert perspective image coordinates to spherical coordinates.

    Uses proper 3D rotation to handle camera orientation correctly.
    This is the inverse of py360convert's e2p transformation.

    Args:
        u: Horizontal coordinate (0-1, left to right).
        v: Vertical coordinate (0-1, top to bottom).
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.
        yaw_offset: Camera yaw (horizontal rotation) in degrees.
        pitch_offset: Camera pitch (vertical rotation) in degrees.

    Returns:
        Tuple of (yaw, pitch) in degrees representing world spherical coordinates.

    Raises:
        ValueError: If FOV values are not positive.
    """
    if horizontal_fov <= 0 or vertical_fov <= 0:
        raise ValueError("FOV must be positive")

    # Convert to centered coordinates (-0.5 to 0.5)
    # x: positive = right, y: positive = up
    x = u - 0.5
    y = 0.5 - v

    half_h_fov = math.radians(horizontal_fov) / 2
    half_v_fov = math.radians(vertical_fov) / 2

    # Direction in camera local coordinates (camera looks along +Z)
    X_local = x * 2 * math.tan(half_h_fov)
    Y_local = y * 2 * math.tan(half_v_fov)
    Z_local = 1.0

    # Normalize to unit vector
    r = math.sqrt(X_local**2 + Y_local**2 + Z_local**2)
    X_local /= r
    Y_local /= r
    Z_local /= r

    # Rotate by camera orientation to get world coordinates
    pitch_rad = math.radians(pitch_offset)
    yaw_rad = math.radians(yaw_offset)

    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Rotation by pitch (around X axis)
    X_pitched = X_local
    Y_pitched = Y_local * cos_pitch + Z_local * sin_pitch
    Z_pitched = -Y_local * sin_pitch + Z_local * cos_pitch

    # Rotation by yaw (around Y axis)
    X_world = X_pitched * cos_yaw + Z_pitched * sin_yaw
    Y_world = Y_pitched
    Z_world = -X_pitched * sin_yaw + Z_pitched * cos_yaw

    world_yaw = math.degrees(math.atan2(X_world, Z_world))
    world_pitch = math.degrees(math.asin(np.clip(Y_world, -1.0, 1.0)))

    return world_yaw, world_pitch


def sphere_to_perspective(
    yaw: float,
    pitch: float,
    horizontal_fov: float,
    vertical_fov: float,
    yaw_offset: float,
    pitch_offset: float,
) -> Tuple[float, float] | None:
    """Convert spherical coordinates to perspective image coordinates.

    Uses proper 3D rotation to handle camera orientation correctly.
    This is the forward transformation matching py360convert's e2p.

    Args:
        yaw: World yaw angle in degrees.
        pitch: World pitch angle in degrees.
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.
        yaw_offset: Camera yaw (horizontal rotation) in degrees.
        pitch_offset: Camera pitch (vertical rotation) in degrees.

    Returns:
        Tuple of (u, v) in 0-1 range if the point is within the FOV,
        None if the point is outside the perspective view.

    Raises:
        ValueError: If FOV values are not positive.
    """
    if horizontal_fov <= 0 or vertical_fov <= 0:
        raise ValueError("FOV must be positive")

    # Convert world spherical to 3D Cartesian
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    X_world = math.cos(pitch_rad) * math.sin(yaw_rad)
    Y_world = math.sin(pitch_rad)
    Z_world = math.cos(pitch_rad) * math.cos(yaw_rad)

    # Apply inverse camera rotation to get local coordinates
    pitch_offset_rad = math.radians(pitch_offset)
    yaw_offset_rad = math.radians(yaw_offset)

    cos_pitch = math.cos(pitch_offset_rad)
    sin_pitch = math.sin(pitch_offset_rad)
    cos_yaw = math.cos(yaw_offset_rad)
    sin_yaw = math.sin(yaw_offset_rad)

    # Inverse yaw rotation (around Y axis)
    X_yawed = X_world * cos_yaw - Z_world * sin_yaw
    Y_yawed = Y_world
    Z_yawed = X_world * sin_yaw + Z_world * cos_yaw

    # Inverse pitch rotation (around X axis)
    X_local = X_yawed
    Y_local = Y_yawed * cos_pitch - Z_yawed * sin_pitch
    Z_local = Y_yawed * sin_pitch + Z_yawed * cos_pitch

    # Check if point is in front of camera
    if Z_local <= 0:
        return None

    # Project to image plane
    half_h_fov = math.radians(horizontal_fov) / 2
    half_v_fov = math.radians(vertical_fov) / 2

    x = X_local / (Z_local * 2 * math.tan(half_h_fov))
    y = Y_local / (Z_local * 2 * math.tan(half_v_fov))

    # Check if within FOV bounds
    if abs(x) > 0.5 or abs(y) > 0.5:
        return None

    # Convert to UV coordinates
    u = x + 0.5
    v = 0.5 - y

    return u, v


def calculate_spherical_centroid(
    polygons: List[List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """Calculate the centroid of spherical polygon(s) using 3D averaging.

    This handles wrap-around at ±180° correctly by converting to 3D Cartesian
    coordinates, averaging in 3D space, and converting back.

    Args:
        polygons: List of polygons, each polygon is a list of (yaw, pitch) tuples
            in degrees.

    Returns:
        Tuple of (center_yaw, center_pitch) in degrees.
    """
    all_points = [pt for polygon in polygons for pt in polygon]
    if not all_points:
        return 0.0, 0.0

    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
    for yaw_deg, pitch_deg in all_points:
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        # Spherical to Cartesian (pitch = latitude, yaw = longitude)
        # x = cos(pitch) * sin(yaw)  [East direction]
        # y = sin(pitch)             [Up direction]
        # z = cos(pitch) * cos(yaw)  [North direction]
        x = math.cos(pitch_rad) * math.sin(yaw_rad)
        y = math.sin(pitch_rad)
        z = math.cos(pitch_rad) * math.cos(yaw_rad)

        sum_x += x
        sum_y += y
        sum_z += z

    n = len(all_points)
    avg_x = sum_x / n
    avg_y = sum_y / n
    avg_z = sum_z / n

    magnitude = math.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
    if magnitude < 1e-10:
        # Degenerate case: points are symmetrically distributed.
        center_yaw = sum(p[0] for p in all_points) / n
        center_pitch = sum(p[1] for p in all_points) / n
        return center_yaw, center_pitch

    avg_x /= magnitude
    avg_y /= magnitude
    avg_z /= magnitude

    center_yaw = math.degrees(math.atan2(avg_x, avg_z))
    center_pitch = math.degrees(math.asin(max(-1.0, min(1.0, avg_y))))

    return center_yaw, center_pitch
