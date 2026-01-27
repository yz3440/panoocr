"""Perspective preset configurations for panorama OCR."""

from __future__ import annotations

from typing import List

from .models import PerspectiveMetadata


def generate_perspectives(
    pixel_size: int,
    horizontal_fov: float,
    vertical_fov: float | None = None,
    pitch_offsets: List[float] | None = None,
) -> List[PerspectiveMetadata]:
    """Generate a list of perspective configurations.

    Creates perspectives evenly distributed around the panorama horizontally,
    with 50% overlap between adjacent perspectives.

    Args:
        pixel_size: Width and height of each perspective image in pixels.
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees. Defaults to horizontal_fov.
        pitch_offsets: List of vertical offsets in degrees. Defaults to [0].

    Returns:
        List of PerspectiveMetadata configurations.
    """
    if vertical_fov is None:
        vertical_fov = horizontal_fov
    if pitch_offsets is None:
        pitch_offsets = [0]

    perspectives = []

    # Calculate number of yaw positions (2x overlap = interval is half of FOV)
    yaw_offset_count = round(360 / horizontal_fov * 2)
    interval = 360 / yaw_offset_count

    # Generate yaw offsets from -180 to 180
    yaw_offsets = [k * interval - 180 for k in range(yaw_offset_count)]

    for yaw_offset in yaw_offsets:
        for pitch_offset in pitch_offsets:
            perspective = PerspectiveMetadata(
                pixel_width=pixel_size,
                pixel_height=pixel_size,
                horizontal_fov=horizontal_fov,
                vertical_fov=vertical_fov,
                yaw_offset=yaw_offset,
                pitch_offset=pitch_offset,
            )
            perspectives.append(perspective)

    return perspectives


def combine_perspectives(
    *perspective_lists: List[PerspectiveMetadata],
) -> List[PerspectiveMetadata]:
    """Combine multiple perspective lists into a single list.

    Useful for multi-scale detection at different FOV settings.

    Args:
        *perspective_lists: Variable number of perspective lists to combine.

    Returns:
        Combined list of all perspectives.
    """
    combined = []
    for perspective_list in perspective_lists:
        combined.extend(perspective_list)
    return combined


def _initialize_default_perspectives() -> List[PerspectiveMetadata]:
    """Initialize default perspective configurations.

    45° FOV, 2048x2048 pixels, 22.5° intervals (16 perspectives).
    """
    return generate_perspectives(
        pixel_size=2048,
        horizontal_fov=45,
        vertical_fov=45,
    )


def _initialize_zoomed_in_perspectives() -> List[PerspectiveMetadata]:
    """Initialize zoomed-in perspective configurations.

    22.5° FOV, 1024x1024 pixels, for detecting smaller text.
    """
    return generate_perspectives(
        pixel_size=1024,
        horizontal_fov=22.5,
        vertical_fov=22.5,
    )


def _initialize_zoomed_out_perspectives() -> List[PerspectiveMetadata]:
    """Initialize zoomed-out perspective configurations.

    60° FOV, 2500x2500 pixels, for detecting larger text or wider context.
    """
    return generate_perspectives(
        pixel_size=2500,
        horizontal_fov=60,
        vertical_fov=60,
    )


# Pre-defined perspective sets
DEFAULT_IMAGE_PERSPECTIVES = _initialize_default_perspectives()
"""Default perspectives: 45° FOV, 2048x2048px, 16 positions."""

ZOOMED_IN_IMAGE_PERSPECTIVES = _initialize_zoomed_in_perspectives()
"""Zoomed-in perspectives: 22.5° FOV, 1024x1024px, for small text."""

ZOOMED_OUT_IMAGE_PERSPECTIVES = _initialize_zoomed_out_perspectives()
"""Zoomed-out perspectives: 60° FOV, 2500x2500px, for large text."""
