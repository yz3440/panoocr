"""Perspective preset configurations for panorama OCR."""

from __future__ import annotations

from typing import List, Optional

from .models import PerspectiveMetadata


def generate_perspectives(
    fov: float = 45,
    resolution: int = 2048,
    overlap: float = 0.5,
    pitch_angles: Optional[List[float]] = None,
    vertical_fov: Optional[float] = None,
) -> List[PerspectiveMetadata]:
    """Generate a set of perspective views covering 360° horizontally.

    This is the main API for creating custom perspective configurations.

    Args:
        fov: Horizontal field of view in degrees (default: 45°).
        resolution: Pixel width and height of each perspective (default: 2048).
        overlap: Overlap ratio between adjacent perspectives, 0-1 (default: 0.5).
            - 0.0 = no overlap (perspectives touch at edges)
            - 0.5 = 50% overlap (recommended for good coverage)
            - 1.0 = 100% overlap (each point covered by 2 perspectives)
        pitch_angles: List of pitch angles in degrees (default: [0]).
            Use multiple values to cover up/down, e.g., [-30, 0, 30].
        vertical_fov: Vertical field of view in degrees (default: same as fov).

    Returns:
        List of PerspectiveMetadata objects covering the panorama.

    Examples:
        >>> # Standard 45° FOV with 50% overlap (16 perspectives)
        >>> perspectives = generate_perspectives(fov=45)

        >>> # Wide angle for large text (8 perspectives)
        >>> perspectives = generate_perspectives(fov=90, resolution=2500)

        >>> # Zoomed in for small text (32 perspectives)
        >>> perspectives = generate_perspectives(fov=22.5, resolution=1024)

        >>> # Cover ceiling and floor too
        >>> perspectives = generate_perspectives(fov=60, pitch_angles=[-45, 0, 45])

        >>> # Dense coverage with 75% overlap
        >>> perspectives = generate_perspectives(fov=45, overlap=0.75)
    """
    if pitch_angles is None:
        pitch_angles = [0]
    if vertical_fov is None:
        vertical_fov = fov

    # Calculate yaw interval based on FOV and overlap
    # With 50% overlap, interval = FOV / 2
    # With 0% overlap, interval = FOV
    yaw_interval = fov * (1 - overlap)
    if yaw_interval <= 0:
        yaw_interval = fov * 0.1  # Minimum 10% step to avoid infinite loop

    # Generate yaw angles centered at 0
    num_yaw = int(round(360 / yaw_interval))
    yaw_angles = [i * (360 / num_yaw) - 180 for i in range(num_yaw)]

    perspectives = []
    for yaw in yaw_angles:
        for pitch in pitch_angles:
            perspectives.append(
                PerspectiveMetadata(
                    pixel_width=resolution,
                    pixel_height=resolution,
                    horizontal_fov=fov,
                    vertical_fov=vertical_fov,
                    yaw_offset=yaw,
                    pitch_offset=pitch,
                )
            )

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


# =============================================================================
# Pre-defined perspective sets (for backwards compatibility and convenience)
# =============================================================================

DEFAULT_IMAGE_PERSPECTIVES = generate_perspectives(fov=45, resolution=2048, overlap=0.5)
"""List[PerspectiveMetadata]: Default perspectives with 45° FOV (16 perspectives)."""

ZOOMED_IN_IMAGE_PERSPECTIVES = generate_perspectives(fov=22.5, resolution=1024, overlap=0.5)
"""List[PerspectiveMetadata]: Zoomed-in perspectives with 22.5° FOV (32 perspectives)."""

ZOOMED_OUT_IMAGE_PERSPECTIVES = generate_perspectives(fov=60, resolution=2500, overlap=0.5)
"""List[PerspectiveMetadata]: Zoomed-out perspectives with 60° FOV (12 perspectives)."""

WIDEANGLE_IMAGE_PERSPECTIVES = generate_perspectives(fov=90, resolution=2500, overlap=0.5)
"""List[PerspectiveMetadata]: Wideangle perspectives with 90° FOV (8 perspectives)."""
