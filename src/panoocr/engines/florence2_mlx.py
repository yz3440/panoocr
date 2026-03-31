"""Florence-2 MLX Engine using mlx-vlm.

Uses the MLX-optimized Florence-2 model with <OCR_WITH_REGION> task
to get structured text + bounding box output.

Install with: pip install "panoocr[mlx-vlm]" torch
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult


def _check_florence2_mlx_dependencies():
    missing = []
    try:
        import mlx_vlm
    except ImportError:
        missing.append("mlx-vlm")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import torchvision
    except ImportError:
        missing.append("torchvision")
    if missing:
        raise ImportError(
            f"Florence-2 MLX dependencies not installed: {', '.join(missing)}\n\n"
            "Install with:\n"
            "  pip install 'panoocr[mlx-vlm]' torch torchvision"
        )


def _patch_florence2_config():
    """Patch config loading for transformers 5.x + Florence-2 MLX compatibility.

    The MLX model's config.json references 'florence2_language' as a model_type
    for its text_config, which isn't registered in transformers 5.x CONFIG_MAPPING.
    We register BartConfig as a stand-in (Florence-2's language model is BART-based).
    We also disable remote code loading to avoid stale cached custom code.
    """
    import os
    os.environ["TRANSFORMERS_NO_REMOTE_CODE"] = "1"

    from transformers import CONFIG_MAPPING
    from transformers.models.bart.configuration_bart import BartConfig

    try:
        CONFIG_MAPPING.register("florence2_language", BartConfig)
    except ValueError:
        pass  # already registered


def _parse_ocr_with_region(text: str, image_width: int, image_height: int):
    """Parse Florence-2 <OCR_WITH_REGION> raw output into structured results.

    Florence-2 returns text like:
      TEXT1<loc_x1><loc_y1><loc_x2><loc_y2><loc_x3><loc_y3><loc_x4><loc_y4>TEXT2<loc_...>
    where loc values are 0-999 normalized coordinates forming quad-boxes.
    """
    results = []
    loc_pattern = r'<loc_(\d+)>'

    # Split text at sequences of <loc_...> tokens
    # Each detection is: label text followed by 8 loc tokens (quad-box)
    parts = re.split(r'((?:<loc_\d+>)+)', text)

    i = 0
    while i < len(parts) - 1:
        label = parts[i].strip()
        if i + 1 < len(parts):
            locs = [int(m) for m in re.findall(loc_pattern, parts[i + 1])]
        else:
            locs = []
        i += 2

        if not label or len(locs) < 4:
            continue

        if len(locs) == 8:
            coords = [(locs[j], locs[j + 1]) for j in range(0, 8, 2)]
        elif len(locs) == 4:
            x1, y1, x2, y2 = locs
            coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            # Take first 4 as x1,y1,x2,y2
            x1, y1, x2, y2 = locs[0], locs[1], locs[2], locs[3]
            coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        px_coords = [
            (c[0] / 999.0 * image_width, c[1] / 999.0 * image_height)
            for c in coords
        ]

        results.append((label, px_coords))

    return results


@dataclass
class Florence2MLXResult:
    """Raw result from Florence-2 MLX."""

    text: str
    bounding_box: List[tuple]  # 4 corner points in pixel coords
    image_width: int
    image_height: int

    def to_flat(self) -> FlatOCRResult:
        left = min(p[0] for p in self.bounding_box)
        right = max(p[0] for p in self.bounding_box)
        top = min(p[1] for p in self.bounding_box)
        bottom = max(p[1] for p in self.bounding_box)

        return FlatOCRResult(
            text=self.text,
            confidence=1.0,
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="FLORENCE_2_MLX",
        )


class Florence2MLXEngine:
    """OCR engine using Florence-2 via mlx-vlm with <OCR_WITH_REGION>.

    This is a structured engine -- it returns per-word bounding boxes.

    Example:
        >>> from panoocr.engines.florence2_mlx import Florence2MLXEngine
        >>> engine = Florence2MLXEngine()
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[mlx-vlm]" torch
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        _check_florence2_mlx_dependencies()
        _patch_florence2_config()

        from mlx_vlm import load

        config = config or {}
        model_id = config.get("model_id", "mlx-community/Florence-2-large-ft-8bit")
        self.max_tokens = config.get("max_tokens", 1024)

        print(f"Loading Florence-2 MLX model: {model_id}...")
        self.model, self.processor = load(model_id, trust_remote_code=False)
        print("Florence-2 MLX model loaded.")

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        from mlx_vlm import generate

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f, format="JPEG")
            tmp_path = f.name

        try:
            result = generate(
                self.model,
                self.processor,
                image=tmp_path,
                prompt="<OCR_WITH_REGION>",
                max_tokens=self.max_tokens,
                verbose=False,
            )
        finally:
            import os
            os.unlink(tmp_path)

        output_text = result.text if hasattr(result, "text") else str(result)
        parsed = _parse_ocr_with_region(output_text, image.width, image.height)

        results = []
        for label, coords in parsed:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            if max(xs) - min(xs) < 1 and max(ys) - min(ys) < 1:
                continue
            results.append(
                Florence2MLXResult(
                    text=label,
                    bounding_box=coords,
                    image_width=image.width,
                    image_height=image.height,
                )
            )

        return [r.to_flat() for r in results]
