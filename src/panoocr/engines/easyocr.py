"""EasyOCR Engine using the easyocr library.

This module provides OCR using the EasyOCR library which supports
80+ languages and runs on CPU or GPU.

Install with: pip install "panoocr[easyocr]"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..ocr.models import BoundingBox, FlatOCRResult


def _check_easyocr_dependencies():
    """Check if EasyOCR dependencies are installed."""
    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "EasyOCR dependencies not installed.\n\n"
            "Install with:\n"
            "  pip install 'panoocr[easyocr]'\n\n"
            "For GPU support, also install PyTorch with CUDA."
        )


class EasyOCRLanguageCode(Enum):
    """Common language codes for EasyOCR.

    See https://www.jaided.ai/easyocr/ for the full list of 80+ languages.
    """

    ABAZA = "abq"
    ADYGHE = "ady"
    AFRIKAANS = "af"
    ANGIKA = "ang"
    ARABIC = "ar"
    ASSAMESE = "as"
    AVAR = "ava"
    AZERBAIJANI = "az"
    BELARUSIAN = "be"
    BULGARIAN = "bg"
    BIHARI = "bh"
    BHOJPURI = "bho"
    BENGALI = "bn"
    BOSNIAN = "bs"
    SIMPLIFIED_CHINESE = "ch_sim"
    TRADITIONAL_CHINESE = "ch_tra"
    CHECHEN = "che"
    CZECH = "cs"
    WELSH = "cy"
    DANISH = "da"
    DARGWA = "dar"
    GERMAN = "de"
    ENGLISH = "en"
    SPANISH = "es"
    ESTONIAN = "et"
    PERSIAN = "fa"
    FRENCH = "fr"
    IRISH = "ga"
    GOAN_KONKANI = "gom"
    HINDI = "hi"
    CROATIAN = "hr"
    HUNGARIAN = "hu"
    INDONESIAN = "id"
    INGUSH = "inh"
    ICELANDIC = "is"
    ITALIAN = "it"
    JAPANESE = "ja"
    KABARDIAN = "kbd"
    KANNADA = "kn"
    KOREAN = "ko"
    KURDISH = "ku"
    LATIN = "la"
    LAK = "lbe"
    LEZGHIAN = "lez"
    LITHUANIAN = "lt"
    LATVIAN = "lv"
    MAGAHI = "mah"
    MAITHILI = "mai"
    MAORI = "mi"
    MONGOLIAN = "mn"
    MARATHI = "mr"
    MALAY = "ms"
    MALTESE = "mt"
    NEPALI = "ne"
    NEWARI = "new"
    DUTCH = "nl"
    NORWEGIAN = "no"
    OCCITAN = "oc"
    PALI = "pi"
    POLISH = "pl"
    PORTUGUESE = "pt"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SERBIAN_CYRILLIC = "rs_cyrillic"
    SERBIAN_LATIN = "rs_latin"
    NAGPURI = "sck"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    ALBANIAN = "sq"
    SWEDISH = "sv"
    SWAHILI = "sw"
    TAMIL = "ta"
    TABASSARAN = "tab"
    TELUGU = "te"
    THAI = "th"
    TAJIK = "tjk"
    TAGALOG = "tl"
    TURKISH = "tr"
    UYGHUR = "ug"
    UKRAINIAN = "uk"
    URDU = "ur"
    UZBEK = "uz"
    VIETNAMESE = "vi"


DEFAULT_LANGUAGE_PREFERENCE = [EasyOCRLanguageCode.ENGLISH]


@dataclass
class EasyOCRResult:
    """Raw result from EasyOCR."""

    text: str
    bounding_box: List[List[float]]  # 4 corner points
    confidence: float
    image_width: int
    image_height: int

    def to_flat(self) -> FlatOCRResult:
        """Convert to FlatOCRResult with normalized coordinates."""
        # EasyOCR returns 4 corner points as [[x,y], [x,y], [x,y], [x,y]]
        left = min(p[0] for p in self.bounding_box)
        right = max(p[0] for p in self.bounding_box)
        top = min(p[1] for p in self.bounding_box)
        bottom = max(p[1] for p in self.bounding_box)

        return FlatOCRResult(
            text=self.text,
            confidence=self.confidence,
            bounding_box=BoundingBox(
                left=left / self.image_width,
                top=top / self.image_height,
                right=right / self.image_width,
                bottom=bottom / self.image_height,
                width=(right - left) / self.image_width,
                height=(bottom - top) / self.image_height,
            ),
            engine="EASYOCR",
        )


class EasyOCREngine:
    """OCR engine using EasyOCR library.

    EasyOCR supports 80+ languages and can run on CPU or GPU.
    It provides good accuracy for many scripts including CJK.

    Attributes:
        language_preference: List of language codes to use.
        reader: EasyOCR Reader instance.

    Example:
        >>> from panoocr.engines.easyocr import EasyOCREngine, EasyOCRLanguageCode
        >>>
        >>> engine = EasyOCREngine(config={
        ...     "language_preference": [EasyOCRLanguageCode.ENGLISH],
        ...     "gpu": True,
        ... })
        >>> results = engine.recognize(image)

    Note:
        Install with: pip install "panoocr[easyocr]"
        For GPU support, install PyTorch with CUDA.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the EasyOCR engine.

        Args:
            config: Configuration dictionary with optional keys:
                - language_preference: List of EasyOCRLanguageCode values.
                - gpu: Whether to use GPU (default: True).

        Raises:
            ImportError: If easyocr is not installed.
            ValueError: If configuration values are invalid.
        """
        # Check dependencies first
        _check_easyocr_dependencies()

        import easyocr

        config = config or {}

        # Parse language preference
        language_preference = config.get(
            "language_preference", DEFAULT_LANGUAGE_PREFERENCE
        )
        try:
            self.language_preference = [
                lang.value if isinstance(lang, EasyOCRLanguageCode) else lang
                for lang in language_preference
            ]
        except (KeyError, AttributeError):
            raise ValueError("Invalid language code in language_preference")

        # Parse GPU setting
        use_gpu = config.get("gpu", True)

        # Initialize reader
        self.reader = easyocr.Reader(self.language_preference, gpu=use_gpu)

    def recognize(self, image: Image.Image) -> List[FlatOCRResult]:
        """Recognize text in an image.

        Args:
            image: Input image as PIL Image.

        Returns:
            List of FlatOCRResult with normalized bounding boxes.
        """
        image_array = np.array(image)
        annotations = self.reader.readtext(image_array)

        easyocr_results = []
        for annotation in annotations:
            bounding_box = annotation[0]
            text = annotation[1]
            confidence = annotation[2]

            easyocr_results.append(
                EasyOCRResult(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    image_width=image.width,
                    image_height=image.height,
                )
            )

        return [result.to_flat() for result in easyocr_results]
