"""Panorama ingestion and validation.

Responsible only for getting a clean equirectangular RGB array into the
pipeline. Video input (Path B in the design doc) will land here too, as
`load_video_frames`, once the reconstruction stage exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Panorama:
    """An equirectangular panorama loaded into memory."""

    image: np.ndarray  # (H, W, 3) uint8, RGB
    source_path: Path

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class InvalidPanoramaError(ValueError):
    """Raised when the input image is not a usable equirectangular panorama."""


def load_panorama(
    path: str | Path,
    *,
    require_equirect_aspect: bool = True,
    aspect_tolerance: float = 0.05,
    max_width: int | None = 8192,
) -> Panorama:
    """Load an image and validate it as an equirectangular panorama.

    Equirectangular panoramas are 2:1 (width:height) — 360° horizontal,
    180° vertical. We check this loosely since many phone-captured
    panoramas are cropped slightly off ratio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Panorama not found: {path}")

    img = Image.open(path).convert("RGB")

    if max_width and img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, round(img.height * scale)), Image.LANCZOS)

    arr = np.asarray(img)

    if require_equirect_aspect:
        ratio = arr.shape[1] / arr.shape[0]
        if abs(ratio - 2.0) > aspect_tolerance * 2.0:
            raise InvalidPanoramaError(
                f"Expected ~2:1 equirectangular aspect ratio, got {ratio:.2f}:1 "
                f"({arr.shape[1]}x{arr.shape[0]}). Pass "
                "require_equirect_aspect=False to override."
            )

    return Panorama(image=arr, source_path=path)
