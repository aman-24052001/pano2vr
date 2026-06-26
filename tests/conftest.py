from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_panorama_path(tmp_path: Path) -> Path:
    """A small synthetic 2:1 checkerboard 'panorama' — no real photo needed
    to exercise capture -> depth -> geometry -> export end to end."""
    w, h = 512, 256
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    checker = (((xv // 32) + (yv // 32)) % 2 * 255).astype(np.uint8)
    rgb = np.stack([checker, np.roll(checker, 16, axis=1), 255 - checker], axis=-1)

    path = tmp_path / "synthetic_pano.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    return path
