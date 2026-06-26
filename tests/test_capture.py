from pathlib import Path

import pytest

from pano2vr.capture import InvalidPanoramaError, load_panorama


def test_load_panorama_ok(synthetic_panorama_path: Path) -> None:
    pano = load_panorama(synthetic_panorama_path)
    assert pano.width == 512
    assert pano.height == 256
    assert pano.aspect_ratio == pytest.approx(2.0)


def test_load_panorama_rejects_bad_aspect(tmp_path: Path) -> None:
    from PIL import Image
    import numpy as np

    bad = Image.fromarray(np.zeros((300, 300, 3), dtype="uint8"))
    path = tmp_path / "square.png"
    bad.save(path)

    with pytest.raises(InvalidPanoramaError):
        load_panorama(path)
