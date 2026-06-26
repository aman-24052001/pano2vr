from pathlib import Path

from pano2vr.pipeline import panorama_to_glb


def test_panorama_to_glb_end_to_end(synthetic_panorama_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.glb"
    result = panorama_to_glb(
        synthetic_panorama_path, out, lat_segments=16, lon_segments=32
    )

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
