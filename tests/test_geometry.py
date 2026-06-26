import numpy as np

from pano2vr.capture import load_panorama
from pano2vr.depth import HeuristicDepthEstimator
from pano2vr.geometry import build_displaced_sphere


def test_build_displaced_sphere_shapes(synthetic_panorama_path) -> None:
    pano = load_panorama(synthetic_panorama_path)
    depth = HeuristicDepthEstimator().estimate(pano)

    mesh = build_displaced_sphere(pano, depth, lat_segments=8, lon_segments=16)

    n_lat, n_lon = 9, 17
    assert mesh.vertices.shape == (n_lat * n_lon, 3)
    assert mesh.uvs.shape == (n_lat * n_lon, 2)
    assert mesh.faces.shape == (8 * 16 * 2, 3)
    assert mesh.faces.max() < mesh.n_vertices


def test_vertex_radius_matches_depth_mapping(synthetic_panorama_path) -> None:
    pano = load_panorama(synthetic_panorama_path)
    # constant "near" depth everywhere -> every vertex should sit at min_radius
    depth = np.zeros((pano.height, pano.width), dtype=np.float32)

    mesh = build_displaced_sphere(
        pano, depth, lat_segments=4, lon_segments=4, min_radius=0.3, max_radius=1.0
    )
    radii = np.linalg.norm(mesh.vertices, axis=1)
    np.testing.assert_allclose(radii, 0.3, atol=1e-4)
