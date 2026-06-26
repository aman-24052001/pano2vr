"""Turn (panorama, depth map) into a depth-displaced UV sphere mesh.

This is the simplest viable 6DoF-ish representation: instead of a flat
sphere at fixed radius (3DoF, no parallax), each vertex's radius is driven
by estimated depth, so near geometry pulls in and far geometry pushes out.
Viewed from slightly off-center (head movement) this produces real motion
parallax. It buys "lean room", not "walk across the room" — see README for
the disocclusion limits of single-image 6DoF and the layered-mesh upgrade
path (LDI / MSI) for when that's not enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from pano2vr.capture import Panorama


@dataclass(frozen=True)
class TexturedMesh:
    """A textured triangle mesh, in a format trivial to hand to any exporter."""

    vertices: np.ndarray  # (N, 3) float32, xyz
    faces: np.ndarray  # (M, 3) int32, triangle vertex indices
    uvs: np.ndarray  # (N, 2) float32, image-space UV, (0,0) = texture top-left
    texture: np.ndarray  # (Ht, Wt, 3) uint8, RGB

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]


def _resize_depth(depth: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    img = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(img.resize((out_w, out_h), Image.BILINEAR), dtype=np.float32)


def build_displaced_sphere(
    panorama: Panorama,
    depth: np.ndarray,
    *,
    lat_segments: int = 128,
    lon_segments: int = 256,
    min_radius: float = 0.3,
    max_radius: float = 1.0,
) -> TexturedMesh:
    """Build a depth-displaced UV sphere centered on the capture point.

    Args:
        panorama: source equirectangular panorama (provides the texture).
        depth: (H, W) float32 array in (0, 1], 0=near, 1=far — see
            `pano2vr.depth.DepthEstimator` convention.
        lat_segments / lon_segments: mesh resolution. 128x256 (~33k verts)
            is a reasonable phone-GPU budget; raise for desktop preview.
        min_radius / max_radius: world-unit radius range the normalized
            depth is mapped onto. Tune to your room/scene scale.
    """
    n_lat, n_lon = lat_segments + 1, lon_segments + 1
    depth_grid = _resize_depth(depth, n_lon, n_lat)

    phi = np.linspace(0.0, np.pi, n_lat, dtype=np.float32)  # colatitude, top->bottom
    theta = np.linspace(0.0, 2.0 * np.pi, n_lon, dtype=np.float32)  # longitude
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")  # (n_lat, n_lon)

    radius = min_radius + (max_radius - min_radius) * depth_grid

    sin_phi = np.sin(phi_grid)
    x = radius * sin_phi * np.cos(theta_grid)
    y = radius * np.cos(phi_grid)
    z = radius * sin_phi * np.sin(theta_grid)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    u = (theta_grid / (2.0 * np.pi)).reshape(-1)
    v = (phi_grid / np.pi).reshape(-1)
    uvs = np.stack([u, v], axis=-1).astype(np.float32)

    # Two triangles per grid cell, vectorized over the (lat_segments x
    # lon_segments) cell grid — no wrap-around modulo needed since theta
    # already spans a full 0..2pi seam as duplicate columns.
    i_idx, j_idx = np.meshgrid(
        np.arange(lat_segments), np.arange(lon_segments), indexing="ij"
    )
    i_idx, j_idx = i_idx.reshape(-1), j_idx.reshape(-1)

    def vid(i, j):
        return i * n_lon + j

    v00 = vid(i_idx, j_idx)
    v01 = vid(i_idx, j_idx + 1)
    v10 = vid(i_idx + 1, j_idx)
    v11 = vid(i_idx + 1, j_idx + 1)

    tri_a = np.stack([v00, v01, v10], axis=-1)
    tri_b = np.stack([v01, v11, v10], axis=-1)
    faces = np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)

    return TexturedMesh(
        vertices=vertices, faces=faces, uvs=uvs, texture=panorama.image
    )
