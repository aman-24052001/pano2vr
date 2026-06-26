"""End-to-end orchestration: panorama -> depth -> mesh -> glb.

Kept deliberately thin — each stage is independently testable, this just
wires them together so the CLI (and later, the MCP server) has one call.
"""

from __future__ import annotations

from pathlib import Path

from pano2vr.capture import load_panorama
from pano2vr.depth import DepthEstimator, HeuristicDepthEstimator
from pano2vr.export import export_glb
from pano2vr.geometry import build_displaced_sphere


def panorama_to_glb(
    panorama_path: str | Path,
    out_path: str | Path,
    *,
    depth_estimator: DepthEstimator | None = None,
    lat_segments: int = 128,
    lon_segments: int = 256,
    min_radius: float = 0.3,
    max_radius: float = 1.0,
) -> Path:
    """Run the full Path-A pipeline on a single panorama image.

    Args:
        panorama_path: input equirectangular (~2:1) image.
        out_path: where to write the .glb.
        depth_estimator: defaults to HeuristicDepthEstimator (no model
            download required). Pass an ONNXDepthEstimator for real depth.
    """
    estimator = depth_estimator or HeuristicDepthEstimator()

    panorama = load_panorama(panorama_path)
    depth = estimator.estimate(panorama)
    mesh = build_displaced_sphere(
        panorama,
        depth,
        lat_segments=lat_segments,
        lon_segments=lon_segments,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    return export_glb(mesh, out_path)
