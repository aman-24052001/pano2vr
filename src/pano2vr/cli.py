"""`pano2vr` command line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from pano2vr.depth import ONNXDepthEstimator
from pano2vr.pipeline import panorama_to_glb

app = typer.Typer(add_completion=False, help="Panorama -> VR-viewable mesh (.glb).")


@app.command()
def convert(
    panorama: Path = typer.Argument(..., help="Input equirectangular image."),
    out: Path = typer.Option(Path("output.glb"), "--out", "-o", help="Output .glb path."),
    depth_model: Path | None = typer.Option(
        None, "--depth-model", help="Path to an ONNX monocular depth model. "
        "Omit to use the no-model heuristic depth (for pipeline testing)."
    ),
    lat_segments: int = typer.Option(128, help="Mesh latitude resolution."),
    lon_segments: int = typer.Option(256, help="Mesh longitude resolution."),
    min_radius: float = typer.Option(0.3, help="Radius for nearest depth."),
    max_radius: float = typer.Option(1.0, help="Radius for farthest depth."),
) -> None:
    """Convert a single equirectangular panorama into a depth-displaced .glb."""
    estimator = ONNXDepthEstimator(depth_model) if depth_model else None

    result = panorama_to_glb(
        panorama,
        out,
        depth_estimator=estimator,
        lat_segments=lat_segments,
        lon_segments=lon_segments,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    typer.echo(f"Wrote {result}")


if __name__ == "__main__":
    app()
