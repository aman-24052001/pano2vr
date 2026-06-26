"""MCP server exposing pano2vr as a tool an agent (or you, via Claude/any
MCP client) can call directly — e.g. "convert this panorama and give me a
viewer link" without touching the CLI.

Run:
    python -m mcp_server.server
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pano2vr.pipeline import panorama_to_glb

mcp = FastMCP("pano2vr")


@mcp.tool()
def convert_panorama_to_glb(
    panorama_path: str,
    out_path: str = "output.glb",
    lat_segments: int = 128,
    lon_segments: int = 256,
    min_radius: float = 0.3,
    max_radius: float = 1.0,
) -> str:
    """Convert a single equirectangular panorama into a depth-displaced
    .glb mesh, viewable in VR via viewer/index.html?glb=<out_path>.

    Uses the no-model heuristic depth estimator by default. For real
    depth, run the CLI with --depth-model instead — ONNX model wiring
    isn't exposed as an MCP parameter yet (model paths are local/large).
    """
    result = panorama_to_glb(
        Path(panorama_path),
        Path(out_path),
        lat_segments=lat_segments,
        lon_segments=lon_segments,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    return str(result)


if __name__ == "__main__":
    mcp.run()
