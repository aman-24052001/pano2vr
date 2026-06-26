# pano2vr

Turn a single equirectangular panorama into a VR-viewable, depth-displaced
mesh (`.glb`) — with the goal of keeping every step phone-feasible.

## Why this design

A flat equirect sphere is **3DoF**: you can look around, but leaning your
head produces no parallax — it feels flat. Full **6DoF** (walking around a
reconstructed scene) needs many views and heavy reconstruction (3D
Gaussian Splatting, NeRF, etc.) and is not realistically on-device today.

This repo targets the middle ground that *is* on-device-feasible from a
**single image**: **3DoF+** — monocular depth displaces a UV sphere so
head movement produces real motion parallax. The tradeoff: it buys "lean
into the room," not "walk across the room" (single-image depth can't
invent what's hidden behind objects). See `docs/` notes below for the
upgrade path (layered meshes / MSI, then full video→3DGS) when that limit
matters.

## Pipeline

```
panorama (.jpg/.png, ~2:1)
   │  capture.load_panorama        — validate equirect aspect, load array
   ▼
depth.DepthEstimator.estimate      — monocular depth map (pluggable backend)
   ▼
geometry.build_displaced_sphere    — depth -> UV sphere mesh (vectorized numpy)
   ▼
export.export_glb                  — mesh+texture -> single .glb file
   ▼
viewer/index.html (WebXR/three.js) — stereo render, parallax on head move
```

Each stage is a separate module with one job, independently testable.
`pipeline.py` just wires them together; `cli.py` and `mcp_server/` are two
different front doors to the same pipeline.

## Modules

| Module | Responsibility |
|---|---|
| `src/pano2vr/capture/` | Load + validate the panorama (video frame ingestion lands here later) |
| `src/pano2vr/depth/` | `DepthEstimator` interface. `HeuristicDepthEstimator` needs no model (for wiring/testing). `ONNXDepthEstimator` runs any monocular-depth ONNX model — the same file you'd load via ONNX Runtime Mobile / Core ML on a phone |
| `src/pano2vr/geometry/` | Depth map → displaced UV sphere (vertices, faces, UVs) |
| `src/pano2vr/export/` | Mesh → `.glb` |
| `src/pano2vr/pipeline.py` | End-to-end orchestration |
| `src/pano2vr/cli.py` | `pano2vr` command line tool |
| `mcp_server/` | Exposes the pipeline as an MCP tool for agent orchestration |
| `viewer/` | Static WebXR viewer (three.js) — open on a phone browser, load a `.glb`, hit "Enter VR" |

## Quickstart

```bash
pip install -e .
pano2vr path/to/panorama.jpg --out output.glb
# open viewer/index.html?glb=output.glb on a phone browser (or any
# static file server) and tap Enter VR
```

With a real depth model (ONNX):

```bash
pano2vr path/to/panorama.jpg --out output.glb --depth-model depth_model.onnx
```

Run tests (no model/network required — uses a synthetic test panorama
and the heuristic depth backend):

```bash
pip install -e ".[dev]"
pytest
```

## Depth models

No model weights are bundled. Export/convert a monocular depth model
(e.g. a 360-aware MiDaS/Depth-Anything variant, or one of the panorama
depth nets referenced below) to ONNX and pass `--depth-model`. The same
`.onnx` is what you'd later convert to Core ML (iOS) or run via NNAPI/ORT
Mobile (Android) for true on-device inference.

## Reference papers

- PanoSynthVR — lightweight single-panorama view synthesis, real-time target.
- Pano2Room (SIGGRAPH Asia 2024) — single panorama → 3DGS via mesh + inpainting.
- Casual 6-DoF — handheld 360 camera, offline depth refine + real-time online synthesis.
- OmniSplat (CVPR 2025) — feed-forward 3DGS for omnidirectional input.
- Mobile-GS — sort-free 3DGS rendering at 100+ FPS on phone GPUs (relevant once/if this repo moves to a splat representation instead of a mesh).

## Roadmap

1. **(done)** Path A MVP: panorama → displaced sphere → glb → WebXR, heuristic depth.
2. Swap in a real ONNX depth model; tune `min_radius`/`max_radius` per scene scale.
3. Layered mesh (LDI-style) to reduce disocclusion holes at the edges of parallax range.
4. Path B (video input): ARKit/ARCore pose capture → server-side feed-forward 3DGS → Mobile-GS-style on-device renderer.
