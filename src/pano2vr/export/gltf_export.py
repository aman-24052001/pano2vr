"""Export a TexturedMesh to glTF/GLB.

glTF's UV convention is (0,0) = texture top-left, which is exactly the
image-space convention `geometry.sphere_mesh` already produces — no V-flip
needed here. GLB is chosen as the on-disk format because it's a single
binary file (mesh + texture) that three.js / WebXR / Quick Look / Scene
Viewer all load natively, with zero custom format on the viewer side.
"""

from __future__ import annotations

from pathlib import Path

from pano2vr.geometry import TexturedMesh


def export_glb(mesh: TexturedMesh, out_path: str | Path) -> Path:
    """Write a TexturedMesh to a single .glb file."""
    try:
        import trimesh
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "export_glb requires trimesh and pillow. "
            "Install with: pip install trimesh pillow"
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texture_image = Image.fromarray(mesh.texture)
    visual = trimesh.visual.TextureVisuals(uv=mesh.uvs, image=texture_image)

    tmesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        visual=visual,
        process=False,  # keep our exact vertex/UV layout, no dedup/merge
    )

    tmesh.export(out_path, file_type="glb")
    return out_path
