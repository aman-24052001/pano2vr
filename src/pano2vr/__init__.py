"""pano2vr — turn a single equirectangular panorama into a depth-displaced
3D mesh viewable in VR (3DoF+ parallax), with the heavy lifting designed to
run on-device (phone GPU/NPU) wherever possible.

Module map:
    capture/   - load & validate equirectangular panorama input
    depth/     - monocular depth estimation backends (pluggable)
    geometry/  - depth map -> displaced sphere mesh / layered representation
    export/    - mesh+texture -> glTF/GLB for any WebXR/three.js viewer
    pipeline   - orchestrates the above end to end
    cli        - `pano2vr` command line entrypoint
"""

__version__ = "0.1.0"
