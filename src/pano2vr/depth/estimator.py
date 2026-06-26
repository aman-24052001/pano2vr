"""Monocular depth estimation for a single equirectangular panorama.

This is the only stage in the pipeline that *needs* a neural net. The
interface is deliberately backend-agnostic so the same pipeline code runs
with:
  - a cheap heuristic (no model, useful for wiring/testing the rest of the
    pipeline without downloading any weights),
  - an ONNX Runtime backend (the path to on-device: ONNX Runtime Mobile /
    Core ML / NNAPI execution providers let this run on-phone),
  - any future panorama-aware depth network (e.g. a 360-finetuned
    MiDaS/Depth Anything variant, or an MSI/MDP predictor).

Real model weights are NOT bundled here — point ONNXDepthEstimator at a
.onnx file you've exported/converted yourself (see README "Depth models").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from pano2vr.capture import Panorama


class DepthEstimator(ABC):
    """Common interface: panorama in, normalized depth map out.

    Depth convention: float32, shape (H, W), values in (0, 1] where
    smaller = closer to the camera, 1.0 = far/background. Downstream
    geometry code only depends on this convention, not on the backend.
    """

    @abstractmethod
    def estimate(self, panorama: Panorama) -> np.ndarray:
        ...


class HeuristicDepthEstimator(DepthEstimator):
    """No-model placeholder depth.

    Uses the simple indoor-scene prior that ceiling/floor (top/bottom of
    an equirect panorama) tends to be closer than the horizon band, which
    is roughly true for room-scale captures. This exists purely so the
    geometry/export/viewer stages can be built, tested, and iterated on
    without a network dependency. Swap for ONNXDepthEstimator once you
    have a real model — nothing else in the pipeline changes.
    """

    def estimate(self, panorama: Panorama) -> np.ndarray:
        h, w = panorama.height, panorama.width
        # phi: vertical angle, 0 at top (ceiling) .. pi at bottom (floor)
        phi = np.linspace(0, np.pi, h, dtype=np.float32)
        # Horizon (phi ~ pi/2) is "far", ceiling/floor (phi ~ 0, pi) is "near"
        vertical_term = np.abs(np.sin(phi))  # 0 at poles, 1 at horizon
        depth_col = 0.15 + 0.85 * vertical_term  # keep a near-field floor
        depth = np.tile(depth_col[:, None], (1, w)).astype(np.float32)
        return depth


class ONNXDepthEstimator(DepthEstimator):
    """Runs a monocular depth ONNX model over the panorama.

    For real on-device deployment, this same .onnx file is what you'd load
    via ONNX Runtime Mobile (Android) or convert to Core ML (iOS) — the
    pre/post-processing below is written to be portable to both.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (384, 384),
        providers: Optional[list[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ONNXDepthEstimator requires onnxruntime. "
                "Install with: pip install onnxruntime"
            ) from e

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Depth model not found: {self.model_path}. Export/convert "
                "a monocular depth model to ONNX and point this at it."
            )
        self.input_size = input_size
        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name

    def estimate(self, panorama: Panorama) -> np.ndarray:
        from PIL import Image

        h, w = panorama.height, panorama.width
        small = Image.fromarray(panorama.image).resize(
            self.input_size, Image.BILINEAR
        )
        x = np.asarray(small, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None, ...]  # NCHW

        (raw_depth,) = self.session.run(None, {self._input_name: x})
        raw_depth = np.squeeze(raw_depth).astype(np.float32)

        d_min, d_max = raw_depth.min(), raw_depth.max()
        norm = (raw_depth - d_min) / max(d_max - d_min, 1e-6)

        depth_img = Image.fromarray((norm * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
        return np.asarray(depth_img, dtype=np.float32) / 255.0
