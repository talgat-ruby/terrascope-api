"""U-Net Roads — segmentation-based road detection.

Port of the standalone ``run_roads_ml.py`` training/inference script into a
``Process``. The training pipeline (OSM-label generation, patch dataset,
U-Net training loop) lives only in the standalone script — this module is
inference-only and expects a checkpoint produced by that script.

Pipeline:
  1. Slide a ``patch_size`` window with stride ``patch_size // 2`` over the
     full raster (HWC uint8 RGB, ImageNet-normalised).
  2. Average per-patch sigmoid road probabilities into a fused probability
     map.
  3. Threshold at ``road_threshold`` to a binary mask.
  4. Vectorise the mask to polygons via ``rasterio.features.shapes``,
     simplifying small artefacts away with ``min_area_px``.
  5. Emit one ``Detection`` (Polygon, ``class_name='road'``) per surviving
     blob, georeferenced through the raster transform.

``kwargs`` knobs:
- ``weights``         (str) — local .pth checkpoint path. Default points at
                      the bundled ``packages/core2/.../weights/road_unet.pth``.
- ``encoder_name``    (str, default "resnet34") — must match the encoder the
                      checkpoint was trained with.
- ``device``          (str, "cpu" / "cuda" / "mps"). Auto-selects when None.
- ``patch_size``      (int, default 512) — sliding-window patch size.
- ``stride``          (int | None) — stride between patches. Defaults to
                      ``patch_size // 2``.
- ``road_threshold``  (float, default 0.4) — sigmoid cutoff for the road
                      class.
- ``min_area_px``     (int, default 200) — drop polygons smaller than this
                      in pixels (kills speckle).
- ``simplify_px``     (float, default 1.0) — Douglas-Peucker tolerance in
                      pixel units before reprojecting to the raster CRS.
- ``class_name``      (str, default "road") — class label assigned to every
                      emitted detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.features import shapes as rio_shapes
from rasterio.transform import xy
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon, shape
from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection, Raster

from core2.processes.base import ProcessSpec
from core2.processes.registry import register

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())

# Repo root holds the trained checkpoint (committed alongside the other
# bundled weights like yolo26x-obb.pt). Resolved relative to this file:
# packages/core2/src/core2/processes/unet_roads.py → repo root is 5 levels up.
_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[5] / "road_unet.pth"
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class UnetRoadsProcess:
    """Run a U-Net road segmentation checkpoint over a raster."""

    spec: ProcessSpec
    name: str = "unet-roads"
    weights: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS)
    encoder_name: str = "resnet34"
    device: str | None = None
    patch_size: int = 512
    stride: int | None = None
    road_threshold: float = 0.4
    min_area_px: int = 200
    simplify_px: float = 1.0
    class_name: str = "road"
    _model: Any = field(default=None, init=False, repr=False)
    _resolved_device: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "UnetRoadsProcess":
        kw = spec.kwargs
        weights = kw.get("weights")
        return cls(
            spec=spec,
            name=spec.name,
            weights=Path(weights).expanduser() if weights else _DEFAULT_WEIGHTS,
            encoder_name=str(kw.get("encoder_name", "resnet34")),
            device=kw.get("device"),
            patch_size=int(kw.get("patch_size", 512)),
            stride=int(kw["stride"]) if kw.get("stride") is not None else None,
            road_threshold=float(kw.get("road_threshold", 0.4)),
            min_area_px=int(kw.get("min_area_px", 200)),
            simplify_px=float(kw.get("simplify_px", 1.0)),
            class_name=str(kw.get("class_name", "road")),
        )

    def _select_device(self) -> str:
        if self.device:
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import segmentation_models_pytorch as smp

        if not self.weights.exists():
            raise FileNotFoundError(
                f"unet-roads weights not found at {self.weights}. "
                "Train via run_roads_ml.py or pass kwargs.weights."
            )

        device = self._select_device()
        net = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=None,  # loaded from checkpoint
            in_channels=3,
            classes=1,
        )
        state = torch.load(self.weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        net.eval()
        net.to(device)
        self._model = net
        self._resolved_device = device
        _log.info(
            "unet-roads: loaded %s on %s (encoder=%s)",
            self.weights, device, self.encoder_name,
        )

    def run(self, raster: Raster) -> list[Detection]:
        allow = set(self.spec.classes) if self.spec.classes else None
        if allow is not None and self.class_name not in allow:
            return []

        self._load()

        img = raster.data
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(
                f"unet-roads expects HWC RGB raster, got shape {img.shape!r}"
            )
        rgb = img[:, :, :3]
        H, W = rgb.shape[:2]
        patch = self.patch_size
        stride = self.stride if self.stride is not None else patch // 2

        if H < patch or W < patch:
            raise ValueError(
                f"unet-roads: raster {W}x{H} smaller than patch_size={patch}; "
                "downsample or shrink the patch via kwargs.patch_size"
            )

        prob = self._infer(rgb, patch, stride)
        mask = (prob > self.road_threshold).astype(np.uint8)
        road_px = int(mask.sum())
        _log.info(
            "unet-roads: %d road px (%.2f%% area) — vectorising",
            road_px, 100.0 * road_px / mask.size,
        )
        if road_px == 0:
            return []

        return self._mask_to_detections(mask, raster)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _infer(self, rgb: np.ndarray, patch: int, stride: int) -> np.ndarray:
        import torch

        H, W = rgb.shape[:2]
        device = self._resolved_device or "cpu"
        prob_sum = np.zeros((H, W), dtype=np.float32)
        prob_cnt = np.zeros((H, W), dtype=np.float32)

        ys = sorted(set(list(range(0, H - patch + 1, stride)) + [H - patch]))
        xs = sorted(set(list(range(0, W - patch + 1, stride)) + [W - patch]))

        with torch.no_grad():
            for y in ys:
                for x in xs:
                    p = rgb[y:y + patch, x:x + patch].astype(np.float32) / 255.0
                    p = (p - _IMAGENET_MEAN) / _IMAGENET_STD
                    t = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).to(device)
                    out = torch.sigmoid(self._model(t)).squeeze().cpu().numpy()
                    prob_sum[y:y + patch, x:x + patch] += out
                    prob_cnt[y:y + patch, x:x + patch] += 1.0

        return prob_sum / np.maximum(prob_cnt, 1.0)

    def _mask_to_detections(
        self, mask: np.ndarray, raster: Raster,
    ) -> list[Detection]:
        # Build pixel-space polygons first, drop tiny ones, simplify, then
        # reproject through the raster transform. Working in pixel space lets
        # us threshold area in px (consistent regardless of CRS) before paying
        # the per-vertex transform cost.
        detections: list[Detection] = []
        polys_px: list[Polygon] = []
        for geom_dict, val in rio_shapes(mask, mask=mask.astype(bool)):
            if val != 1:
                continue
            geom = shape(geom_dict)
            if geom.is_empty or not isinstance(geom, Polygon):
                continue
            if geom.area < self.min_area_px:
                continue
            if self.simplify_px > 0:
                simplified = geom.simplify(self.simplify_px, preserve_topology=True)
                if simplified.is_empty or not isinstance(simplified, Polygon):
                    continue
                geom = simplified
            # Drop degenerate survivors that would make oriented_envelope NaN
            # in the renderer: zero-area (collinear) or fewer than 4 ring pts.
            if not geom.is_valid or geom.area <= 0:
                continue
            if len(geom.exterior.coords) < 4:
                continue
            polys_px.append(geom)

        if not polys_px:
            return []

        a = raster.transform
        # Affine for shapely.affinity.affine_transform: [a, b, d, e, xoff, yoff]
        # rasterio.transform.Affine = (a, b, c, d, e, f) where x = a*col + b*row + c
        # rio_shapes returns coords as (col, row) pairs, so apply (a, b, d, e, c, f).
        affine_params = [a.a, a.b, a.d, a.e, a.c, a.f]

        for poly_px in polys_px:
            poly_geo: BaseGeometry = affine_transform(poly_px, affine_params)
            if poly_geo.is_empty or not poly_geo.is_valid or poly_geo.area <= 0:
                continue
            minx, miny, maxx, maxy = poly_geo.bounds
            cx_px, cy_px = poly_px.centroid.x, poly_px.centroid.y
            # poly_px.bounds is (col_min, row_min, col_max, row_max).
            c0, r0, c1, r1 = poly_px.bounds
            pixel_bbox = (int(c0), int(r0), int(c1), int(r1))
            centroid_xy = xy(raster.transform, cy_px, cx_px)
            detections.append(
                Detection(
                    id=len(detections),
                    class_name=self.class_name,
                    confidence=1.0,
                    bbox=(minx, miny, maxx, maxy),
                    pixel_bbox=pixel_bbox,
                    centroid=Point(float(centroid_xy[0]), float(centroid_xy[1])),
                    source_model=self.name,
                    geometry=poly_geo,
                )
            )

        _log.info("unet-roads: emitted %d road polygons", len(detections))
        return detections


register("unet-roads", UnetRoadsProcess.from_spec)

__all__ = ["UnetRoadsProcess"]
