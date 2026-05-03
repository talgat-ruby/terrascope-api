"""YOLO + SAHI sliced inference process.

Owns the full pipeline: optional GSD downsample → SAHI slice predict →
project bboxes back into the raster's native pixel grid → georeference via
the raster transform. Loads the model exactly once per Process instance
(unlike the standalone `run_detection.py` script that reloaded per
threshold).

Compared to `core.detection.YoloSahiDetector`, this version exposes a
`source_gsd_m_per_px` / `model_gsd_m_per_px` pair so a fine-tuned
checkpoint trained at a different ground sample distance can be evaluated
against high-resolution rasters without manual preprocessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import box

from core.detection.types import Detection, Raster

from core2.processes.base import ProcessSpec
from core2.processes.registry import register

_DEFAULT_WEIGHTS_CACHE = Path("./tmp/weights")


@dataclass
class YoloSahiProcess:
    """Run an Ultralytics YOLO checkpoint via SAHI sliced inference.

    `kwargs` knobs:
    - `weights`              (str) — local .pt path, an Ultralytics alias
                              (e.g. `yolov8n.pt`), or a HuggingFace Hub
                              repo id (`<user>/<repo>`). HF repos are
                              downloaded once into `weights_dir` and the
                              local .pt path is handed to Ultralytics.
    - `weights_dir`          (str, default `./tmp/weights`) — local cache
                              directory for HF-resolved checkpoints.
    - `weights_filename`     (str | None) — explicit filename inside the
                              HF repo (e.g. `best.pt`). When omitted, the
                              first `*.pt` listed in the repo is used.
    - `device`               (str, "cpu" or "cuda").
    - `slice_size`           (int, default 640).
    - `overlap_ratio`        (float, default 0.2).
    - `full_image_threshold` (int, default 1024) — long-side under which we
                              skip slicing and run a single forward pass.
    - `source_gsd_m_per_px`  (float | None) — raster's ground sample
                              distance. Defaults to None (no rescale).
    - `model_gsd_m_per_px`   (float | None) — checkpoint's training GSD.
                              When both are set, the raster is downsampled
                              to match before inference.
    """

    spec: ProcessSpec
    name: str
    weights: str = "yolov26x.pt"
    device: str = "cpu"
    slice_size: int = 640
    overlap_ratio: float = 0.2
    full_image_threshold: int = 1024
    source_gsd_m_per_px: float | None = None
    model_gsd_m_per_px: float | None = None
    weights_dir: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS_CACHE)
    weights_filename: str | None = None
    _model: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "YoloSahiProcess":
        kw = spec.kwargs
        return cls(
            spec=spec,
            name=spec.name,
            weights=str(kw.get("weights", "yolov26x.pt")),
            device=str(kw.get("device", "cpu")),
            slice_size=int(kw.get("slice_size", 640)),
            overlap_ratio=float(kw.get("overlap_ratio", 0.2)),
            full_image_threshold=int(kw.get("full_image_threshold", 1024)),
            source_gsd_m_per_px=_optional_float(kw.get("source_gsd_m_per_px")),
            model_gsd_m_per_px=_optional_float(kw.get("model_gsd_m_per_px")),
            weights_dir=Path(
                kw.get("weights_dir", _DEFAULT_WEIGHTS_CACHE)
            ).expanduser(),
            weights_filename=kw.get("weights_filename"),
        )

    def _resolve_weights(self) -> str:
        """Return a local filesystem path to a `.pt` checkpoint.

        - Local file or Ultralytics alias (no slash) → returned as-is;
          Ultralytics handles its own download to its package dir.
        - HF Hub id (`<user>/<repo>`) → fetched via huggingface_hub into
          `self.weights_dir` and the local path is returned.
        """
        w = self.weights
        if os.path.exists(w):
            return w
        if not _looks_like_hf_repo(w):
            return w  # let Ultralytics resolve aliases / its own search

        from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-untyped]
        from huggingface_hub.utils import (  # type: ignore[import-untyped]
            GatedRepoError,
            RepositoryNotFoundError,
        )

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        try:
            filename = self.weights_filename
            if filename is None:
                api = HfApi()
                files = api.list_repo_files(w)
                pt_files = [f for f in files if f.endswith(".pt")]
                if not pt_files:
                    raise FileNotFoundError(
                        f"No .pt files found in HF repo {w!r}; "
                        "set kwargs.weights_filename explicitly."
                    )
                filename = pt_files[0]

            local = hf_hub_download(
                repo_id=w,
                filename=filename,
                local_dir=str(self.weights_dir / w.replace("/", "__")),
            )
            return local
        except (RepositoryNotFoundError, GatedRepoError) as e:
            raise FileNotFoundError(
                f"HuggingFace repo {w!r} is unreachable ({e.__class__.__name__}). "
                "It may be private, gated, or removed. Either authenticate "
                "(`huggingface-cli login`), pass a local checkpoint via "
                "kwargs.weights, or pick a different model."
            ) from e

    def _load(self) -> None:
        if self._model is not None:
            return
        from sahi import AutoDetectionModel  # type: ignore[import-untyped]

        weights_path = self._resolve_weights()

        # Load with the lowest threshold we'd ever need (0.0); the spec's
        # `min_confidence` is enforced downstream by the orchestrator. This
        # mirrors the cheap-post-filter pattern from run_detection.py and
        # avoids ever reloading the model just to sweep thresholds.
        self._model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,
            confidence_threshold=0.0,
            device=self.device,
        )

    def run(self, raster: Raster) -> list[Detection]:
        self._load()
        scale = self._scale()
        image = raster.data
        if scale != 1.0:
            image = _resize(image, scale)

        long_side = max(image.shape[0], image.shape[1])
        if long_side <= self.full_image_threshold:
            from sahi.predict import get_prediction  # type: ignore[import-untyped]

            result = get_prediction(image=image, detection_model=self._model)
        else:
            from sahi.predict import get_sliced_prediction  # type: ignore[import-untyped]

            result = get_sliced_prediction(
                image=image,
                detection_model=self._model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
            )
        return self._to_detections(result.object_prediction_list, raster, scale)

    def _scale(self) -> float:
        if self.source_gsd_m_per_px is None or self.model_gsd_m_per_px is None:
            return 1.0
        if self.model_gsd_m_per_px <= 0 or self.source_gsd_m_per_px <= 0:
            return 1.0
        return self.source_gsd_m_per_px / self.model_gsd_m_per_px

    def _to_detections(
        self, predictions: list, raster: Raster, scale: float
    ) -> list[Detection]:
        inv = 1.0 / scale if scale != 0 else 1.0
        detections: list[Detection] = []
        for i, pred in enumerate(predictions):
            b = pred.bbox  # pixel coords in the (possibly rescaled) image
            c0 = int(b.minx * inv)
            r0 = int(b.miny * inv)
            c1 = int(b.maxx * inv)
            r1 = int(b.maxy * inv)

            x0, y0 = raster.transform * (c0, r1)
            x1, y1 = raster.transform * (c1, r0)
            minx, maxx = sorted((x0, x1))
            miny, maxy = sorted((y0, y1))
            geo_bbox = (minx, miny, maxx, maxy)

            detections.append(
                Detection(
                    id=i,
                    class_name=str(pred.category.name),
                    confidence=float(pred.score.value),
                    bbox=geo_bbox,
                    pixel_bbox=(c0, r0, c1, r1),
                    centroid=box(*geo_bbox).centroid,
                    source_model=self.name,
                )
            )
        return detections


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Downsample (or upsample) an HWC uint8 image with PIL/LANCZOS."""
    h, w = image.shape[:2]
    new = (max(1, int(w * scale)), max(1, int(h * scale)))
    return np.asarray(Image.fromarray(image).resize(new, Image.LANCZOS))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _looks_like_hf_repo(value: str) -> bool:
    """Heuristic: HF repo ids look like `user/repo`, not local paths.

    Treat as HF when the string has exactly one slash and doesn't start
    with `.` or `/` (which would indicate a relative or absolute file
    path).
    """
    if value.startswith((".", "/")):
        return False
    if value.count("/") != 1:
        return False
    return True


# Register a few presets keyed by the same names core uses, so users can
# port configs across pipelines with minimal change. Custom finetunes can
# always be configured ad-hoc by passing `kwargs.weights`.
def _register_yolo_presets() -> None:
    presets = {
        "yolov8n-sahi": "yolov8n.pt",
        "yolov8-obb-aerial": "yolov8s-obb.pt",
        "yolov8-obb-dota-v2": "yolov8x-obb.pt",
        "yolov8-satellite-vehicle": "keremberke/yolov8m-satellite-vehicle-detection",
        # YOLO26 (Ultralytics, late-2025 successor to YOLO11). Same loading
        # path as YOLOv8 — Ultralytics auto-downloads from
        # github.com/ultralytics/assets on first use.
        "yolo26n-sahi": "yolo26n.pt",
        "yolo26m-sahi": "yolo26m.pt",
        "yolo26-obb": "yolo26x-obb.pt",
        "yolo26-seg": "yolo26x-seg.pt",
    }
    for key, default_weights in presets.items():

        def builder(spec: ProcessSpec, _w: str = default_weights) -> YoloSahiProcess:
            spec_kwargs = dict(spec.kwargs)
            spec_kwargs.setdefault("weights", _w)
            patched = ProcessSpec(
                name=spec.name,
                classes=spec.classes,
                min_confidence=spec.min_confidence,
                kwargs=spec_kwargs,
            )
            return YoloSahiProcess.from_spec(patched)

        register(key, builder)


_register_yolo_presets()


__all__ = ["YoloSahiProcess"]
