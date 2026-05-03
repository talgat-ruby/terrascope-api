"""Grounded-SAM — GroundingDINO (text → bboxes) + Meta SAM (bboxes → masks).

This is the de-facto pipeline for "SAM detects X by name". Vanilla SAM
is class-agnostic; GroundingDINO is an open-vocabulary detector that
takes a text prompt (e.g. "road. building.") and returns labelled
bboxes. We feed those bboxes to SAM as box prompts to get tight masks,
then emit one bbox `Detection` per mask using the GroundingDINO label.

Detection is bbox-only (the orchestrator's contract today). The GD
bbox is already class-aware, so unlike `sam-rs` there is no shape-rule
guessing — the label comes straight from GD.

`kwargs` knobs:

- `gdino_model_id`        (str, default
                           `IDEA-Research/grounding-dino-tiny`) — HF
                           repo. Use `grounding-dino-base` for higher
                           quality at ~3x cost.
- `sam_weights`           (str, default `sam_vit_b_01ec64.pth`) — Meta
                           SAM checkpoint filename or local path.
- `sam_model_type`        (str, default `vit_b`).
- `weights_dir`           (str, default `./tmp/weights`).
- `device`                (str, default `cpu`).
- `box_threshold`         (float, default 0.30) — GD box-confidence cut.
- `text_threshold`        (float, default 0.25) — GD text-confidence cut.
- `max_long_side_px`      (int, default 1536) — downsample huge rasters
                           before inference. GD struggles above ~2048
                           on CPU and SAM image-encoder is per-pixel
                           expensive.
- `extra_text_prompts`    (dict[str, list[str]], optional) — synonyms
                           per class to broaden GD's recall. Example:
                           `{"road": ["street", "highway"]}`.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
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

_log = logging.getLogger("core2.grounded_sam")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[grounded-sam] %(asctime)s %(levelname)s %(message)s")
    )
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False

_SAM_URLS = {
    "sam_vit_h_4b8939.pth": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    ),
    "sam_vit_l_0b3195.pth": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    ),
    "sam_vit_b_01ec64.pth": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    ),
}


@dataclass
class GroundedSamProcess:
    """GroundingDINO + SAM pipeline driven by `spec.classes`."""

    spec: ProcessSpec
    name: str = "grounded-sam"
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_weights: str = "sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"
    weights_dir: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS_CACHE)
    device: str = "cpu"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    max_long_side_px: int = 1536
    classes: tuple[str, ...] = ("road", "building")
    extra_text_prompts: dict[str, list[str]] = field(default_factory=dict)
    _gd_processor: Any = field(default=None, init=False, repr=False)
    _gd_model: Any = field(default=None, init=False, repr=False)
    _sam_predictor: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "GroundedSamProcess":
        kw = spec.kwargs
        if spec.classes is None or len(spec.classes) == 0:
            raise ValueError(
                "grounded-sam requires spec.classes — at least one class "
                "name to use as a GroundingDINO text prompt."
            )

        extra = kw.get("extra_text_prompts") or {}
        if not isinstance(extra, dict):
            raise ValueError("kwargs.extra_text_prompts must be a dict")
        normalized: dict[str, list[str]] = {}
        for k, v in extra.items():
            if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
                raise ValueError(
                    f"kwargs.extra_text_prompts[{k!r}] must be a list of strings"
                )
            normalized[k] = list(v)

        return cls(
            spec=spec,
            gdino_model_id=str(
                kw.get("gdino_model_id", "IDEA-Research/grounding-dino-tiny")
            ),
            sam_weights=str(kw.get("sam_weights", "sam_vit_b_01ec64.pth")),
            sam_model_type=str(kw.get("sam_model_type", "vit_b")),
            weights_dir=Path(
                kw.get("weights_dir", _DEFAULT_WEIGHTS_CACHE)
            ).expanduser(),
            device=str(kw.get("device", "cpu")),
            box_threshold=float(kw.get("box_threshold", 0.30)),
            text_threshold=float(kw.get("text_threshold", 0.25)),
            max_long_side_px=int(kw.get("max_long_side_px", 1536)),
            classes=tuple(spec.classes),
            extra_text_prompts=normalized,
        )

    def _resolve_sam_weights(self) -> str:
        w = self.sam_weights
        if Path(w).exists():
            _log.info(
                "using local SAM weights %s (%.1f MB)",
                w,
                Path(w).stat().st_size / (1024 * 1024),
            )
            return w
        if w not in _SAM_URLS:
            raise FileNotFoundError(
                f"SAM weights {w!r} not found locally and not a known Meta "
                f"filename. Use one of: {', '.join(sorted(_SAM_URLS))}."
            )
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        local = self.weights_dir / w
        if local.exists():
            _log.info(
                "using cached SAM weights %s (%.1f MB)",
                local,
                local.stat().st_size / (1024 * 1024),
            )
            return str(local)
        url = _SAM_URLS[w]
        _log.info("downloading SAM weights %s -> %s", url, local)
        urllib.request.urlretrieve(url, local)
        _log.info(
            "download complete (%.1f MB)",
            local.stat().st_size / (1024 * 1024),
        )
        return str(local)

    def _load(self) -> None:
        if self._gd_model is not None and self._sam_predictor is not None:
            return
        _log.info("loading GroundingDINO from HF: %s", self.gdino_model_id)
        t0 = time.monotonic()
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self._gd_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gdino_model_id
        ).to(self.device)
        self._gd_model.eval()
        _log.info("  GroundingDINO ready in %.1fs", time.monotonic() - t0)

        _log.info("loading SAM (%s)...", self.sam_model_type)
        t0 = time.monotonic()
        from segment_anything import (  # type: ignore[import-untyped]
            SamPredictor,
            sam_model_registry,
        )

        sam_path = self._resolve_sam_weights()
        sam = sam_model_registry[self.sam_model_type](checkpoint=sam_path)
        sam.to(device=self.device)
        self._sam_predictor = SamPredictor(sam)
        _log.info("  SAM ready in %.1fs", time.monotonic() - t0)

    def run(self, raster: Raster) -> list[Detection]:
        _log.info(
            "run start: raster %dx%d, device=%s, classes=%s",
            raster.width,
            raster.height,
            self.device,
            list(self.classes),
        )
        self._load()

        image, scale = _maybe_downsample(raster.data, self.max_long_side_px)
        if scale != 1.0:
            _log.info(
                "downsampled %dx%d -> %dx%d (scale=%.3f)",
                raster.width,
                raster.height,
                image.shape[1],
                image.shape[0],
                scale,
            )

        # GD wants a single text query with periods between classes;
        # synonyms get folded into the same query and we map labels back
        # to the canonical class name post-hoc.
        prompt, label_map = _build_prompt(self.classes, self.extra_text_prompts)
        _log.info("GroundingDINO text prompt: %r", prompt)

        boxes, scores, labels = self._gdino_detect(image, prompt)
        _log.info("GroundingDINO returned %d candidate boxes", len(boxes))
        if len(boxes) == 0:
            _log.warning(
                "GD returned zero boxes — try lowering kwargs.box_threshold "
                "(currently %.2f) or kwargs.text_threshold (%.2f).",
                self.box_threshold,
                self.text_threshold,
            )
            return []

        _log.info("running SAM on %d boxes (heartbeats every 30s)...", len(boxes))
        t0 = time.monotonic()
        stop = threading.Event()
        watch = threading.Thread(target=_heartbeat, args=(stop, t0), daemon=True)
        watch.start()
        try:
            self._sam_predictor.set_image(image)
            mask_boxes = self._sam_predict_boxes(boxes)
        finally:
            stop.set()
            watch.join(timeout=1.0)
        _log.info(
            "SAM produced %d mask bboxes in %.1fs",
            len(mask_boxes),
            time.monotonic() - t0,
        )

        # Rescale bboxes back to original raster pixel grid.
        inv = 1.0 / scale if scale != 0 else 1.0

        detections: list[Detection] = []
        kept_per_class: dict[str, int] = {c: 0 for c in self.classes}
        unmatched = 0
        for i, ((c0, r0, c1, r1), score, raw_label) in enumerate(
            zip(mask_boxes, scores, labels, strict=True)
        ):
            class_name = label_map.get(raw_label.lower().strip())
            if class_name is None:
                unmatched += 1
                continue
            c0 = int(c0 * inv)
            r0 = int(r0 * inv)
            c1 = int(c1 * inv)
            r1 = int(r1 * inv)

            x0, y0 = raster.transform * (c0, r1)
            x1, y1 = raster.transform * (c1, r0)
            minx, maxx = sorted((x0, x1))
            miny, maxy = sorted((y0, y1))
            geo_bbox = (minx, miny, maxx, maxy)

            detections.append(
                Detection(
                    id=i,
                    class_name=class_name,
                    confidence=float(score),
                    bbox=geo_bbox,
                    pixel_bbox=(c0, r0, c1, r1),
                    centroid=box(*geo_bbox).centroid,
                    source_model=self.name,
                )
            )
            kept_per_class[class_name] += 1

        _log.info(
            "kept %d / %d detections (per-class: %s, unmapped labels: %d)",
            len(detections),
            len(mask_boxes),
            ", ".join(f"{k}={v}" for k, v in kept_per_class.items()),
            unmatched,
        )
        return detections

    def _gdino_detect(
        self, image: np.ndarray, prompt: str
    ) -> tuple[list[tuple[float, float, float, float]], list[float], list[str]]:
        import torch  # type: ignore[import-untyped]

        pil = Image.fromarray(image)
        inputs = self._gd_processor(
            images=pil, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._gd_model(**inputs)
        results = self._gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]

        boxes_t = results["boxes"].detach().cpu().numpy()
        scores_t = results["scores"].detach().cpu().numpy()
        labels_t = results["labels"]
        boxes = [tuple(map(float, b)) for b in boxes_t]
        scores = [float(s) for s in scores_t]
        labels = [str(l) for l in labels_t]
        return boxes, scores, labels

    def _sam_predict_boxes(
        self, boxes: list[tuple[float, float, float, float]]
    ) -> list[tuple[int, int, int, int]]:
        """Run SAM box-prompt prediction; return tight bboxes around masks."""
        import torch  # type: ignore[import-untyped]

        boxes_np = np.asarray(boxes, dtype=np.float32)
        boxes_t = torch.as_tensor(boxes_np, device=self.device)
        transformed = self._sam_predictor.transform.apply_boxes_torch(
            boxes_t, self._sam_predictor.original_size
        )
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        masks_np = masks.detach().cpu().numpy()  # (N, 1, H, W)
        out: list[tuple[int, int, int, int]] = []
        for m in masks_np[:, 0, :, :]:
            ys, xs = np.where(m)
            if xs.size == 0:
                # Fall back to the GD box if SAM returned an empty mask.
                gx0, gy0, gx1, gy1 = boxes[len(out)]
                out.append((int(gx0), int(gy0), int(gx1), int(gy1)))
                continue
            out.append(
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            )
        return out


def _build_prompt(
    classes: tuple[str, ...],
    extra: dict[str, list[str]],
) -> tuple[str, dict[str, str]]:
    """Build a GD text query and a label->canonical-class lookup.

    GroundingDINO uses period-separated prompts; matched labels come back
    as substrings of the prompt. We pre-build a lower-cased lookup so
    synonyms map back to the canonical class.
    """
    parts: list[str] = []
    label_map: dict[str, str] = {}
    for canonical in classes:
        terms = [canonical, *extra.get(canonical, [])]
        for t in terms:
            t_norm = t.lower().strip()
            label_map[t_norm] = canonical
            parts.append(t_norm)
    prompt = ". ".join(parts) + "."
    return prompt, label_map


def _maybe_downsample(
    image: np.ndarray, max_long_side_px: int
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if max_long_side_px <= 0 or long_side <= max_long_side_px:
        return image, 1.0
    scale = max_long_side_px / long_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil), scale


def _heartbeat(stop: threading.Event, started_at: float) -> None:
    while not stop.wait(30.0):
        _log.info(
            "  ... still running (elapsed %.0fs)", time.monotonic() - started_at
        )


register("grounded-sam", GroundedSamProcess.from_spec)


__all__ = ["GroundedSamProcess"]
