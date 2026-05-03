"""SAM-RS — Meta Segment Anything for class-shape detection (bbox-only).

Vanilla SAM is class-agnostic: it segments every "thing" it sees, with
no notion of "road" or "building". To turn it into a labelled detector
we run `SamAutomaticMaskGenerator` over the raster, then assign each
mask to the first class whose shape rule it satisfies. Classes are
chosen by the job config's `classes` list — currently `road` and
`building` are supported out-of-the-box.

Per-class shape defaults (override via `kwargs.class_rules`):
  road:     elongated (long/short >= 4), long-side >= 80 px
  building: roughly compact (long/short <= 2.5), long-side <= 400 px

Each detection carries both an axis-aligned `bbox`/`pixel_bbox` and a
`geometry` set to the **oriented bounding box** (minimum rotated
rectangle) of the underlying SAM mask. Renderers/exporters that respect
`geometry` get a tight parallelogram-shaped quad for diagonal roads
instead of the loose AA envelope.

`kwargs` knobs:

- `weights`              (str) — local .pth checkpoint path OR the
                          official Meta filename
                          (`sam_vit_h_4b8939.pth`,
                          `sam_vit_l_0b3195.pth`,
                          `sam_vit_b_01ec64.pth`). Filenames are fetched
                          from Meta's CDN into `weights_dir`.
- `model_type`           (str, default `vit_h`) — `vit_h | vit_l | vit_b`.
                          Must match the checkpoint.
- `weights_dir`          (str, default `./tmp/weights`) — local cache.
- `device`               (str, default `cpu`) — `cpu` or `cuda`.
- `class_rules`          (dict, optional) — per-class shape rules. Keys
                          are class names; values are dicts with any of:
                          `min_elongation`, `max_elongation`,
                          `min_long_side_px`, `max_long_side_px`,
                          `min_area_px`, `max_area_frac`. When the job's
                          `classes` is set, only those classes are used;
                          masks matching none are dropped. When unset,
                          the default rules for `road` and `building`
                          are used.
- `points_per_side`      (int, default 32) — SAM grid density.
- `pred_iou_thresh`      (float, default 0.86).
- `stability_score_thresh` (float, default 0.92).
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
from rasterio.transform import Affine
from shapely.geometry import MultiPoint, Polygon, box

from core.detection.types import Detection, Raster

from core2.processes.base import ProcessSpec
from core2.processes.registry import register

_DEFAULT_WEIGHTS_CACHE = Path("./tmp/weights")

_log = logging.getLogger("core2.sam_rs")
if not _log.handlers:
    # Default to a stderr handler so CLI users see progress without
    # having to configure logging upstream. Honour any pre-existing
    # handler if the host app already configured logging.
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[sam-rs] %(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False

_DEFAULT_CLASS_RULES: dict[str, dict[str, float]] = {
    # Order matters: the classifier picks the FIRST matching class, so
    # `building` is checked before `road`. With looser rules, masks can
    # match both — assigning compact-ish shapes to "building" first
    # avoids labelling rectangular buildings as roads.
    "building": {
        "max_elongation": 4.0,
        "min_long_side_px": 8,
        "max_long_side_px": 600,
        "min_area_px": 40,
        "max_area_frac": 0.15,
    },
    # Road segments out of SAM are usually short fragments, not full
    # roads. With elongation > 4 anything left after the building rule
    # is genuinely skinny.
    "road": {
        "min_elongation": 4.0,
        "min_long_side_px": 30,
        "min_area_px": 150,
        "max_area_frac": 0.5,
    },
}


_OFFICIAL_URLS = {
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

_DEFAULT_WEIGHTS_FOR_MODEL = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


@dataclass
class SamRsProcess:
    """Meta SAM ViT-H + automatic mask generation, classified by shape rules.

    The job config's `classes` controls which class buckets are emitted;
    each surviving mask is assigned to the first matching class.
    """

    spec: ProcessSpec
    name: str = "sam-rs"
    weights: str = "sam_vit_h_4b8939.pth"
    model_type: str = "vit_h"
    weights_dir: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS_CACHE)
    device: str = "cpu"
    class_rules: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(_DEFAULT_CLASS_RULES)
    )
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    # Cap the long side fed to SAM. SAM's ViT runs internally at 1024
    # regardless of input, but the automatic mask generator does extra
    # work proportional to input resolution. CPU at 88MP is hours;
    # downsampling to 2048 keeps detection useful and runtime sane.
    max_long_side_px: int = 2048
    _generator: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "SamRsProcess":
        kw = spec.kwargs

        # Merge user-supplied per-class rules over defaults so users only
        # need to override the knobs they care about.
        rules: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in _DEFAULT_CLASS_RULES.items()
        }
        user_rules = kw.get("class_rules") or {}
        if not isinstance(user_rules, dict):
            raise ValueError("kwargs.class_rules must be a dict[str, dict]")
        for cls_name, overrides in user_rules.items():
            if not isinstance(overrides, dict):
                raise ValueError(
                    f"kwargs.class_rules[{cls_name!r}] must be a dict"
                )
            rules.setdefault(cls_name, {}).update(
                {k: float(v) for k, v in overrides.items()}
            )

        # If the job restricts classes, drop unrelated rules so we never
        # accidentally classify a mask as something the user didn't ask
        # for. Unknown class names are an error — better to fail loudly
        # than silently emit nothing.
        if spec.classes is not None:
            unknown = [c for c in spec.classes if c not in rules]
            if unknown:
                raise ValueError(
                    f"sam-rs has no shape rules for classes: {unknown!r}. "
                    f"Add them via kwargs.class_rules. Known: "
                    f"{sorted(rules)!r}"
                )
            rules = {c: rules[c] for c in spec.classes}

        model_type = str(kw.get("model_type", "vit_h"))
        weights = kw.get("weights")
        if weights is None:
            if model_type not in _DEFAULT_WEIGHTS_FOR_MODEL:
                raise ValueError(
                    f"sam-rs: unknown model_type {model_type!r}. "
                    f"Known: {sorted(_DEFAULT_WEIGHTS_FOR_MODEL)!r}. "
                    f"Pass kwargs.weights explicitly to use a custom checkpoint."
                )
            weights = _DEFAULT_WEIGHTS_FOR_MODEL[model_type]

        return cls(
            spec=spec,
            weights=str(weights),
            model_type=model_type,
            weights_dir=Path(
                kw.get("weights_dir", _DEFAULT_WEIGHTS_CACHE)
            ).expanduser(),
            device=str(kw.get("device", "cpu")),
            class_rules=rules,
            points_per_side=int(kw.get("points_per_side", 32)),
            points_per_batch=int(kw.get("points_per_batch", 64)),
            pred_iou_thresh=float(kw.get("pred_iou_thresh", 0.86)),
            stability_score_thresh=float(kw.get("stability_score_thresh", 0.92)),
            max_long_side_px=int(kw.get("max_long_side_px", 2048)),
        )

    def _resolve_weights(self) -> str:
        w = self.weights
        if Path(w).exists():
            size_mb = Path(w).stat().st_size / (1024 * 1024)
            _log.info("using local weights %s (%.1f MB)", w, size_mb)
            return w
        if w in _OFFICIAL_URLS:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            local = self.weights_dir / w
            if local.exists():
                size_mb = local.stat().st_size / (1024 * 1024)
                _log.info("using cached weights %s (%.1f MB)", local, size_mb)
                return str(local)
            url = _OFFICIAL_URLS[w]
            _log.info("downloading SAM weights %s -> %s", url, local)
            t0 = time.monotonic()
            _last_pct = [-1]

            def _hook(blocks: int, block_size: int, total: int) -> None:
                if total <= 0:
                    return
                done = min(blocks * block_size, total)
                pct = int(done * 100 / total)
                if pct != _last_pct[0] and pct % 5 == 0:
                    _last_pct[0] = pct
                    _log.info(
                        "  download %d%% (%.1f / %.1f MB)",
                        pct,
                        done / (1024 * 1024),
                        total / (1024 * 1024),
                    )

            urllib.request.urlretrieve(url, local, reporthook=_hook)
            _log.info(
                "download complete in %.1fs (%.1f MB)",
                time.monotonic() - t0,
                local.stat().st_size / (1024 * 1024),
            )
            return str(local)
        raise FileNotFoundError(
            f"SAM weights {w!r} not found locally and not a known Meta filename. "
            f"Pass kwargs.weights as a local .pth path or one of: "
            f"{', '.join(sorted(_OFFICIAL_URLS))}."
        )

    def _load(self) -> None:
        if self._generator is not None:
            return
        _log.info("importing segment_anything...")
        from segment_anything import (  # type: ignore[import-untyped]
            SamAutomaticMaskGenerator,
            sam_model_registry,
        )

        weights_path = self._resolve_weights()
        _log.info("building SAM %s from checkpoint...", self.model_type)
        t0 = time.monotonic()
        sam = sam_model_registry[self.model_type](checkpoint=weights_path)
        _log.info("  built in %.1fs, moving to device=%s", time.monotonic() - t0, self.device)
        t0 = time.monotonic()
        sam.to(device=self.device)
        _log.info("  moved to %s in %.1fs", self.device, time.monotonic() - t0)
        self._generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        _log.info(
            "mask generator ready (points_per_side=%d, points_per_batch=%d, "
            "pred_iou=%.2f, stability=%.2f)",
            self.points_per_side,
            self.points_per_batch,
            self.pred_iou_thresh,
            self.stability_score_thresh,
        )

    def run(self, raster: Raster) -> list[Detection]:
        _log.info(
            "run start: raster %dx%d, device=%s, model=%s",
            raster.width,
            raster.height,
            self.device,
            self.model_type,
        )
        self._load()

        # Downsample if the input is huge — SAM's auto mask generator
        # does work proportional to input resolution, and CPU vit_h on a
        # ~90MP raster is effectively a non-terminating job.
        image, scale = _maybe_downsample(raster.data, self.max_long_side_px)
        if scale != 1.0:
            _log.info(
                "downsampled raster %dx%d -> %dx%d (scale=%.3f) for SAM",
                raster.width,
                raster.height,
                image.shape[1],
                image.shape[0],
                scale,
            )

        n_points = self.points_per_side * self.points_per_side
        n_batches = max(1, (n_points + self.points_per_batch - 1) // self.points_per_batch)
        _log.info(
            "generating masks: %d prompt points in %d batch(es) of %d "
            "(this is the slow step — heartbeats every 30s)",
            n_points,
            n_batches,
            self.points_per_batch,
        )
        if self.device == "cpu" and self.model_type == "vit_h":
            _log.warning(
                "vit_h on CPU is SLOW. If this takes too long, set "
                "kwargs.model_type='vit_b' (~10x faster), lower "
                "kwargs.points_per_side (e.g. 16), or lower "
                "kwargs.max_long_side_px (e.g. 1024)."
            )

        t0 = time.monotonic()
        stop = threading.Event()
        watchdog = threading.Thread(
            target=_heartbeat, args=(stop, t0), daemon=True
        )
        watchdog.start()
        try:
            masks = self._generator.generate(image)
        finally:
            stop.set()
            watchdog.join(timeout=1.0)
        _log.info(
            "generated %d masks in %.1fs", len(masks), time.monotonic() - t0
        )

        # Rescale bboxes back into the original raster's pixel grid so
        # geographic projection via raster.transform stays correct.
        if scale != 1.0:
            inv = 1.0 / scale
            for m in masks:
                x, y, bw, bh = m["bbox"]
                m["bbox"] = [x * inv, y * inv, bw * inv, bh * inv]
                m["area"] = float(m.get("area", bw * bh)) * inv * inv

        h, w = raster.data.shape[:2]
        tile_area = float(h * w)

        detections: list[Detection] = []
        idx = 0
        kept_per_class: dict[str, int] = {c: 0 for c in self.class_rules}
        unmatched = 0
        for m in masks:
            x, y, bw, bh = m["bbox"]  # XYWH pixel coords from SAM
            c0, r0 = int(x), int(y)
            c1, r1 = int(x + bw), int(y + bh)
            long_side = float(max(bw, bh))
            short_side = max(1.0, float(min(bw, bh)))
            elongation = long_side / short_side
            area = float(m.get("area", bw * bh))

            class_name = _classify(
                self.class_rules,
                area=area,
                tile_area=tile_area,
                long_side=long_side,
                elongation=elongation,
            )
            if class_name is None:
                unmatched += 1
                continue

            x0, y0 = raster.transform * (c0, r1)
            x1, y1 = raster.transform * (c1, r0)
            minx, maxx = sorted((x0, x1))
            miny, maxy = sorted((y0, y1))
            geo_bbox = (minx, miny, maxx, maxy)

            confidence = float(
                m.get("predicted_iou", m.get("stability_score", 1.0))
            )

            geometry = _mask_to_obb_geometry(
                m.get("segmentation"), raster.transform, scale,
            )

            detections.append(
                Detection(
                    id=idx,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=geo_bbox,
                    pixel_bbox=(c0, r0, c1, r1),
                    centroid=box(*geo_bbox).centroid,
                    source_model=self.name,
                    geometry=geometry,
                )
            )
            kept_per_class[class_name] += 1
            idx += 1

        _log.info(
            "kept %d / %d masks (per-class: %s, unmatched: %d)",
            len(detections),
            len(masks),
            ", ".join(f"{k}={v}" for k, v in kept_per_class.items()),
            unmatched,
        )
        return detections


def _maybe_downsample(
    image: np.ndarray, max_long_side_px: int
) -> tuple[np.ndarray, float]:
    """Downsample HWC uint8 image so its long side <= max_long_side_px.

    Returns (image, scale) where scale = new_size / original_size.
    A scale of 1.0 means no resize was needed.
    """
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
    """Print a heartbeat every 30s while SAM.generate() is opaque to us."""
    while not stop.wait(30.0):
        elapsed = time.monotonic() - started_at
        _log.info("  ... still generating masks (elapsed %.0fs)", elapsed)


def _classify(
    rules: dict[str, dict[str, float]],
    *,
    area: float,
    tile_area: float,
    long_side: float,
    elongation: float,
) -> str | None:
    """Return the first class whose shape rule the mask satisfies, or None."""
    for class_name, r in rules.items():
        if area < r.get("min_area_px", 0):
            continue
        max_area_frac = r.get("max_area_frac")
        if max_area_frac is not None and area > max_area_frac * tile_area:
            continue
        if long_side < r.get("min_long_side_px", 0):
            continue
        max_long_side = r.get("max_long_side_px")
        if max_long_side is not None and long_side > max_long_side:
            continue
        if elongation < r.get("min_elongation", 0):
            continue
        max_elongation = r.get("max_elongation")
        if max_elongation is not None and elongation > max_elongation:
            continue
        return class_name
    return None


def _mask_to_obb_geometry(
    mask: np.ndarray | None,
    transform: Affine,
    scale: float,
) -> Polygon | None:
    """Build an oriented bbox (Polygon) in geographic coords from a SAM mask.

    Returns ``None`` if the mask is missing, too small to define an OBB,
    or degenerates to a line/point. The renderer falls back to the
    detection's axis-aligned ``pixel_bbox`` in that case.

    The mask comes in at the rescaled image's resolution; `scale` is the
    same factor used for the bbox rescale upstream, so multiplying mask
    pixel coords by `1/scale` puts them on the original raster's grid.
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    rows, cols = np.where(arr)
    if rows.size < 3:
        return None

    # Decimate large masks — OBB is determined by extreme points so a
    # uniform stride barely affects the result and keeps cost bounded.
    if rows.size > 5000:
        stride = rows.size // 5000
        rows = rows[::stride]
        cols = cols[::stride]

    inv = 1.0 / scale if scale and scale != 0 else 1.0
    pts_px = [(float(c) * inv, float(r) * inv) for r, c in zip(rows, cols)]
    obb = MultiPoint(pts_px).minimum_rotated_rectangle
    if not isinstance(obb, Polygon):
        return None  # collinear pixels — degenerates to LineString/Point

    corners_geo = [transform * (x, y) for x, y in obb.exterior.coords]
    return Polygon(corners_geo)


# Canonical name. Users pick which classes to emit via the spec's
# `classes` field (e.g. ["road", "building"]). Override the checkpoint
# or per-class shape rules via `kwargs` without changing the registry key.
register("sam-rs", SamRsProcess.from_spec)


def _build_roads_only(spec: ProcessSpec) -> SamRsProcess:
    """Back-compat alias: same process, classes pre-pinned to ['road']."""
    pinned = ProcessSpec(
        name=spec.name,
        classes=("road",) if spec.classes is None else spec.classes,
        min_confidence=spec.min_confidence,
        kwargs=spec.kwargs,
    )
    return SamRsProcess.from_spec(pinned)


register("sam-rs-roads", _build_roads_only)


# If "vanilla SAM auto-mask + bbox" is too noisy or too coarse for roads,
# the realistic alternatives (in roughly increasing accuracy / effort):
#
# 1. Tighten heuristics here: raise `min_elongation`, raise
#    `min_long_side_px`, narrow `max_area_frac`. Cheap, no code changes.
#
# 2. SAM with point/box prompts seeded from OSM road centerlines instead
#    of the dense grid. Still vanilla SAM, much higher precision because
#    you only ask it to refine known road pixels.
#
# 3. GroundingDINO + SAM ("grounded-sam" / `LangSAM`). Text prompt
#    "road" picks bboxes; SAM masks them. The de-facto choice for
#    open-vocabulary segmentation; needs an extra model.
#
# 4. Extend `Detection` with an optional polygon field and emit the SAM
#    mask as a real polygon — bbox is the wrong primitive for roads.
#
# 5. Swap to a road-specific segmentation checkpoint (e.g. SpaceNet
#    road extraction, DeepGlobe). Single-purpose but accurate.
_ = None


__all__ = ["SamRsProcess"]
