"""SAM-Road process — road graph extraction via SAM-Road (htcr/sam_road).

Wraps the slim inference fork in `core2.vendor.sam_road`. Splits the raster
into the model's native square tiles, runs the trained SAM-Road network on
each tile, stitches the per-tile road graphs back together, and emits one
``Detection`` (LineString) per predicted edge.

The model itself does its own internal patch tiling per tile (the upstream
`infer_one_img` loop). We tile *on top* of that so an arbitrarily large
raster can be processed by a model that was trained on fixed-size crops.

`kwargs` knobs:
- `weights`               (str, default `congrui/sam_road`) — HF repo id (or
                          local .ckpt path) holding the SAM-Road checkpoint.
- `weights_filename`      (str, default `cityscale_vitb_512_e10.ckpt`) —
                          checkpoint filename inside the HF repo.
- `sam_weights_url`       (str, default Meta's ViT-B URL) — SAM base weights
                          to seed the image encoder. Only used when the SAM
                          weights aren't already in `weights_dir`.
- `weights_dir`           (str, default `./tmp/weights`) — local cache.
- `device`                (str, "cpu" or "cuda"). Default "cpu".
- `class_name`            (str, default "road") — class assigned to every
                          emitted edge.
- `tile_size`             (int, default 2048) — per-tile size fed into the
                          inference loop. Must be a multiple of `patch_size`.
                          Cityscale was trained on 2048×2048 tiles; changing
                          this will silently degrade results.
- `tile_overlap`          (int, default 256) — overlap between adjacent
                          inference tiles, used to merge graphs across seams.
- `infer_patches_per_edge` (int, default 16) — internal patch grid the model
                          ensembles over within each tile. Upstream uses 16
                          for the 2048 cityscale tile (heavy overlap → noise
                          suppression). Lower values are faster but lose
                          recall.
- `source_gsd_m_per_px`   (float | None) — raster's ground sample distance.
                          When None (default), the GSD is computed from the
                          raster's affine transform (handling EPSG:4326 by
                          converting degrees to meters at the raster's
                          latitude). Cityscale was trained at ~1 m/px; if
                          your raster is much higher (e.g. 0.3 m/px) the
                          model sees out-of-distribution scale and emits no
                          roads — the auto-resample fixes this.
- `model_gsd_m_per_px`    (float, default 1.0) — checkpoint's training GSD.
- `itsc_threshold`        (float | None) — override config.ITSC_THRESHOLD
                          (default 0.248). Lower → more keypoints.
- `road_threshold`        (float | None) — override config.ROAD_THRESHOLD
                          (default 0.364). Lower → more road pixels.
- `topo_threshold`        (float | None) — override config.TOPO_THRESHOLD
                          (default 0.500). Lower → more edges accepted.
- `min_edge_len_m`        (float, default 0) — drop edges shorter than this
                          in geodesic meters.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from pyproj import Geod
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform
from shapely.geometry import LineString, Point

from core.detection.types import Detection, Raster

from core2.processes.base import ProcessSpec
from core2.processes.registry import register

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())

_DEFAULT_WEIGHTS_CACHE = Path("./tmp/weights")
_DEFAULT_HF_REPO = "congrui/sam_road"
_DEFAULT_HF_FILE = "cityscale_vitb_512_e10.ckpt"
_DEFAULT_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
_DEFAULT_SAM_FILE = "sam_vit_b_01ec64.pth"

# Defaults match `config/toponet_vitb_512_cityscale.yaml` from upstream.
# Cityscale: ~1 m/px Google-Maps-style aerial, 2048×2048 training tiles.
_CITYSCALE_VITB_512: dict[str, Any] = {
    "SAM_VERSION": "vit_b",
    "PATCH_SIZE": 512,
    "TOPONET_VERSION": "normal",
    "INFER_BATCH_SIZE": 16,
    "SAMPLE_MARGIN": 64,
    # Upstream cityscale config uses 16 for a 2048×2048 tile, which yields
    # ~4× overlap per pixel. Lower values lose ensembling and silently
    # drop recall — see CVPRW 2024 paper §4.
    "INFER_PATCHES_PER_EDGE": 16,
    "ITSC_THRESHOLD": 0.248,
    "ROAD_THRESHOLD": 0.364,
    "TOPO_THRESHOLD": 0.500,
    "ITSC_NMS_RADIUS": 8,
    "ROAD_NMS_RADIUS": 16,
    "NEIGHBOR_RADIUS": 64,
    "MAX_NEIGHBOR_QUERIES": 16,
}

# Defaults match `config/toponet_vitb_256_spacenet.yaml` from upstream.
# SpaceNet: ~0.3 m/px DigitalGlobe commercial satellite, 256×256 training
# patches. Closer to typical commercial satellite imagery than Cityscale.
_SPACENET_VITB_256: dict[str, Any] = {
    "SAM_VERSION": "vit_b",
    "PATCH_SIZE": 256,
    "TOPONET_VERSION": "normal",
    "INFER_BATCH_SIZE": 64,
    "SAMPLE_MARGIN": 0,
    "INFER_PATCHES_PER_EDGE": 16,
    "ITSC_THRESHOLD": 0.195,
    "ROAD_THRESHOLD": 0.341,
    "TOPO_THRESHOLD": 0.705,
    "ITSC_NMS_RADIUS": 8,
    "ROAD_NMS_RADIUS": 16,
    "NEIGHBOR_RADIUS": 64,
    "MAX_NEIGHBOR_QUERIES": 16,
}


@dataclass
class SamRoadProcess:
    """Run SAM-Road on a raster and emit one road-segment Detection per edge."""

    spec: ProcessSpec
    name: str = "sam-road"
    weights: str = _DEFAULT_HF_REPO
    weights_filename: str = _DEFAULT_HF_FILE
    sam_weights_url: str = _DEFAULT_SAM_URL
    weights_dir: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS_CACHE)
    device: str = "cpu"
    class_name: str = "road"
    tile_size: int = 2048
    tile_overlap: int = 256
    infer_patches_per_edge: int = 16
    source_gsd_m_per_px: float | None = None
    model_gsd_m_per_px: float = 1.0
    itsc_threshold: float | None = None
    road_threshold: float | None = None
    topo_threshold: float | None = None
    min_edge_len_m: float = 0.0
    # Selected by the registry preset (e.g. `sam-road` → cityscale,
    # `sam-road-spacenet` → spacenet). Defaults to cityscale so a bare
    # SamRoadProcess(spec=...) construction in tests still works.
    _config_preset: dict[str, Any] = field(
        default_factory=lambda: dict(_CITYSCALE_VITB_512), repr=False,
    )
    _model: Any = field(default=None, init=False, repr=False)
    _config: SimpleNamespace = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "SamRoadProcess":
        kw = spec.kwargs
        return cls(
            spec=spec,
            name=spec.name,
            weights=str(kw.get("weights", _DEFAULT_HF_REPO)),
            weights_filename=str(kw.get("weights_filename", _DEFAULT_HF_FILE)),
            sam_weights_url=str(kw.get("sam_weights_url", _DEFAULT_SAM_URL)),
            weights_dir=Path(kw.get("weights_dir", _DEFAULT_WEIGHTS_CACHE)).expanduser(),
            device=str(kw.get("device", "cpu")),
            class_name=str(kw.get("class_name", "road")),
            tile_size=int(kw.get("tile_size", 2048)),
            tile_overlap=int(kw.get("tile_overlap", 256)),
            infer_patches_per_edge=int(kw.get("infer_patches_per_edge", 16)),
            source_gsd_m_per_px=_optional_float(kw.get("source_gsd_m_per_px")),
            model_gsd_m_per_px=float(kw.get("model_gsd_m_per_px", 1.0)),
            itsc_threshold=_optional_float(kw.get("itsc_threshold")),
            road_threshold=_optional_float(kw.get("road_threshold")),
            topo_threshold=_optional_float(kw.get("topo_threshold")),
            min_edge_len_m=float(kw.get("min_edge_len_m", 0.0)),
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_sam_weights(self) -> Path:
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        local = self.weights_dir / _DEFAULT_SAM_FILE
        if not local.exists():
            _log.info("sam-road: downloading SAM base weights → %s", local)
            urllib.request.urlretrieve(self.sam_weights_url, local)
        return local

    def _resolve_sam_road_ckpt(self) -> Path:
        """Return a local .ckpt path, downloading from HF if necessary."""
        if os.path.exists(self.weights):
            return Path(self.weights)
        if self.weights.count("/") != 1 or self.weights.startswith((".", "/")):
            raise FileNotFoundError(
                f"sam-road weights {self.weights!r} is not a local path nor an HF repo id"
            )
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(
            repo_id=self.weights,
            filename=self.weights_filename,
            local_dir=str(self.weights_dir / self.weights.replace("/", "__")),
        )
        return Path(local)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        from core2.vendor.sam_road.model import SAMRoad

        sam_ckpt = self._ensure_sam_weights()
        ckpt_path = self._resolve_sam_road_ckpt()
        cfg_dict = dict(self._config_preset)
        cfg_dict["INFER_PATCHES_PER_EDGE"] = self.infer_patches_per_edge
        if self.itsc_threshold is not None:
            cfg_dict["ITSC_THRESHOLD"] = self.itsc_threshold
        if self.road_threshold is not None:
            cfg_dict["ROAD_THRESHOLD"] = self.road_threshold
        if self.topo_threshold is not None:
            cfg_dict["TOPO_THRESHOLD"] = self.topo_threshold
        cfg = SimpleNamespace(**cfg_dict)
        self._config = cfg

        net = SAMRoad(cfg)
        # Seed the image encoder with SAM's pretrained weights at the right
        # resolution. Upstream does this in __init__ via SAM_CKPT_PATH; we
        # do it explicitly here so the path stays user-controllable.
        sam_state = torch.load(sam_ckpt, map_location="cpu")
        sam_state = _resize_sam_pos_embed(
            sam_state, image_size=cfg.PATCH_SIZE, vit_patch_size=16,
            global_attn_indexes=[2, 5, 8, 11],
        )
        to_load = {
            k: v for k, v in sam_state.items()
            if k in net.state_dict() and net.state_dict()[k].shape == v.shape
        }
        net.load_state_dict(to_load, strict=False)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Lightning prefixes keys depending on how training was wrapped — strip a
        # leading "net." or "model." if present so the bare nn.Module accepts them.
        cleaned = {}
        for k, v in state.items():
            if k.startswith("net."):
                cleaned[k[4:]] = v
            elif k.startswith("model."):
                cleaned[k[6:]] = v
            else:
                cleaned[k] = v
        missing, unexpected = net.load_state_dict(cleaned, strict=False)
        if missing:
            _log.warning("sam-road: %d missing keys when loading ckpt", len(missing))
        if unexpected:
            _log.warning("sam-road: %d unexpected keys when loading ckpt", len(unexpected))

        net.eval()
        net.to(self.device)
        self._model = net

    def _resolve_source_gsd(self, raster: Raster) -> float:
        """Return GSD in meters per pixel for `raster`.

        Honours an explicit `source_gsd_m_per_px` override; otherwise
        derives it from the raster transform — handling EPSG:4326 by
        converting degrees to meters at the raster's centre latitude.
        """
        if self.source_gsd_m_per_px is not None and self.source_gsd_m_per_px > 0:
            return self.source_gsd_m_per_px

        a = raster.transform
        px_x = abs(a.a)
        py_y = abs(a.e)
        crs = raster.crs.upper()
        if crs in ("EPSG:4326", "OGC:CRS84"):
            # transform.f is the top latitude; centre lat = top + height * e/2
            centre_lat = a.f + a.e * (raster.height / 2.0)
            metres_per_deg_lon = 111_320.0 * math.cos(math.radians(centre_lat))
            metres_per_deg_lat = 110_540.0
            return (px_x * metres_per_deg_lon + py_y * metres_per_deg_lat) / 2.0
        # Projected CRS (UTM/Web Mercator etc.) — transform is already in metres.
        return (px_x + py_y) / 2.0

    def _scale(self, raster: Raster | None = None) -> float:
        """Resample factor mapping source GSD onto model GSD.

        With no raster, falls back to the explicit `source_gsd_m_per_px`
        only — used in tests where we don't want to construct a raster.
        """
        if raster is None:
            if self.source_gsd_m_per_px is None or self.source_gsd_m_per_px <= 0:
                return 1.0
            if self.model_gsd_m_per_px <= 0:
                return 1.0
            return self.source_gsd_m_per_px / self.model_gsd_m_per_px

        gsd = self._resolve_source_gsd(raster)
        if gsd <= 0 or self.model_gsd_m_per_px <= 0:
            return 1.0
        return gsd / self.model_gsd_m_per_px

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, raster: Raster) -> list[Detection]:
        self._load()
        cfg = self._config
        img = raster.data
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(
                f"sam-road expects HWC RGB raster, got shape {img.shape!r}"
            )
        img = img[:, :, :3]

        # Resample to the model's training GSD (~1 m/px for cityscale).
        # Without this, high-res rasters look like a microscope image to a
        # network trained at street-level scale → silent zero-detection.
        source_gsd = self._resolve_source_gsd(raster)
        scale = self._scale(raster)
        if scale != 1.0:
            img = _resize(img, scale)
            _log.info(
                "sam-road: resampled raster %.3fx to match training GSD "
                "(%.3f → %.3f m/px), now %dx%d",
                scale, source_gsd, self.model_gsd_m_per_px,
                img.shape[1], img.shape[0],
            )
        else:
            _log.info(
                "sam-road: GSD %.3f m/px ≈ training scale, no resample",
                source_gsd,
            )
        H, W = img.shape[:2]

        # Tile the raster into squares the inference loop can handle.
        tile = self.tile_size
        overlap = self.tile_overlap
        if tile % cfg.PATCH_SIZE != 0:
            raise ValueError(
                f"tile_size ({tile}) must be a multiple of model patch size ({cfg.PATCH_SIZE})"
            )
        # If the (possibly resampled) raster is smaller than one configured
        # tile, shrink the tile down to the smallest multiple of PATCH_SIZE
        # that still wraps it with SAMPLE_MARGIN on all sides. Avoids feeding
        # the model 90% black-padding when the source is intrinsically small.
        long_side = max(H, W)
        min_tile = cfg.PATCH_SIZE + 2 * cfg.SAMPLE_MARGIN
        needed = max(min_tile, long_side + 2 * cfg.SAMPLE_MARGIN)
        # Round up to next multiple of PATCH_SIZE.
        fitted = ((needed + cfg.PATCH_SIZE - 1) // cfg.PATCH_SIZE) * cfg.PATCH_SIZE
        if fitted < tile:
            _log.info(
                "sam-road: shrinking tile_size %d → %d to fit %dx%d raster",
                tile, fitted, W, H,
            )
            tile = fitted
            overlap = min(overlap, tile // 4)
        # Re-derive the patch-grid for the chosen tile size so SAMPLE_MARGIN
        # coverage is preserved. Upstream tunes patches_per_edge for tile=2048;
        # smaller tiles need fewer patches to maintain the same overlap density.
        cfg.INFER_PATCHES_PER_EDGE = max(
            2, ((tile - cfg.PATCH_SIZE - 2 * cfg.SAMPLE_MARGIN) // (cfg.PATCH_SIZE // 4)) + 1,
        )

        all_nodes: list[np.ndarray] = []  # (r, c) in raster coords
        all_edges: list[tuple[int, int]] = []
        node_offset = 0

        for y0 in _strides(H, tile, overlap):
            for x0 in _strides(W, tile, overlap):
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)
                ph, pw = y1 - y0, x1 - x0
                crop = img[y0:y1, x0:x1, :3]
                if ph == tile and pw == tile:
                    tile_img = crop
                else:
                    # Edge-replicate so the encoder doesn't see a hard black
                    # seam where roads are absent only by padding.
                    tile_img = np.pad(
                        crop,
                        ((0, tile - ph), (0, tile - pw), (0, 0)),
                        mode="edge",
                    )

                nodes_rc, edges = _infer_tile(self._model, tile_img, cfg, self.device)
                if nodes_rc.shape[0] == 0:
                    continue
                # Drop nodes that landed in the padding region.
                in_bounds = (nodes_rc[:, 0] < ph) & (nodes_rc[:, 1] < pw)
                if not in_bounds.all():
                    keep_idx = np.flatnonzero(in_bounds)
                    remap = -np.ones(nodes_rc.shape[0], dtype=np.int64)
                    remap[keep_idx] = np.arange(keep_idx.size)
                    nodes_rc = nodes_rc[keep_idx]
                    edges = [
                        (int(remap[a]), int(remap[b]))
                        for a, b in edges
                        if remap[a] >= 0 and remap[b] >= 0
                    ]
                # Shift node coords into the raster frame.
                nodes_rc = nodes_rc.astype(np.float64) + np.array([y0, x0])
                all_nodes.append(nodes_rc)
                for a, b in edges:
                    all_edges.append((node_offset + int(a), node_offset + int(b)))
                node_offset += nodes_rc.shape[0]

        if not all_nodes:
            _log.info("sam-road: no road graph found")
            return []

        nodes_rc = np.concatenate(all_nodes, axis=0)
        # Stitch graphs across tile seams: collapse only near-duplicate nodes
        # (within ROAD_NMS_RADIUS, the same scale upstream uses for
        # within-tile point dedup). A larger radius merges genuine adjacent
        # graph nodes — particularly damaging on small rasters where most
        # nodes live within a single tile and there are no real seams.
        merge_radius = float(cfg.ROAD_NMS_RADIUS) if len(all_nodes) > 1 else 0.0
        nodes_rc, edges = _merge_close_nodes(nodes_rc, all_edges, radius=merge_radius)

        # Project node coords from the (possibly resampled) inference frame
        # back into the raster's native pixel grid before georeferencing.
        if scale != 1.0:
            nodes_rc = nodes_rc / scale

        _log.info(
            "sam-road: %d nodes, %d edges after stitching",
            nodes_rc.shape[0], len(edges),
        )
        return self._edges_to_detections(nodes_rc, edges, raster)

    def _edges_to_detections(
        self,
        nodes_rc: np.ndarray,
        edges: list[tuple[int, int]],
        raster: Raster,
    ) -> list[Detection]:
        if not edges:
            return []
        # Project all nodes to raster CRS in one shot.
        rows = nodes_rc[:, 0]
        cols = nodes_rc[:, 1]
        xs, ys = xy(raster.transform, rows.tolist(), cols.tolist())
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        # For min_edge_len_m, geodesic measurements need WGS84 coords.
        if self.min_edge_len_m > 0 and raster.crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
            lons, lats = warp_transform(raster.crs, "EPSG:4326", xs.tolist(), ys.tolist())
            geod_lons = np.asarray(lons)
            geod_lats = np.asarray(lats)
        else:
            geod_lons = xs
            geod_lats = ys
        geod = Geod(ellps="WGS84")

        allow = set(self.spec.classes) if self.spec.classes else None
        if allow is not None and self.class_name not in allow:
            return []

        detections: list[Detection] = []
        seen: set[tuple[int, int]] = set()
        for a, b in edges:
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            x0, y0 = float(xs[a]), float(ys[a])
            x1, y1 = float(xs[b]), float(ys[b])
            line = LineString([(x0, y0), (x1, y1)])
            if line.length == 0:
                continue
            if self.min_edge_len_m > 0:
                _, _, dist = geod.inv(
                    geod_lons[a], geod_lats[a], geod_lons[b], geod_lats[b]
                )
                if dist < self.min_edge_len_m:
                    continue

            r0, c0 = int(nodes_rc[a, 0]), int(nodes_rc[a, 1])
            r1, c1 = int(nodes_rc[b, 0]), int(nodes_rc[b, 1])
            pixel_bbox = (
                min(c0, c1), min(r0, r1), max(c0, c1), max(r0, r1),
            )
            geo_bbox = (
                min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
            )
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            detections.append(
                Detection(
                    id=len(detections),
                    class_name=self.class_name,
                    confidence=1.0,
                    bbox=geo_bbox,
                    pixel_bbox=pixel_bbox,
                    centroid=Point(cx, cy),
                    source_model=self.name,
                    geometry=line,
                )
            )
        _log.info("sam-road: emitted %d edge detections", len(detections))
        return detections


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Downsample (or upsample) an HWC uint8 image with PIL/LANCZOS."""
    from PIL import Image

    h, w = image.shape[:2]
    new = (max(1, int(w * scale)), max(1, int(h * scale)))
    return np.asarray(Image.fromarray(image).resize(new, Image.LANCZOS))


def _strides(total: int, tile: int, overlap: int) -> list[int]:
    """Return tile start positions covering [0, total] with `overlap` overlap."""
    if total <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, total - tile + 1, step))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def _resize_sam_pos_embed(
    state_dict: dict, image_size: int, vit_patch_size: int, global_attn_indexes: list[int]
) -> dict:
    """Resize SAM's positional embeddings to match a non-default PATCH_SIZE.

    Lifted from upstream `SAMRoad.resize_sam_pos_embed`. The official SAM
    weights are trained at 1024×1024; we run at 512×512 (or 256×256).
    """
    import torch.nn.functional as F

    new_state = {k: v for k, v in state_dict.items()}
    pos_key = "image_encoder.pos_embed"
    if pos_key not in new_state:
        return new_state
    pos_embed = new_state[pos_key]
    token_size = image_size // vit_patch_size
    if pos_embed.shape[1] == token_size:
        return new_state
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, (token_size, token_size), mode="bilinear", align_corners=False
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1)
    new_state[pos_key] = pos_embed

    rel_pos_keys = [k for k in state_dict.keys() if "rel_pos" in k]
    global_rel_keys = [
        k for k in rel_pos_keys
        if any(str(i) in k for i in global_attn_indexes)
    ]
    for k in global_rel_keys:
        rel = new_state[k]
        h, w = rel.shape
        rel = rel.unsqueeze(0).unsqueeze(0)
        rel = F.interpolate(rel, (token_size * 2 - 1, w), mode="bilinear", align_corners=False)
        new_state[k] = rel[0, 0, ...]
    return new_state


def _infer_tile(
    net, tile_img: np.ndarray, config, device: str
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Run inference on one square tile, return (nodes_rc, edge_list).

    Reimplements upstream `inferencer.infer_one_img` against the slim model.
    Nodes come back in (row, col) raster coords relative to the tile.
    """
    import torch
    import scipy.spatial
    import rtree
    from collections import defaultdict

    from core2.vendor.sam_road.infer_helpers import (
        extract_graph_points,
        get_patch_info_one_img,
    )

    image_size = tile_img.shape[0]
    batch_size = config.INFER_BATCH_SIZE
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE,
        config.INFER_PATCHES_PER_EDGE,
    )
    patch_num = len(all_patch_info)
    batch_num = math.ceil(patch_num / batch_size)

    fused_kp = torch.zeros(image_size, image_size, dtype=torch.float32, device=device)
    fused_road = torch.zeros(image_size, image_size, dtype=torch.float32, device=device)
    pixel_counter = torch.zeros(image_size, image_size, dtype=torch.float32, device=device)
    img_features: list[torch.Tensor] = []

    for bi in range(batch_num):
        offset = bi * batch_size
        batch = all_patch_info[offset : offset + batch_size]
        patches = torch.stack(
            [
                torch.tensor(tile_img[y0:y1, x0:x1, :], dtype=torch.float32)
                for _, (x0, y0), (x1, y1) in batch
            ],
            dim=0,
        ).to(device)
        mask_scores, feats = net.infer_masks_and_img_features(patches)
        img_features.append(feats)
        for pi, (_, (x0, y0), (x1, y1)) in enumerate(batch):
            fused_kp[y0:y1, x0:x1] += mask_scores[pi, :, :, 0]
            fused_road[y0:y1, x0:x1] += mask_scores[pi, :, :, 1]
            pixel_counter[y0:y1, x0:x1] += 1.0

    pixel_counter = torch.clamp(pixel_counter, min=1.0)
    fused_kp = (fused_kp / pixel_counter * 255).to(torch.uint8).cpu().numpy()
    fused_road = (fused_road / pixel_counter * 255).to(torch.uint8).cpu().numpy()

    # Diagnostic: surface the raw mask stats so a "no detections" run can
    # be told apart from a "model is firing but threshold is too tight" run.
    _log.debug(
        "sam-road tile: keypoint mask max=%d mean=%.2f, "
        "road mask max=%d mean=%.2f (ITSC_THR=%.0f ROAD_THR=%.0f, both /255)",
        int(fused_kp.max()), float(fused_kp.mean()),
        int(fused_road.max()), float(fused_road.mean()),
        config.ITSC_THRESHOLD * 255, config.ROAD_THRESHOLD * 255,
    )

    graph_points = extract_graph_points(fused_kp, fused_road, config)  # (N, 2) xy
    if graph_points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), []

    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = int(v[0]), int(v[1])
        graph_rtree.insert(i, (x, y, x, y))

    edge_scores: dict[tuple[int, int], float] = defaultdict(float)
    edge_counts: dict[tuple[int, int], float] = defaultdict(float)

    for bi in range(batch_num):
        offset = bi * batch_size
        batch = all_patch_info[offset : offset + batch_size]

        topo_data: dict[str, list[np.ndarray]] = {"points": [], "pairs": [], "valid": []}
        idx_maps: list[dict[int, int]] = []

        for _, (x0, y0), (x1, y1) in batch:
            patch_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {i: a for i, a in enumerate(patch_indices)}
            n_pts = len(patch_indices)
            if n_pts == 0:
                topo_data["points"].append(np.zeros((0, 2), dtype=np.float32))
                topo_data["pairs"].append(
                    np.zeros((0, config.MAX_NEIGHBOR_QUERIES, 2), dtype=np.int64)
                )
                topo_data["valid"].append(
                    np.zeros((0, config.MAX_NEIGHBOR_QUERIES), dtype=bool)
                )
                idx_maps.append(idx_patch2all)
                continue
            patch_points = graph_points[patch_indices, :] - np.array(
                [[x0, y0]], dtype=graph_points.dtype
            )
            kdtree = scipy.spatial.KDTree(patch_points)
            _, knn_idx = kdtree.query(
                patch_points,
                k=config.MAX_NEIGHBOR_QUERIES + 1,
                distance_upper_bound=config.NEIGHBOR_RADIUS,
            )
            knn_idx = knn_idx[:, 1:]
            src_idx = np.tile(
                np.arange(n_pts)[:, np.newaxis], (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < n_pts
            tgt_idx = np.where(valid, knn_idx, src_idx)
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data["points"].append(patch_points)
            topo_data["pairs"].append(pairs)
            topo_data["valid"].append(valid)
            idx_maps.append(idx_patch2all)

        max_pts = max((p.shape[0] for p in topo_data["points"]), default=0)
        if max_pts == 0:
            continue
        collated: dict[str, np.ndarray] = {}
        for key, x_list in topo_data.items():
            length = max_pts
            collated[key] = np.stack(
                [
                    np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
                    for x in x_list
                ],
                axis=0,
            )

        batch_features = img_features[bi]
        batch_points = torch.tensor(collated["points"], device=device, dtype=torch.float32)
        batch_pairs = torch.tensor(collated["pairs"], device=device, dtype=torch.long)
        batch_valid = torch.tensor(collated["valid"], device=device, dtype=torch.bool)

        topo_scores = net.infer_toponet(
            batch_features, batch_points, batch_pairs, batch_valid,
        )
        topo_scores = torch.where(
            torch.isnan(topo_scores), torch.full_like(topo_scores, -100.0), topo_scores
        ).squeeze(-1).cpu().numpy()

        b_size, n_samples, n_pairs = topo_scores.shape
        for bb in range(b_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated["valid"][bb, si, pi]:
                        continue
                    src_p, tgt_p = collated["pairs"][bb, si, pi, :]
                    src_a = idx_maps[bb][int(src_p)]
                    tgt_a = idx_maps[bb][int(tgt_p)]
                    score = float(topo_scores[bb, si, pi])
                    edge_scores[(src_a, tgt_a)] += score
                    edge_counts[(src_a, tgt_a)] += 1.0

    edges: list[tuple[int, int]] = []
    for edge, total in edge_scores.items():
        avg = total / edge_counts[edge]
        if avg > config.TOPO_THRESHOLD:
            edges.append(edge)

    # Convert (x, y) → (r, c) for the caller.
    nodes_rc = graph_points[:, ::-1].astype(np.int64)
    return nodes_rc, edges


def _merge_close_nodes(
    nodes_rc: np.ndarray,
    edges: list[tuple[int, int]],
    radius: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Collapse nodes within `radius` pixels into single representatives.

    Used to stitch road graphs across overlapping inference tiles. Returns
    deduped (nodes, edges) with edges remapped onto the survivor indices.
    """
    if nodes_rc.shape[0] == 0 or radius <= 0:
        return nodes_rc, edges
    import scipy.spatial

    tree = scipy.spatial.KDTree(nodes_rc)
    parent = np.arange(nodes_rc.shape[0])

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    pairs = tree.query_pairs(r=radius)
    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(nodes_rc.shape[0])])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    new_nodes = np.zeros((unique_roots.size, 2), dtype=nodes_rc.dtype)
    counts = np.zeros(unique_roots.size, dtype=np.int64)
    for i, root_idx in enumerate(inverse):
        new_nodes[root_idx] += nodes_rc[i]
        counts[root_idx] += 1
    new_nodes = (new_nodes.T // counts).T

    new_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b in edges:
        na, nb = int(inverse[a]), int(inverse[b])
        if na == nb:
            continue
        key = (na, nb) if na <= nb else (nb, na)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append((na, nb))
    return new_nodes, new_edges


# Preset registrations. Each preset hardwires the dataset-specific config
# dict (PATCH_SIZE, thresholds, etc.) and a default checkpoint filename, but
# every knob remains overridable via `ProcessSpec.kwargs` — same closure
# capture pattern as `_register_yolo_presets()` in yolo_sahi.py.
_PRESETS: dict[str, dict[str, Any]] = {
    "sam-road": {
        "config": _CITYSCALE_VITB_512,
        "weights_filename": "cityscale_vitb_512_e10.ckpt",
        "tile_size": 2048,
        "model_gsd_m_per_px": 1.0,
    },
    "sam-road-spacenet": {
        "config": _SPACENET_VITB_256,
        "weights_filename": "spacenet_vitb_256_e10.ckpt",
        # 4 × PATCH_SIZE, mirroring the cityscale 4× ratio (2048 / 512).
        "tile_size": 1024,
        # SpaceNet was trained on ~0.3 m/px DigitalGlobe imagery.
        "model_gsd_m_per_px": 0.3,
    },
}


def _register_sam_road_presets() -> None:
    for key, defaults in _PRESETS.items():
        def builder(spec: ProcessSpec, _d: dict[str, Any] = defaults) -> SamRoadProcess:
            kw = dict(spec.kwargs)
            kw.setdefault("weights_filename", _d["weights_filename"])
            kw.setdefault("tile_size", _d["tile_size"])
            kw.setdefault("model_gsd_m_per_px", _d["model_gsd_m_per_px"])
            patched = ProcessSpec(
                name=spec.name,
                classes=spec.classes,
                min_confidence=spec.min_confidence,
                kwargs=kw,
            )
            proc = SamRoadProcess.from_spec(patched)
            proc._config_preset = dict(_d["config"])
            return proc

        register(key, builder)


_register_sam_road_presets()


__all__ = ["SamRoadProcess"]
