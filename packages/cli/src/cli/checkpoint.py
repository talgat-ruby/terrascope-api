"""Checkpoint/resume system for the local CLI processing pipeline."""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from core.config import Settings
from core.models.tile import Tile
from core.services.detector import RawDetection

# Step IDs
STEP_LOAD = "step_01_imagery"
STEP_TILE = "step_02_tiles"
STEP_DETECT = "step_03_detections"
STEP_PP_NMS = "step_04a_nms"
STEP_PP_CONFIDENCE = "step_04b_confidence"
STEP_PP_SIZE = "step_04c_size"
STEP_PP_SHAPE = "step_04d_shape"
STEP_PP_SIMPLIFY = "step_04e_simplify"
STEP_PP_CLIP = "step_04f_clip"
STEP_EXPORT = "step_05_export"
STEP_INDICATORS = "step_06_indicators"


def compute_fingerprint(
    input_path: Path, aoi_path: Path | None, settings: Settings
) -> str:
    """Compute a SHA-256 fingerprint of all inputs that affect processing."""
    h = hashlib.sha256()
    resolved = input_path.resolve()
    h.update(str(resolved).encode())
    stat = resolved.stat()
    h.update(f"{stat.st_mtime}:{stat.st_size}".encode())

    if aoi_path is not None:
        h.update(aoi_path.read_bytes())
    else:
        h.update(b"no-aoi")

    params = (
        f"{settings.tile_size}:{settings.tile_overlap}:"
        f"{settings.confidence_threshold}:{settings.min_area_m2}:"
        f"{settings.max_area_m2}:{settings.nms_iou_threshold}:"
        f"{settings.simplify_tolerance_m}:{settings.device}"
    )
    h.update(params.encode())
    return h.hexdigest()


class CheckpointManager:
    """Manages file-based checkpoints for pipeline steps."""

    def __init__(self, output_dir: Path, fingerprint: str) -> None:
        self._dir = output_dir / ".checkpoints"
        self._manifest_path = self._dir / "manifest.json"
        self._fingerprint = fingerprint

        self._dir.mkdir(parents=True, exist_ok=True)

        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text())
            if manifest.get("fingerprint") != fingerprint:
                self.clear()
                self._dir.mkdir(parents=True, exist_ok=True)
                self._manifest = self._new_manifest()
            else:
                self._manifest = manifest
        else:
            self._manifest = self._new_manifest()

    def _new_manifest(self) -> dict:
        return {
            "version": 1,
            "fingerprint": self._fingerprint,
            "completed_steps": [],
            "timestamps": {},
        }

    def _save_manifest(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._manifest, f, indent=2)
            os.replace(tmp, self._manifest_path)
        except BaseException:
            os.unlink(tmp)
            raise

    def is_step_complete(self, step_id: str) -> bool:
        if step_id not in self._manifest["completed_steps"]:
            return False
        # Verify checkpoint file(s) actually exist
        return self._checkpoint_files_exist(step_id)

    def _checkpoint_files_exist(self, step_id: str) -> bool:
        if step_id == STEP_LOAD:
            return (self._dir / f"{step_id}.npz").exists() and (
                self._dir / f"{step_id}_meta.json"
            ).exists()
        if step_id == STEP_TILE:
            return (self._dir / f"{step_id}.npz").exists() and (
                self._dir / f"{step_id}_meta.json"
            ).exists()
        # Detection steps use geojson
        if step_id in (
            STEP_DETECT,
            STEP_PP_NMS,
            STEP_PP_CONFIDENCE,
            STEP_PP_SIZE,
            STEP_PP_SHAPE,
            STEP_PP_SIMPLIFY,
            STEP_PP_CLIP,
        ):
            return (self._dir / f"{step_id}.geojson").exists()
        # Export and indicators: just check manifest
        return True

    def mark_complete(self, step_id: str) -> None:
        if step_id not in self._manifest["completed_steps"]:
            self._manifest["completed_steps"].append(step_id)
        self._manifest["timestamps"][step_id] = datetime.now(timezone.utc).isoformat()
        self._save_manifest()

    def clear(self) -> None:
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._manifest = self._new_manifest()

    # -- Imagery (step 1) --

    def save_imagery(
        self,
        data: NDArray[np.float32],
        transform: Affine,
        crs: str,
        aoi: BaseGeometry,
    ) -> None:
        np.savez_compressed(self._dir / f"{STEP_LOAD}.npz", data=data)
        meta = {
            "transform": list(transform)[:6],
            "crs": crs,
            "aoi_wkt": aoi.wkt,
        }
        (self._dir / f"{STEP_LOAD}_meta.json").write_text(json.dumps(meta))

    def load_imagery(
        self,
    ) -> tuple[NDArray[np.float32], Affine, str, BaseGeometry]:
        npz = np.load(self._dir / f"{STEP_LOAD}.npz")
        data = npz["data"]
        meta = json.loads((self._dir / f"{STEP_LOAD}_meta.json").read_text())
        transform = Affine(*meta["transform"])
        crs = meta["crs"]
        aoi = wkt.loads(meta["aoi_wkt"])
        return data, transform, crs, aoi

    # -- Tiles (step 2) --

    def save_tiles(self, tiles: list[Tile]) -> None:
        arrays: dict[str, np.ndarray] = {}
        tile_meta: list[dict] = []
        for i, tile in enumerate(tiles):
            arrays[f"tile_{i}_data"] = tile.data
            arrays[f"tile_{i}_mask"] = tile.valid_mask
            tile_meta.append(
                {
                    "index": list(tile.index),
                    "pixel_window": list(tile.pixel_window),
                    "transform": list(tile.transform)[:6],
                    "crs": tile.crs,
                }
            )
        np.savez_compressed(self._dir / f"{STEP_TILE}.npz", **arrays)
        (self._dir / f"{STEP_TILE}_meta.json").write_text(json.dumps(tile_meta))

    def load_tiles(self) -> list[Tile]:
        npz = np.load(self._dir / f"{STEP_TILE}.npz")
        tile_meta = json.loads((self._dir / f"{STEP_TILE}_meta.json").read_text())
        tiles: list[Tile] = []
        for i, m in enumerate(tile_meta):
            tiles.append(
                Tile(
                    index=tuple(m["index"]),  # type: ignore[arg-type]
                    pixel_window=tuple(m["pixel_window"]),  # type: ignore[arg-type]
                    transform=Affine(*m["transform"]),
                    data=npz[f"tile_{i}_data"],
                    valid_mask=npz[f"tile_{i}_mask"],
                    crs=m["crs"],
                )
            )
        return tiles

    # -- Detections (step 3 + step 4 substeps) --

    def save_detections(
        self, step_id: str, detections: list[RawDetection], crs: str
    ) -> None:
        if not detections:
            # Save empty GeoJSON
            gdf = gpd.GeoDataFrame(
                columns=["class_name", "confidence", "source", "geometry"],
                geometry="geometry",
                crs=crs,
            )
        else:
            records = [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "source": d.source,
                    "geometry": d.geometry,
                }
                for d in detections
            ]
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf.to_file(self._dir / f"{step_id}.geojson", driver="GeoJSON")

    def load_detections(self, step_id: str) -> list[RawDetection]:
        path = self._dir / f"{step_id}.geojson"
        gdf = gpd.read_file(path)
        detections: list[RawDetection] = []
        for _, row in gdf.iterrows():
            detections.append(
                RawDetection(
                    class_name=row["class_name"],
                    confidence=int(row["confidence"]),
                    geometry=row.geometry,
                    source=row["source"],
                )
            )
        return detections
