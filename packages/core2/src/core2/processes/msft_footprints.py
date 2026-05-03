"""MSFT Global Building Footprints process.

Vector-only — no model inference. Given a Raster (already loaded + clipped
in WGS84), determine the Bing-style quadkeys covering the AOI, fetch each
shard from Microsoft's CDN (cached on disk), parse the JSON-Lines payload,
clip to the AOI, and emit one `Detection` per surviving footprint.

This is a port of the standalone `run_buildings.py` script with the bugs
fixed (canonical bounds tuple ordering, real CSV/JSON parsing instead of
string-replace tricks) and the matplotlib visualisation removed.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyproj import Geod
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection, Raster

from core2.processes.base import ProcessSpec
from core2.processes.registry import register

_DEFAULT_INDEX_URL = (
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
)
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "terrascope" / "msft_buildings"
_GEOD = Geod(ellps="WGS84")


@dataclass
class MsftFootprintProcess:
    """Pull pre-extracted building footprints from Microsoft's open dataset.

    `kwargs` knobs:
    - `country`     (str, default "Kazakhstan") — restrict the index lookup.
    - `min_area_m2` (float, default 20.0) — drop shards under this area.
    - `cache_dir`   (str | Path, default `~/.cache/terrascope/msft_buildings`).
    - `index_url`   (str) — override the dataset-links.csv URL.
    """

    spec: ProcessSpec
    name: str = "msft-buildings"
    country: str = "Kazakhstan"
    min_area_m2: float = 20.0
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    index_url: str = _DEFAULT_INDEX_URL

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "MsftFootprintProcess":
        kw = spec.kwargs
        return cls(
            spec=spec,
            country=str(kw.get("country", "Kazakhstan")),
            min_area_m2=float(kw.get("min_area_m2", 20.0)),
            cache_dir=Path(kw.get("cache_dir", _DEFAULT_CACHE_DIR)).expanduser(),
            index_url=str(kw.get("index_url", _DEFAULT_INDEX_URL)),
        )

    def run(self, raster: Raster) -> list[Detection]:
        roi_wgs84 = _raster_roi_wgs84(raster)
        roi_geom: BaseGeometry = box(*roi_wgs84)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.cache_dir / "dataset-links.csv"
        if not index_path.exists():
            urllib.request.urlretrieve(self.index_url, index_path)

        shards = _shards_overlapping(index_path, self.country, roi_geom)
        detections: list[Detection] = []
        idx = 0

        for qk, url in shards.items():
            shard_path = self.cache_dir / f"{qk}.csv.gz"
            if not shard_path.exists():
                urllib.request.urlretrieve(url, shard_path)

            for poly in _iter_polygons(shard_path, roi_geom):
                clipped = poly.intersection(roi_geom)
                if clipped.is_empty:
                    continue
                area_m2 = abs(_GEOD.geometry_area_perimeter(poly)[0])
                if area_m2 < self.min_area_m2:
                    continue

                bbox = clipped.bounds
                pixel_bbox = _bbox_to_pixels(bbox, raster)
                detections.append(
                    Detection(
                        id=idx,
                        class_name="building",
                        confidence=1.0,
                        bbox=bbox,
                        pixel_bbox=pixel_bbox,
                        centroid=Point(clipped.centroid.x, clipped.centroid.y),
                        source_model=self.name,
                        geometry=clipped,
                    )
                )
                idx += 1

        return detections


def _raster_roi_wgs84(raster: Raster) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for the raster footprint.

    Honours `aoi_geom` when present; otherwise computes from the transform
    + width/height of the loaded raster.
    """
    if raster.aoi_geom is not None and raster.crs.upper() in ("EPSG:4326", "OGC:CRS84"):
        minx, miny, maxx, maxy = raster.aoi_geom.bounds
        return (minx, miny, maxx, maxy)

    a = raster.transform
    left, top = a.c, a.f
    right = a.c + a.a * raster.width
    bottom = a.f + a.e * raster.height
    minx, maxx = sorted((left, right))
    miny, maxy = sorted((bottom, top))

    if raster.crs.upper() in ("EPSG:4326", "OGC:CRS84"):
        return (minx, miny, maxx, maxy)
    return tuple(transform_bounds(raster.crs, "EPSG:4326", minx, miny, maxx, maxy))  # type: ignore[return-value]


def _shards_overlapping(
    index_path: Path, country: str, roi: BaseGeometry
) -> dict[str, str]:
    """Parse dataset-links.csv and return {quadkey: url} for ROI-overlapping shards."""
    needed: dict[str, str] = {}
    with open(index_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col_country = 0
        col_qk = 1
        col_url = 2
        if header is not None and header and header[0].lower().startswith("location"):
            # Real header — locate columns by name where possible.
            try:
                col_country = header.index("Location")
                col_qk = header.index("QuadKey")
                col_url = header.index("Url")
            except ValueError:
                pass
        for row in reader:
            if len(row) <= max(col_country, col_qk, col_url):
                continue
            if row[col_country] != country:
                continue
            qk = row[col_qk]
            try:
                if box(*_qk_bounds(qk)).intersects(roi):
                    needed[qk] = row[col_url]
            except ValueError:
                continue
    return needed


def _iter_polygons(shard_path: Path, roi: BaseGeometry):
    """Yield shapely polygons from an MSFT footprint shard, ROI-prefiltered.

    The MSFT shards are gzipped JSON-Lines (one GeoJSON Feature per line).
    Some older mirrors are CSV with a `geometry` column whose value is a
    JSON object — handle both.
    """
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        line = first.strip()

        as_geojson = _try_parse_feature(line)
        if as_geojson is not None:
            geom = shape(as_geojson["geometry"])
            if geom.intersects(roi):
                yield geom
            for line in f:
                feat = _try_parse_feature(line.strip())
                if feat is None:
                    continue
                geom = shape(feat["geometry"])
                if geom.intersects(roi):
                    yield geom
            return

        # Fallback: CSV with a geometry column of JSON.
        f.seek(0)
        reader = csv.DictReader(f)
        if reader.fieldnames and "geometry" in reader.fieldnames:
            for row in reader:
                raw = row.get("geometry")
                if not raw:
                    continue
                try:
                    geom = shape(json.loads(raw))
                except (json.JSONDecodeError, ValueError):
                    continue
                if geom.intersects(roi):
                    yield geom


def _try_parse_feature(line: str) -> dict[str, Any] | None:
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "geometry" not in obj:
        return None
    return obj


def _qk_bounds(qk: str) -> tuple[float, float, float, float]:
    x = y = 0
    zoom = len(qk)
    for i, c in enumerate(qk):
        mask = 1 << (zoom - 1 - i)
        if c in ("1", "3"):
            x |= mask
        if c in ("2", "3"):
            y |= mask
        if c not in ("0", "1", "2", "3"):
            raise ValueError(f"invalid quadkey char {c!r}")
    n = 2**zoom
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max


def _bbox_to_pixels(
    bbox: tuple[float, float, float, float], raster: Raster
) -> tuple[int, int, int, int]:
    """Convert a geographic bbox to pixel (col_min, row_min, col_max, row_max).

    Handles north-up rasters; for non-WGS84 rasters the geographic bbox is
    first projected back to the raster CRS.
    """
    minx, miny, maxx, maxy = bbox
    if raster.crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        minx, miny, maxx, maxy = transform_bounds(
            "EPSG:4326", raster.crs, minx, miny, maxx, maxy
        )
    r1, c1 = rowcol(raster.transform, minx, maxy)
    r2, c2 = rowcol(raster.transform, maxx, miny)
    row_min, row_max = sorted((int(r1), int(r2)))
    col_min, col_max = sorted((int(c1), int(c2)))
    return (col_min, row_min, col_max, row_max)


register("msft-buildings", MsftFootprintProcess.from_spec)


__all__ = ["MsftFootprintProcess"]
