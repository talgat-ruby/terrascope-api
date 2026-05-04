"""Canonical Detection + Raster types shared by every Process.

`Raster` is the unit of input fed into a Process: an HWC uint8 array, a
rasterio affine transform, the source CRS, and an optional WGS84 AOI used
to clip vector-data fetches without re-reading the raster.

`Detection` is the unit of output. Every Process emits a list of these,
which the orchestrator filters/concatenates/renumbers into a single result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


@dataclass
class Raster:
    """A loaded, georeferenced image tile fed into a Process."""

    data: np.ndarray              # HWC, uint8 RGB (extra bands tolerated)
    transform: Affine             # rasterio affine: pixel (col,row) → world
    crs: str                      # e.g. "EPSG:4326", "EPSG:32643"
    aoi_geom: Optional[BaseGeometry] = None  # optional WGS84 clip polygon

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


@dataclass(frozen=True)
class Detection:
    """One detected object emitted by a Process.

    Geographic and pixel bboxes are kept in parallel so downstream code can
    pick whichever frame it needs without re-projecting. `geometry` is
    optional and, when present, lives in the same CRS as `bbox` (raster
    CRS for ML processes, WGS84 for vector-data processes).
    """

    id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]            # (minx, miny, maxx, maxy) in raster CRS
    pixel_bbox: tuple[int, int, int, int]              # (col_min, row_min, col_max, row_max)
    centroid: Point
    source_model: str
    geometry: Optional[BaseGeometry] = field(default=None)


__all__ = ["Detection", "Raster"]
