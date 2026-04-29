"""Domain types for the detection pipeline.

Detector implementations consume a `Raster` and emit `list[Detection]`. Every
other layer (filter, renderer, exporter, persistence) operates only on these
types — no model-specific structures leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


@dataclass
class Raster:
    """Pixel data + georeferencing for a single image to detect on.

    `data` is HWC uint8 (RGB display-ready). Detectors consume this directly;
    the renderer also draws onto it. CRS is the WKT/EPSG string of the image.
    `aoi_geom` is the optional clipping polygon, in the same CRS.
    """

    data: NDArray[np.uint8]
    transform: Affine
    crs: str
    aoi_geom: BaseGeometry | None = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


@dataclass
class Detection:
    """A single bbox detection.

    `bbox` is geographic (minx, miny, maxx, maxy) in the raster's CRS — for
    EPSG:4326 that's (lon_min, lat_min, lon_max, lat_max).
    `pixel_bbox` is (col_min, row_min, col_max, row_max) in the raster's pixel
    grid — kept so the renderer can draw without re-projecting.
    `id` is assigned sequentially after filtering (0..N).
    """

    id: int
    class_name: str
    confidence: float  # [0.0, 1.0]
    bbox: tuple[float, float, float, float]
    pixel_bbox: tuple[int, int, int, int]
    centroid: Point


class Detector(Protocol):
    """Pluggable detector contract.

    Implementations are looked up by `name` via `build_detector`. Each call
    to `detect` is independent; implementations are free to cache models on
    instance state.
    """

    name: str

    def detect(self, raster: Raster) -> list[Detection]: ...
