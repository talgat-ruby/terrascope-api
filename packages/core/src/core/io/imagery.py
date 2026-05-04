"""Load a GeoTIFF (optionally clipped to an AOI) into a `Raster`.

Reads up to three bands (RGB) into HWC uint8. Float rasters get scaled to
[0, 255] using the per-band 2nd/98th percentiles, which gives a sane visual
contrast on raw satellite imagery without requiring user knobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from core.detection.types import Raster


def load_raster(path: str | Path, aoi: BaseGeometry | None = None) -> Raster:
    """Load a GeoTIFF as an HWC uint8 `Raster`, optionally clipped to AOI.

    AOI is interpreted as WGS84 and reprojected to the raster CRS before
    masking. When AOI is None, the full raster footprint is returned.
    """
    with rasterio.open(str(path)) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        if aoi is not None:
            geom = aoi
            if crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
                geom = transform_geom("EPSG:4326", crs, mapping(aoi))
            else:
                geom = mapping(aoi)
            arr, transform = rio_mask(src, [geom], crop=True, filled=True)
        else:
            arr = src.read()
            transform = src.transform

        # Take up to three bands as RGB.
        if arr.shape[0] >= 3:
            rgb = arr[:3]
        else:
            rgb = np.repeat(arr[:1], 3, axis=0)

        if rgb.dtype != np.uint8:
            rgb = _to_uint8(rgb)

        # Move CHW → HWC.
        hwc = np.transpose(rgb, (1, 2, 0))

    return Raster(data=hwc, transform=transform, crs=crs, aoi_geom=aoi)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-band 2-98% percentile rescale to uint8."""
    out = np.empty(arr.shape, dtype=np.uint8)
    for i in range(arr.shape[0]):
        band = arr[i].astype(np.float32)
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            out[i] = 0
            continue
        lo, hi = np.percentile(finite, (2.0, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
        out[i] = scaled.astype(np.uint8)
    return out


__all__ = ["load_raster"]
