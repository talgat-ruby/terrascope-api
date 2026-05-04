"""Raster CRS / bbox helpers shared across vector-data processes.

These were previously duplicated in `osm_roads`, `msft_footprints`, and
`renderer`. Pulling them into one module avoids drift and lets bug fixes
land in one place.
"""

from __future__ import annotations

from rasterio.transform import rowcol
from rasterio.warp import transform_bounds

from core.detection.types import Raster


_WGS84_CRS_TAGS = ("EPSG:4326", "OGC:CRS84")


def raster_roi_wgs84(raster: Raster) -> tuple[float, float, float, float]:
    """Return ``(lon_min, lat_min, lon_max, lat_max)`` for the raster footprint.

    Honours ``raster.aoi_geom`` when present (and the raster is in WGS84);
    otherwise computes from the affine transform + width/height. Non-WGS84
    rasters are reprojected via ``rasterio.warp.transform_bounds``.
    """
    if raster.aoi_geom is not None and raster.crs.upper() in _WGS84_CRS_TAGS:
        minx, miny, maxx, maxy = raster.aoi_geom.bounds
        return (minx, miny, maxx, maxy)

    a = raster.transform
    left, top = a.c, a.f
    right = a.c + a.a * raster.width
    bottom = a.f + a.e * raster.height
    minx, maxx = sorted((left, right))
    miny, maxy = sorted((bottom, top))

    if raster.crs.upper() in _WGS84_CRS_TAGS:
        return (minx, miny, maxx, maxy)
    return tuple(  # type: ignore[return-value]
        transform_bounds(raster.crs, "EPSG:4326", minx, miny, maxx, maxy)
    )


def bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    raster: Raster,
) -> tuple[int, int, int, int]:
    """Convert a WGS84 bbox to pixel ``(col_min, row_min, col_max, row_max)``.

    Reprojects to the raster's native CRS first when necessary.
    """
    minx, miny, maxx, maxy = bbox
    if raster.crs.upper() not in _WGS84_CRS_TAGS:
        minx, miny, maxx, maxy = transform_bounds(
            "EPSG:4326", raster.crs, minx, miny, maxx, maxy
        )
    r1, c1 = rowcol(raster.transform, minx, maxy)
    r2, c2 = rowcol(raster.transform, maxx, miny)
    row_min, row_max = sorted((int(r1), int(r2)))
    col_min, col_max = sorted((int(c1), int(c2)))
    return (col_min, row_min, col_max, row_max)


__all__ = ["bbox_to_pixels", "raster_roi_wgs84"]
