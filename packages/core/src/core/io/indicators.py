"""Compute zone-level indicators for an exported detection set (§4.2).

Per class:
- `count`: number of detections
- `total_area_m2`: sum of geodesic polygon areas (0 for non-polygonal)
- `total_length_m`: sum of geodesic line lengths (0 for non-linear)
- `density_per_km2`: count / zone_area_km2
- `area_fraction`: total_area_m2 / zone_area_m2 (polygon classes only)

Plus a top-level `zone_area_m2` so downstream consumers can re-derive any
of the rates without re-projecting the AOI.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from pyproj import Geod
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection, Raster
from core.geo.raster_utils import raster_roi_wgs84

_GEOD = Geod(ellps="WGS84")


def compute_indicators(
    detections: list[Detection],
    *,
    zone: BaseGeometry | None = None,
    zone_area_m2: float | None = None,
    raster_crs: str = "EPSG:4326",
) -> dict:
    """Return a dict ready to serialise to JSON.

    `zone` is the AOI polygon in WGS84 (used to derive `zone_area_m2`
    when not supplied). When neither is supplied, density / area_fraction
    are reported as `None`.

    `raster_crs` is the geometries' source CRS — needed to compute area /
    length geodesically against the WGS84 ellipsoid.
    """
    z_area_m2 = zone_area_m2
    if z_area_m2 is None and zone is not None:
        z_area_m2 = abs(_GEOD.geometry_area_perimeter(zone)[0])

    needs_reproj = raster_crs.upper() not in ("EPSG:4326", "OGC:CRS84")
    per_class: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_area_m2": 0.0, "total_length_m": 0.0}
    )

    for det in detections:
        geom_native = det.geometry if det.geometry is not None else box(*det.bbox)
        if needs_reproj:
            from rasterio.warp import transform_geom
            from shapely.geometry import mapping, shape

            geom = shape(transform_geom(raster_crs, "EPSG:4326", mapping(geom_native)))
        else:
            geom = geom_native

        bucket = per_class[det.class_name]
        bucket["count"] += 1
        gtype = geom.geom_type
        if gtype in ("Polygon", "MultiPolygon"):
            bucket["total_area_m2"] += abs(_GEOD.geometry_area_perimeter(geom)[0])
        elif gtype == "LineString":
            bucket["total_length_m"] += _line_length(list(geom.coords))
        elif gtype == "MultiLineString":
            bucket["total_length_m"] += sum(
                _line_length(list(p.coords)) for p in geom.geoms
            )

    by_class: list[dict] = []
    for class_name in sorted(per_class):
        b = per_class[class_name]
        entry: dict[str, object] = {
            "class": class_name,
            "count": int(b["count"]),
            "total_area_m2": round(b["total_area_m2"], 3),
            "total_length_m": round(b["total_length_m"], 3),
        }
        if z_area_m2 and z_area_m2 > 0:
            zone_km2 = z_area_m2 / 1_000_000.0
            entry["density_per_km2"] = round(b["count"] / zone_km2, 4)
            if b["total_area_m2"] > 0:
                entry["area_fraction"] = round(b["total_area_m2"] / z_area_m2, 6)
        by_class.append(entry)

    return {
        "zone_area_m2": round(z_area_m2, 3) if z_area_m2 is not None else None,
        "total_detections": len(detections),
        "by_class": by_class,
    }


def write_indicators(
    detections: list[Detection],
    output_dir: str | Path,
    *,
    raster: Raster | None = None,
    zone: BaseGeometry | None = None,
) -> tuple[Path, Path]:
    """Write `indicators.json` + `indicators.csv` and return both paths.

    When `raster` is given, the AOI is taken from `raster.aoi_geom` if set,
    falling back to the raster footprint via `raster_roi_wgs84`. An
    explicit `zone` argument always wins.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z = zone
    raster_crs = raster.crs if raster is not None else "EPSG:4326"
    if z is None and raster is not None:
        if raster.aoi_geom is not None:
            z = raster.aoi_geom
        else:
            lon_min, lat_min, lon_max, lat_max = raster_roi_wgs84(raster)
            z = box(lon_min, lat_min, lon_max, lat_max)

    payload = compute_indicators(detections, zone=z, raster_crs=raster_crs)

    json_path = output_dir / "indicators.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = output_dir / "indicators.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class", "count", "total_area_m2", "total_length_m",
             "density_per_km2", "area_fraction"]
        )
        for row in payload["by_class"]:
            writer.writerow(
                [
                    row["class"],
                    row["count"],
                    row["total_area_m2"],
                    row["total_length_m"],
                    row.get("density_per_km2", ""),
                    row.get("area_fraction", ""),
                ]
            )
    return json_path, csv_path


def _line_length(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
        total += dist
    return total


__all__ = ["compute_indicators", "write_indicators"]
