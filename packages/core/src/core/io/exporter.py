"""Export Detection lists to GeoJSON.

Output is reprojected to EPSG:4326 so the resulting layer opens cleanly in
QGIS aligned to any web base map without manual CRS adjustment, as
required by the assignment (§5, §10). Each feature carries the schema
described in §5: id, class, confidence (0-100), source, geometry, plus
optional area_m2 / length_m derived geodesically (pyproj.Geod, WGS84).
"""

from __future__ import annotations

import json
from pathlib import Path

from pyproj import Geod
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection

_GEOD = Geod(ellps="WGS84")
_WGS84 = "EPSG:4326"


def export_geojson(
    detections: list[Detection],
    output_path: str | Path,
    *,
    crs: str = _WGS84,
    source: str | None = None,
) -> Path:
    """Write `detections` as a GeoJSON FeatureCollection in EPSG:4326.

    `crs` is the geometries' source CRS (typically the raster CRS); they
    are reprojected to WGS84 on the way out. `source` is the originating
    scene identifier (e.g. raster filename stem) attached to every feature
    — falls back to each detection's `source_model` when omitted.

    Each feature exposes the §5 attribute schema:

        id, class, confidence (0-100), source, geometry,
        area_m2 (polygons), length_m (linestrings), centroid_wkt,
        bbox, pixel_bbox, source_model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    needs_reproj = crs.upper() not in (_WGS84, "OGC:CRS84")
    features: list[dict] = []
    for det in detections:
        geom_native: BaseGeometry = (
            det.geometry if det.geometry is not None else box(*det.bbox)
        )
        if needs_reproj:
            geom_wgs84 = shape(transform_geom(crs, _WGS84, mapping(geom_native)))
        else:
            geom_wgs84 = geom_native

        props: dict[str, object] = {
            "id": det.id,
            "class": det.class_name,
            # §5: confidence on 0-100. Round to 1 decimal for compact output.
            "confidence": round(float(det.confidence) * 100.0, 1),
            "source": source if source is not None else det.source_model,
            "source_model": det.source_model,
            "centroid_wkt": det.centroid.wkt,
            "bbox": list(det.bbox),
            "pixel_bbox": list(det.pixel_bbox),
        }
        area_m2 = _geodesic_area_m2(geom_wgs84)
        if area_m2 is not None:
            props["area_m2"] = round(area_m2, 3)
        length_m = _geodesic_length_m(geom_wgs84)
        if length_m is not None:
            props["length_m"] = round(length_m, 3)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom_wgs84),
                "properties": props,
            }
        )

    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": _WGS84}},
        "features": features,
    }
    output_path.write_text(json.dumps(payload))
    return output_path


def _geodesic_area_m2(geom: BaseGeometry) -> float | None:
    """Polygon / MultiPolygon → geodesic area in m² (else None)."""
    gtype = geom.geom_type
    if gtype not in ("Polygon", "MultiPolygon"):
        return None
    return abs(_GEOD.geometry_area_perimeter(geom)[0])


def _geodesic_length_m(geom: BaseGeometry) -> float | None:
    """LineString / MultiLineString → geodesic length in m (else None)."""
    gtype = geom.geom_type
    if gtype == "LineString":
        return _line_length(list(geom.coords))
    if gtype == "MultiLineString":
        return sum(_line_length(list(part.coords)) for part in geom.geoms)
    return None


def _line_length(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
        total += dist
    return total


__all__ = ["export_geojson"]
