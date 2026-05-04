"""IoU helpers for polygon and line geometries (WGS84 input).

Areas / lengths are computed geodesically via `pyproj.Geod` against the WGS84
ellipsoid — the same convention used by the exporter and indicators modules,
so an IoU here is consistent with the area_m2 / length_m fields on disk.

For LineString classes (e.g. roads) we use buffered-line IoU: both prediction
and truth are buffered by `buffer_m` and standard polygon IoU is taken on the
result. This is the standard road-extraction protocol (e.g. SpaceNet roads).
"""

from __future__ import annotations

from pyproj import Geod
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

_GEOD = Geod(ellps="WGS84")


def _geodesic_area_m2(geom: BaseGeometry) -> float:
    if geom.is_empty:
        return 0.0
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return 0.0
    return abs(_GEOD.geometry_area_perimeter(geom)[0])


def polygon_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """Geodesic IoU on polygonal geometries. Returns 0 on empty/non-polygon."""
    if a.is_empty or b.is_empty:
        return 0.0
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b)
    inter_area = _geodesic_area_m2(inter)
    if inter_area <= 0:
        return 0.0
    union_area = _geodesic_area_m2(a) + _geodesic_area_m2(b) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def buffered_line_iou(
    a: BaseGeometry, b: BaseGeometry, *, buffer_m: float = 5.0
) -> float:
    """IoU on `a` and `b` after buffering each by `buffer_m` metres.

    Both inputs may be LineString / MultiLineString / Polygon — non-line
    inputs are accepted unchanged so this function works for road truth that
    happens to be polygonal (e.g. OSM way buffers).

    The buffer is applied in a local azimuthal-equidistant projection
    centred on the geometry, so `buffer_m` is metric regardless of latitude.
    """
    if a.is_empty or b.is_empty:
        return 0.0
    a_buf = _buffer_metric(a, buffer_m)
    b_buf = _buffer_metric(b, buffer_m)
    return polygon_iou(a_buf, b_buf)


def pairwise_iou(
    preds: list[BaseGeometry],
    truths: list[BaseGeometry],
    *,
    kind: str = "polygon",
    buffer_m: float = 5.0,
) -> list[list[float]]:
    """Return an N_pred × N_truth IoU matrix. `kind` ∈ {"polygon", "line"}."""
    if kind == "polygon":
        fn = polygon_iou
        return [[fn(p, t) for t in truths] for p in preds]
    if kind == "line":
        # Pre-buffer once per geometry — buffering dominates cost for lines.
        preds_buf = [_buffer_metric(p, buffer_m) for p in preds]
        truths_buf = [_buffer_metric(t, buffer_m) for t in truths]
        return [[polygon_iou(p, t) for t in truths_buf] for p in preds_buf]
    raise ValueError(f"unknown iou kind: {kind!r}")


def _buffer_metric(geom: BaseGeometry, buffer_m: float) -> BaseGeometry:
    """Buffer `geom` (lon/lat) by `buffer_m` metres via local AEQD projection."""
    if buffer_m <= 0:
        return geom
    cx, cy = _representative_lonlat(geom)
    # WGS84 -> local AEQD around (cx, cy). Inputs are degrees, output metres.
    from pyproj import Transformer

    fwd = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    ).transform
    inv = Transformer.from_crs(
        f"+proj=aeqd +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m +no_defs",
        "EPSG:4326",
        always_xy=True,
    ).transform
    projected = transform(fwd, geom)
    buffered = projected.buffer(buffer_m)
    return transform(inv, buffered)


def _representative_lonlat(geom: BaseGeometry) -> tuple[float, float]:
    pt = geom.representative_point()
    return float(pt.x), float(pt.y)


def union_all(geoms: list[BaseGeometry]) -> BaseGeometry:
    """Convenience: GEOSException-safe union for class-level coverage IoU."""
    if not geoms:
        return shape({"type": "GeometryCollection", "geometries": []})
    return unary_union([shape(mapping(g)) for g in geoms])


__all__ = [
    "buffered_line_iou",
    "pairwise_iou",
    "polygon_iou",
    "union_all",
]
