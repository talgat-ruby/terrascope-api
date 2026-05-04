"""Tests for the GeoJSON exporter (§5 schema, EPSG:4326 reprojection)."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import LineString, Polygon

from core.detection.types import Detection
from core.io.exporter import export_geojson


def _polygon_det(class_name: str = "building") -> Detection:
    poly = Polygon([(10.0, 50.0), (10.001, 50.0), (10.001, 50.001), (10.0, 50.001)])
    return Detection(
        id=0,
        class_name=class_name,
        confidence=0.42,
        bbox=poly.bounds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=poly.centroid,
        source_model="msft-buildings",
        geometry=poly,
    )


def _line_det() -> Detection:
    line = LineString([(10.0, 50.0), (10.01, 50.0)])
    return Detection(
        id=1,
        class_name="road",
        confidence=1.0,
        bbox=line.bounds,
        pixel_bbox=(0, 0, 100, 1),
        centroid=line.centroid,
        source_model="osm-roads",
        geometry=line,
    )


def test_schema_uses_assignment_field_names(tmp_path: Path):
    out = export_geojson([_polygon_det()], tmp_path / "out.geojson")
    payload = json.loads(out.read_text())
    feat = payload["features"][0]
    p = feat["properties"]
    # §5 mandatory fields under the spec'd names:
    assert p["id"] == 0
    assert p["class"] == "building"
    assert p["confidence"] == 42.0  # 0-100 scale
    assert p["source"] == "msft-buildings"  # falls back to source_model
    # Optional area_m2 present for polygons.
    assert "area_m2" in p
    assert p["area_m2"] > 0


def test_explicit_source_overrides_per_detection(tmp_path: Path):
    out = export_geojson(
        [_polygon_det()], tmp_path / "out.geojson", source="Astana_4"
    )
    payload = json.loads(out.read_text())
    assert payload["features"][0]["properties"]["source"] == "Astana_4"
    # source_model is preserved separately for provenance.
    assert payload["features"][0]["properties"]["source_model"] == "msft-buildings"


def test_lines_get_length_m_not_area(tmp_path: Path):
    out = export_geojson([_line_det()], tmp_path / "out.geojson")
    p = json.loads(out.read_text())["features"][0]["properties"]
    assert "length_m" in p
    assert p["length_m"] > 0
    assert "area_m2" not in p


def test_crs_is_wgs84_in_output(tmp_path: Path):
    out = export_geojson([_polygon_det()], tmp_path / "out.geojson")
    payload = json.loads(out.read_text())
    assert payload["crs"]["properties"]["name"] == "EPSG:4326"


def test_empty_detections_writes_empty_collection(tmp_path: Path):
    out = export_geojson([], tmp_path / "empty.geojson")
    payload = json.loads(out.read_text())
    assert payload["type"] == "FeatureCollection"
    assert payload["features"] == []


def test_reprojects_from_utm_to_wgs84(tmp_path: Path):
    """A polygon in EPSG:32643 (UTM 43N) lands in WGS84 in the output."""
    # Roughly central Kazakhstan (Astana) in UTM 43N.
    poly_utm = Polygon([
        (350000, 5650000),
        (350100, 5650000),
        (350100, 5650100),
        (350000, 5650100),
    ])
    det = Detection(
        id=0, class_name="building", confidence=0.9,
        bbox=poly_utm.bounds, pixel_bbox=(0, 0, 10, 10),
        centroid=poly_utm.centroid, source_model="x", geometry=poly_utm,
    )
    out = export_geojson([det], tmp_path / "out.geojson", crs="EPSG:32643")
    geom = json.loads(out.read_text())["features"][0]["geometry"]
    # Expect WGS84 lon~70°E, lat~50°N (the rough region for UTM 43N at this E/N).
    lon, lat = geom["coordinates"][0][0]
    assert 65 < lon < 75
    assert 45 < lat < 55
