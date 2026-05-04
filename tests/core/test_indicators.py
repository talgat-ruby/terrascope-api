"""Tests for zone-indicator computation (§4.2)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from shapely.geometry import LineString, Polygon, box

from core.detection.types import Detection
from core.io.indicators import compute_indicators, write_indicators


def _det(class_name: str, geom):
    return Detection(
        id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=geom.bounds,
        pixel_bbox=(0, 0, 1, 1),
        centroid=geom.centroid,
        source_model="x",
        geometry=geom,
    )


def test_counts_per_class():
    poly = Polygon([(10, 50), (10.001, 50), (10.001, 50.001), (10, 50.001)])
    line = LineString([(10, 50), (10.01, 50)])
    detections = [
        _det("building", poly),
        _det("building", poly),
        _det("road", line),
    ]
    out = compute_indicators(detections, zone=box(10, 50, 10.1, 50.1))
    by = {row["class"]: row for row in out["by_class"]}
    assert by["building"]["count"] == 2
    assert by["road"]["count"] == 1
    assert out["total_detections"] == 3


def test_polygon_area_and_density():
    poly = Polygon([(10, 50), (10.001, 50), (10.001, 50.001), (10, 50.001)])
    zone = box(10, 50, 10.1, 50.1)  # ~10km × 10km ≈ ~100 km²
    out = compute_indicators([_det("building", poly)], zone=zone)
    bld = next(r for r in out["by_class"] if r["class"] == "building")
    assert bld["total_area_m2"] > 0
    # Single detection in roughly 100 km² → density on the order of 0.01.
    assert 0 < bld["density_per_km2"] < 1.0
    assert "area_fraction" in bld


def test_line_length_no_area_for_roads():
    line = LineString([(10, 50), (10.01, 50)])  # ~700m at lat 50
    out = compute_indicators([_det("road", line)], zone=box(10, 50, 10.1, 50.1))
    road = next(r for r in out["by_class"] if r["class"] == "road")
    assert road["total_length_m"] > 500
    assert road["total_area_m2"] == 0


def test_write_indicators_emits_json_and_csv(tmp_path: Path):
    poly = Polygon([(10, 50), (10.001, 50), (10.001, 50.001), (10, 50.001)])
    detections = [_det("building", poly)]
    json_path, csv_path = write_indicators(
        detections, tmp_path, zone=box(10, 50, 10.1, 50.1)
    )
    payload = json.loads(json_path.read_text())
    assert payload["total_detections"] == 1

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["class"] == "building"
    assert int(rows[0]["count"]) == 1


def test_no_zone_means_no_density():
    poly = Polygon([(10, 50), (10.001, 50), (10.001, 50.001), (10, 50.001)])
    out = compute_indicators([_det("building", poly)])
    bld = next(r for r in out["by_class"] if r["class"] == "building")
    assert "density_per_km2" not in bld
    assert out["zone_area_m2"] is None
