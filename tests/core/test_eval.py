"""Tests for the evaluation module (Precision/Recall/F1/AP/IoU)."""

from __future__ import annotations

import json

from shapely.geometry import LineString, Polygon

from core.eval import (
    average_precision,
    buffered_line_iou,
    evaluate,
    greedy_match,
    load_features,
    pairwise_iou,
    polygon_iou,
)
from core.eval.match import Match


def _square(lon: float, lat: float, side_deg: float = 0.001) -> Polygon:
    return Polygon(
        [
            (lon, lat),
            (lon + side_deg, lat),
            (lon + side_deg, lat + side_deg),
            (lon, lat + side_deg),
        ]
    )


def test_polygon_iou_identity_and_disjoint() -> None:
    a = _square(76.9, 43.2)
    assert polygon_iou(a, a) == 1.0
    b = _square(80.0, 50.0)
    assert polygon_iou(a, b) == 0.0


def test_polygon_iou_half_overlap() -> None:
    a = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    b = Polygon([(1, 0), (3, 0), (3, 1), (1, 1)])
    iou = polygon_iou(a, b)
    # 1/3 in planar — geodesic at near-equator is very close to that.
    assert 0.30 < iou < 0.36


def test_buffered_line_iou_overlap() -> None:
    a = LineString([(76.9, 43.2), (76.91, 43.2)])
    b = LineString([(76.9, 43.20001), (76.91, 43.20001)])
    iou = buffered_line_iou(a, b, buffer_m=10.0)
    assert iou > 0.5  # near-identical lines, generous buffer
    far = LineString([(70.0, 40.0), (70.01, 40.0)])
    assert buffered_line_iou(a, far, buffer_m=5.0) == 0.0


def test_pairwise_iou_shape() -> None:
    preds = [_square(0, 0), _square(1, 1)]
    truths = [_square(0, 0), _square(1, 1), _square(2, 2)]
    m = pairwise_iou(preds, truths, kind="polygon")
    assert len(m) == 2 and len(m[0]) == 3
    assert m[0][0] == 1.0
    assert m[1][1] == 1.0
    assert m[0][2] == 0.0


def test_greedy_match_picks_best_truth() -> None:
    # Pred 0 has high confidence and matches truth 1 better.
    iou = [[0.6, 0.9], [0.7, 0.0]]
    matches, matched = greedy_match(iou, [0.9, 0.5], iou_threshold=0.5)
    assert matched == {0, 1}
    by_pred = {m.pred_index: m for m in matches}
    assert by_pred[0].truth_index == 1 and by_pred[0].is_tp
    assert by_pred[1].truth_index == 0 and by_pred[1].is_tp


def test_greedy_match_threshold_drops_weak() -> None:
    iou = [[0.4, 0.0]]
    matches, matched = greedy_match(iou, [0.9], iou_threshold=0.5)
    assert matched == set()
    assert matches[0].is_tp is False


def test_average_precision_perfect() -> None:
    matches = [
        Match(pred_index=0, truth_index=0, iou=1.0, confidence=0.9, is_tp=True),
        Match(pred_index=1, truth_index=1, iou=1.0, confidence=0.8, is_tp=True),
    ]
    assert average_precision(matches, n_truth=2) == 1.0


def test_average_precision_no_truth() -> None:
    assert average_precision([], n_truth=0) == 0.0


def test_evaluate_polygon_end_to_end() -> None:
    truth = [_square(0, 0), _square(1, 1), _square(2, 2)]
    # 2 TP (perfect overlap), 1 FP elsewhere, 1 FN (truth at 2,2 missed).
    preds = [
        (_square(0, 0), 0.9),
        (_square(1, 1), 0.8),
        (_square(5, 5), 0.7),
    ]
    report = evaluate(
        {"building": preds},
        {"building": truth},
        iou_threshold=0.5,
        iou_kind="polygon",
    )
    assert len(report.by_class) == 1
    c = report.by_class[0]
    assert c.tp == 2 and c.fp == 1 and c.fn == 1
    assert abs(c.precision - 2 / 3) < 1e-6
    assert abs(c.recall - 2 / 3) < 1e-6
    assert 0 < c.average_precision <= 1.0


def test_evaluate_handles_missing_class_sides() -> None:
    # Class only on truth side -> all FN.
    report = evaluate({}, {"road": [_square(0, 0)]})
    c = report.by_class[0]
    assert c.tp == 0 and c.fp == 0 and c.fn == 1
    # Class only on pred side -> all FP.
    report = evaluate({"car": [(_square(0, 0), 0.9)]}, {})
    c = report.by_class[0]
    assert c.tp == 0 and c.fp == 1 and c.fn == 0


def test_load_features_normalises_confidence_and_remap(tmp_path) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"class": "car", "confidence": 87},
                "geometry": _square(0, 0).__geo_interface__,
            },
            {
                "type": "Feature",
                "properties": {"class": "tree"},
                "geometry": _square(1, 1).__geo_interface__,
            },
        ],
    }
    p = tmp_path / "truth.geojson"
    p.write_text(json.dumps(payload))
    out = load_features(
        p,
        class_remap={"car": "small vehicle"},
        class_filter={"small vehicle"},
    )
    assert len(out) == 1
    cname, _, conf = out[0]
    assert cname == "small vehicle"
    # 87 (0-100) -> 0.87
    assert abs(conf - 0.87) < 1e-9


def test_load_features_rejects_non_wgs84(tmp_path) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32643"}},
        "features": [],
    }
    p = tmp_path / "truth.geojson"
    p.write_text(json.dumps(payload))
    try:
        load_features(p)
    except ValueError as e:
        assert "EPSG:32643" in str(e)
    else:
        raise AssertionError("expected ValueError for non-WGS84 input")
