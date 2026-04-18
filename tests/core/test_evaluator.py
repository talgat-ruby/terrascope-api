"""Tests for QualityEvaluatorService and VisualizationService."""

import json

import numpy as np
from shapely.geometry import box

from core.services.detector import RawDetection
from core.services.evaluator import QualityEvaluatorService


def _det(class_name: str, geom, confidence: int = 80) -> RawDetection:
    return RawDetection(
        class_name=class_name, confidence=confidence, geometry=geom, source="test"
    )


# --- QualityEvaluatorService tests ---


class TestEvaluate:
    def test_perfect_match(self):
        """All predictions match all ground truth exactly."""
        geom = box(0, 0, 1, 1)
        preds = [_det("building", geom)]
        gts = [_det("building", geom)]

        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate(preds, gts, iou_threshold=0.5)

        assert len(result.metrics) == 1
        m = result.metrics[0]
        assert m.class_name == "building"
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.iou == 1.0
        assert m.true_positives == 1
        assert m.false_positives == 0
        assert m.false_negatives == 0

    def test_no_overlap(self):
        """Predictions and ground truth don't overlap."""
        preds = [_det("building", box(0, 0, 1, 1))]
        gts = [_det("building", box(5, 5, 6, 6))]

        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate(preds, gts, iou_threshold=0.5)

        m = result.metrics[0]
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1

    def test_partial_overlap_above_threshold(self):
        """Overlapping polygons with IoU > threshold."""
        pred_geom = box(0, 0, 1, 1)
        gt_geom = box(0.2, 0.2, 1.2, 1.2)
        # IoU = 0.64 / 1.36 ≈ 0.47 -- below 0.5 with these exact coords

        evaluator = QualityEvaluatorService()

        # Use overlapping boxes that exceed threshold
        pred_geom = box(0, 0, 1, 1)
        gt_geom = box(0.1, 0.1, 1.0, 1.0)  # IoU > 0.5

        preds = [_det("building", pred_geom)]
        gts = [_det("building", gt_geom)]
        result = evaluator.evaluate(preds, gts, iou_threshold=0.5)

        m = result.metrics[0]
        assert m.true_positives == 1
        assert m.precision == 1.0
        assert m.recall == 1.0

    def test_multiple_classes(self):
        """Metrics are computed per class."""
        preds = [
            _det("building", box(0, 0, 1, 1)),
            _det("road", box(2, 2, 3, 3)),
        ]
        gts = [
            _det("building", box(0, 0, 1, 1)),
            _det("vegetation", box(4, 4, 5, 5)),
        ]

        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate(preds, gts, iou_threshold=0.5)

        class_names = {m.class_name for m in result.metrics}
        assert class_names == {"building", "road", "vegetation"}

        building = next(m for m in result.metrics if m.class_name == "building")
        assert building.precision == 1.0
        assert building.recall == 1.0

        road = next(m for m in result.metrics if m.class_name == "road")
        assert road.false_positives == 1
        assert road.precision == 0.0

        veg = next(m for m in result.metrics if m.class_name == "vegetation")
        assert veg.false_negatives == 1
        assert veg.recall == 0.0

    def test_empty_predictions(self):
        """No predictions, some ground truth."""
        gts = [_det("building", box(0, 0, 1, 1))]
        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate([], gts, iou_threshold=0.5)

        m = result.metrics[0]
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.false_negatives == 1

    def test_empty_ground_truth(self):
        """Predictions but no ground truth."""
        preds = [_det("building", box(0, 0, 1, 1))]
        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate(preds, [], iou_threshold=0.5)

        m = result.metrics[0]
        assert m.precision == 0.0
        assert m.false_positives == 1

    def test_both_empty(self):
        """No predictions, no ground truth."""
        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate([], [], iou_threshold=0.5)
        assert len(result.metrics) == 0

    def test_greedy_matching_prevents_double_match(self):
        """Each GT can only be matched once, even if multiple preds overlap it."""
        gt_geom = box(0, 0, 1, 1)
        pred1 = _det("building", box(0, 0, 1, 1), confidence=90)  # Perfect match
        pred2 = _det("building", box(0, 0, 1, 1), confidence=80)  # Also perfect

        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate(
            [pred1, pred2], [_det("building", gt_geom)], iou_threshold=0.5
        )

        m = result.metrics[0]
        assert m.true_positives == 1
        assert m.false_positives == 1

    def test_error_examples_collected(self):
        """Error examples include FP and FN."""
        preds = [_det("building", box(0, 0, 1, 1))]
        gts = [_det("building", box(5, 5, 6, 6))]

        evaluator = QualityEvaluatorService()
        result = evaluator.evaluate(preds, gts, iou_threshold=0.5)

        fp_errors = [e for e in result.error_examples if e["type"] == "false_positive"]
        fn_errors = [e for e in result.error_examples if e["type"] == "false_negative"]
        assert len(fp_errors) == 1
        assert len(fn_errors) == 1
        assert fp_errors[0]["class"] == "building"

    def test_iou_threshold_sensitivity(self):
        """Higher threshold requires tighter match."""
        pred_geom = box(0, 0, 1, 1)
        gt_geom = box(0.1, 0.1, 1.0, 1.0)

        evaluator = QualityEvaluatorService()

        # At 0.5 threshold -- should match
        r1 = evaluator.evaluate(
            [_det("building", pred_geom)],
            [_det("building", gt_geom)],
            iou_threshold=0.5,
        )

        # At 0.95 threshold -- should not match
        r2 = evaluator.evaluate(
            [_det("building", pred_geom)],
            [_det("building", gt_geom)],
            iou_threshold=0.95,
        )

        assert r1.metrics[0].true_positives == 1
        assert r2.metrics[0].true_positives == 0


class TestGenerateReport:
    def test_report_written(self, tmp_path):
        """Report is written as valid JSON."""
        evaluator = QualityEvaluatorService()
        preds = [_det("building", box(0, 0, 1, 1))]
        gts = [_det("building", box(0, 0, 1, 1))]
        result = evaluator.evaluate(preds, gts)

        report_path = evaluator.generate_report(result, tmp_path / "report.json")

        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "metrics" in data
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["class_name"] == "building"


class TestControlSample:
    def test_sample_size(self):
        """Sample returns correct number of detections."""
        evaluator = QualityEvaluatorService()
        dets = [_det("building", box(i, i, i + 1, i + 1)) for i in range(100)]

        sample = evaluator.create_control_sample(dets, sample_size=10, seed=42)
        assert len(sample) == 10

    def test_sample_smaller_than_input(self):
        """When input is smaller than sample_size, return all."""
        evaluator = QualityEvaluatorService()
        dets = [_det("building", box(0, 0, 1, 1))]

        sample = evaluator.create_control_sample(dets, sample_size=50)
        assert len(sample) == 1

    def test_sample_reproducible(self):
        """Same seed produces same sample."""
        evaluator = QualityEvaluatorService()
        dets = [_det("building", box(i, i, i + 1, i + 1)) for i in range(100)]

        s1 = evaluator.create_control_sample(dets, sample_size=10, seed=42)
        s2 = evaluator.create_control_sample(dets, sample_size=10, seed=42)
        assert [d.geometry.bounds for d in s1] == [d.geometry.bounds for d in s2]


# --- VisualizationService tests ---


class TestVisualization:
    def test_render_overlay(self, tmp_path):
        """Render overlay produces a PNG file."""
        from core.services.visualization import VisualizationService

        # Create a synthetic GeoTIFF
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "test.tif"
        transform = from_bounds(0, 0, 1, 1, 100, 100)
        data = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)

        with rasterio.open(
            str(tif_path),
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        detections = [
            _det("building", box(0.1, 0.1, 0.3, 0.3)),
            _det("vegetation", box(0.5, 0.5, 0.8, 0.8)),
        ]

        viz = VisualizationService()
        output = viz.render_overlay(tif_path, detections, tmp_path / "overlay.png")

        assert output.exists()
        assert output.suffix == ".png"
        assert output.stat().st_size > 0

    def test_render_overlay_empty_detections(self, tmp_path):
        """Render overlay works with no detections."""
        from core.services.visualization import VisualizationService

        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "test.tif"
        transform = from_bounds(0, 0, 1, 1, 50, 50)
        data = np.zeros((3, 50, 50), dtype=np.uint8)

        with rasterio.open(
            str(tif_path),
            "w",
            driver="GTiff",
            height=50,
            width=50,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        viz = VisualizationService()
        output = viz.render_overlay(tif_path, [], tmp_path / "empty_overlay.png")

        assert output.exists()
