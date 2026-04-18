"""QualityEvaluatorService -- precision, recall, F1, IoU, mAP evaluation."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from pyproj import Geod
from shapely.geometry.base import BaseGeometry

from core.services.detector import RawDetection


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics."""

    class_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou: float = 0.0
    map: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class EvaluationResult:
    """Full evaluation result across all classes."""

    metrics: list[ClassMetrics] = field(default_factory=list)
    error_examples: list[dict] = field(default_factory=list)


class QualityEvaluatorService:
    """Evaluates detection quality against ground truth."""

    def __init__(self) -> None:
        self._geod = Geod(ellps="WGS84")

    def evaluate(
        self,
        predictions: list[RawDetection],
        ground_truth: list[RawDetection],
        iou_threshold: float = 0.5,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth, returning per-class metrics.

        Uses IoU-based matching: a prediction is a true positive if it overlaps
        a ground truth polygon with IoU >= threshold. Each ground truth polygon
        can only be matched once (greedy, highest IoU first).

        Args:
            predictions: Detected features.
            ground_truth: Reference features.
            iou_threshold: Minimum IoU to count as a match.

        Returns:
            EvaluationResult with per-class metrics and error examples.
        """
        # Group by class
        pred_by_class: dict[str, list[RawDetection]] = {}
        for p in predictions:
            pred_by_class.setdefault(p.class_name, []).append(p)

        gt_by_class: dict[str, list[RawDetection]] = {}
        for g in ground_truth:
            gt_by_class.setdefault(g.class_name, []).append(g)

        all_classes = sorted(set(pred_by_class.keys()) | set(gt_by_class.keys()))

        metrics: list[ClassMetrics] = []
        error_examples: list[dict] = []

        for class_name in all_classes:
            preds = pred_by_class.get(class_name, [])
            gts = gt_by_class.get(class_name, [])

            cm, errors = self._evaluate_class(preds, gts, iou_threshold, class_name)
            metrics.append(cm)
            error_examples.extend(errors)

        return EvaluationResult(metrics=metrics, error_examples=error_examples)

    def _evaluate_class(
        self,
        predictions: list[RawDetection],
        ground_truth: list[RawDetection],
        iou_threshold: float,
        class_name: str,
    ) -> tuple[ClassMetrics, list[dict]]:
        """Evaluate a single class using greedy IoU matching."""
        if not predictions and not ground_truth:
            return ClassMetrics(class_name=class_name), []

        # Compute IoU matrix and match greedily
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        iou_scores: list[float] = []
        errors: list[dict] = []

        # Build all pairs sorted by IoU descending (greedy matching)
        pairs: list[tuple[float, int, int]] = []
        for pi, pred in enumerate(predictions):
            for gi, gt in enumerate(ground_truth):
                iou = self._compute_iou(pred.geometry, gt.geometry)
                if iou >= iou_threshold:
                    pairs.append((iou, pi, gi))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou, pi, gi in pairs:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
            iou_scores.append(iou)

        tp = len(matched_pred)
        fp = len(predictions) - tp
        fn = len(ground_truth) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

        # Compute AP (area under precision-recall curve at various thresholds)
        ap = self._compute_ap(predictions, ground_truth, iou_threshold)

        # Collect error examples (false positives and false negatives)
        for pi, pred in enumerate(predictions):
            if pi not in matched_pred:
                errors.append(
                    {
                        "type": "false_positive",
                        "class": class_name,
                        "confidence": pred.confidence,
                        "geometry_wkt": pred.geometry.wkt,
                    }
                )

        for gi, gt in enumerate(ground_truth):
            if gi not in matched_gt:
                errors.append(
                    {
                        "type": "false_negative",
                        "class": class_name,
                        "geometry_wkt": gt.geometry.wkt,
                    }
                )

        return (
            ClassMetrics(
                class_name=class_name,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                iou=round(mean_iou, 4),
                map=round(ap, 4),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
            ),
            errors,
        )

    def _compute_iou(self, geom_a: BaseGeometry, geom_b: BaseGeometry) -> float:
        """Compute Intersection over Union between two geometries."""
        if not geom_a.intersects(geom_b):
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        union = geom_a.area + geom_b.area - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def _compute_ap(
        self,
        predictions: list[RawDetection],
        ground_truth: list[RawDetection],
        iou_threshold: float,
    ) -> float:
        """Compute Average Precision using 11-point interpolation.

        Predictions are sorted by confidence. At each confidence level,
        precision and recall are computed, then AP is the mean of max
        precisions at 11 recall levels [0.0, 0.1, ..., 1.0].
        """
        if not ground_truth:
            return 0.0 if predictions else 1.0

        # Sort predictions by confidence descending
        sorted_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)

        matched_gt: set[int] = set()
        tp_cumsum: list[int] = []
        fp_cumsum: list[int] = []
        tp = 0
        fp = 0

        for pred in sorted_preds:
            best_iou = 0.0
            best_gi = -1
            for gi, gt in enumerate(ground_truth):
                if gi in matched_gt:
                    continue
                iou = self._compute_iou(pred.geometry, gt.geometry)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1

            tp_cumsum.append(tp)
            fp_cumsum.append(fp)

        # Compute precision/recall at each prediction
        n_gt = len(ground_truth)
        precisions = [
            tp_cumsum[i] / (tp_cumsum[i] + fp_cumsum[i])
            for i in range(len(sorted_preds))
        ]
        recalls = [tp_cumsum[i] / n_gt for i in range(len(sorted_preds))]

        # 11-point interpolation
        ap = 0.0
        for t in [i / 10.0 for i in range(11)]:
            p_interp = max(
                (p for p, r in zip(precisions, recalls) if r >= t), default=0.0
            )
            ap += p_interp
        ap /= 11.0

        return ap

    def generate_report(
        self,
        result: EvaluationResult,
        output_path: str | Path,
    ) -> Path:
        """Generate a JSON evaluation report.

        Args:
            result: EvaluationResult from evaluate().
            output_path: Path to write the JSON report.

        Returns:
            Path to the written report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metrics": [
                {
                    "class_name": m.class_name,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "iou": m.iou,
                    "map": m.map,
                    "true_positives": m.true_positives,
                    "false_positives": m.false_positives,
                    "false_negatives": m.false_negatives,
                }
                for m in result.metrics
            ],
            "error_examples": result.error_examples[:50],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return output_path

    def create_control_sample(
        self,
        detections: list[RawDetection],
        sample_size: int = 50,
        seed: int | None = None,
    ) -> list[RawDetection]:
        """Create a random control sample for manual review.

        Args:
            detections: Full detection list.
            sample_size: Number of detections to sample.
            seed: Random seed for reproducibility.

        Returns:
            Sampled subset of detections.
        """
        if seed is not None:
            random.seed(seed)
        if len(detections) <= sample_size:
            return list(detections)
        return random.sample(detections, sample_size)
