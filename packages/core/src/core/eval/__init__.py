"""Evaluation: Precision / Recall / F1 / mAP / IoU against a ground-truth GeoJSON.

Satisfies §7 of the assignment. Two flavours of IoU are supported so the same
machinery can score polygon classes (buildings, vehicles-as-OBB) and linear
classes (roads): polygon IoU for areal geometries, buffered-line IoU for
LineString / MultiLineString.
"""

from core.eval.iou import buffered_line_iou, pairwise_iou, polygon_iou
from core.eval.loader import load_features
from core.eval.match import greedy_match
from core.eval.metrics import (
    ClassMetrics,
    EvalReport,
    average_precision,
    evaluate,
    precision_recall_f1,
)

__all__ = [
    "ClassMetrics",
    "EvalReport",
    "average_precision",
    "buffered_line_iou",
    "evaluate",
    "greedy_match",
    "load_features",
    "pairwise_iou",
    "polygon_iou",
    "precision_recall_f1",
]
