"""Precision / Recall / F1 / Average Precision / mean AP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from shapely.geometry.base import BaseGeometry

from core.eval.iou import pairwise_iou
from core.eval.match import Match, greedy_match


@dataclass(slots=True)
class ClassMetrics:
    class_name: str
    iou_threshold: float
    n_pred: int
    n_truth: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    average_precision: float  # AP at this single IoU threshold


@dataclass(slots=True)
class EvalReport:
    iou_threshold: float
    by_class: list[ClassMetrics]
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_average_precision: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "by_class": [asdict(c) for c in self.by_class],
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "mean_average_precision": self.mean_average_precision,
            "notes": self.notes,
        }


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def average_precision(matches: list[Match], n_truth: int) -> float:
    """All-points AP from a list of matches (one per prediction).

    Predictions are re-sorted by descending confidence; the precision-recall
    curve is built incrementally and AP is the area under the
    monotonically-decreasing envelope (the COCO definition collapses to this
    in the single-threshold case).
    """
    if n_truth == 0:
        return 0.0
    ordered = sorted(matches, key=lambda m: m.confidence, reverse=True)
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for m in ordered:
        if m.is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_truth)

    # Interpolate: replace each precision with the max precision for any
    # recall ≥ current — the standard PR-envelope used by VOC/COCO AP.
    interp = list(precisions)
    for i in range(len(interp) - 2, -1, -1):
        if interp[i + 1] > interp[i]:
            interp[i] = interp[i + 1]

    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recalls, interp):
        ap += (r - prev_recall) * p
        prev_recall = r
    return ap


def evaluate(
    preds_by_class: dict[str, list[tuple[BaseGeometry, float]]],
    truths_by_class: dict[str, list[BaseGeometry]],
    *,
    iou_threshold: float = 0.5,
    iou_kind: str = "polygon",
    line_buffer_m: float = 5.0,
) -> EvalReport:
    """Score predictions against truths, grouped by class.

    `preds_by_class[class] = [(geom_wgs84, confidence_0_1), ...]`
    `truths_by_class[class] = [geom_wgs84, ...]`

    Classes present in either dict are evaluated; classes with no truth
    contribute only false positives, classes with no preds only false
    negatives. Geometries must be in EPSG:4326 (matches exporter output).
    """
    classes = sorted(set(preds_by_class) | set(truths_by_class))
    by_class: list[ClassMetrics] = []
    notes: list[str] = []

    total_tp = total_fp = total_fn = 0

    for cname in classes:
        preds = preds_by_class.get(cname, [])
        truths = truths_by_class.get(cname, [])
        n_pred = len(preds)
        n_truth = len(truths)

        if n_pred == 0 and n_truth == 0:
            notes.append(f"{cname}: no predictions or truths — skipped.")
            continue

        if n_pred == 0:
            tp, fp, fn = 0, 0, n_truth
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            by_class.append(
                ClassMetrics(
                    class_name=cname,
                    iou_threshold=iou_threshold,
                    n_pred=0,
                    n_truth=n_truth,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=p,
                    recall=r,
                    f1=f1,
                    average_precision=0.0,
                )
            )
            total_fn += fn
            continue

        if n_truth == 0:
            tp, fp, fn = 0, n_pred, 0
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            by_class.append(
                ClassMetrics(
                    class_name=cname,
                    iou_threshold=iou_threshold,
                    n_pred=n_pred,
                    n_truth=0,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=p,
                    recall=r,
                    f1=f1,
                    average_precision=0.0,
                )
            )
            total_fp += fp
            continue

        geoms = [g for g, _ in preds]
        confs = [c for _, c in preds]
        iou_matrix = pairwise_iou(geoms, truths, kind=iou_kind, buffer_m=line_buffer_m)
        matches, matched_truths = greedy_match(
            iou_matrix, confs, iou_threshold=iou_threshold
        )
        tp = sum(1 for m in matches if m.is_tp)
        fp = n_pred - tp
        fn = n_truth - len(matched_truths)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        ap = average_precision(matches, n_truth=n_truth)

        by_class.append(
            ClassMetrics(
                class_name=cname,
                iou_threshold=iou_threshold,
                n_pred=n_pred,
                n_truth=n_truth,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=p,
                recall=r,
                f1=f1,
                average_precision=ap,
            )
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p, micro_r, micro_f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    if by_class:
        macro_p = sum(c.precision for c in by_class) / len(by_class)
        macro_r = sum(c.recall for c in by_class) / len(by_class)
        macro_f1 = sum(c.f1 for c in by_class) / len(by_class)
        m_ap = sum(c.average_precision for c in by_class) / len(by_class)
    else:
        macro_p = macro_r = macro_f1 = m_ap = 0.0

    return EvalReport(
        iou_threshold=iou_threshold,
        by_class=by_class,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        mean_average_precision=m_ap,
        notes=notes,
    )


__all__ = [
    "ClassMetrics",
    "EvalReport",
    "average_precision",
    "evaluate",
    "precision_recall_f1",
]
