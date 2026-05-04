"""Greedy IoU matching between predictions and ground-truth.

Predictions are sorted by descending confidence; each prediction is matched to
the highest-IoU unmatched truth above `iou_threshold`. This is the standard
COCO-style matching used to derive precision/recall and AP curves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Match:
    pred_index: int
    truth_index: int | None  # None == unmatched (false positive)
    iou: float
    confidence: float
    is_tp: bool


def greedy_match(
    iou_matrix: list[list[float]],
    confidences: list[float],
    *,
    iou_threshold: float = 0.5,
) -> tuple[list[Match], set[int]]:
    """Greedy match preds → truths at `iou_threshold`.

    Returns `(matches, matched_truths)` where `matches` has one entry per
    prediction (in order of descending confidence) and `matched_truths`
    contains the indices of truths that were assigned. Truths not in that
    set are false negatives.
    """
    n_preds = len(iou_matrix)
    n_truths = len(iou_matrix[0]) if iou_matrix else 0
    if len(confidences) != n_preds:
        raise ValueError("confidences length must equal number of predictions")

    order = sorted(range(n_preds), key=lambda i: confidences[i], reverse=True)
    matched_truths: set[int] = set()
    matches: list[Match] = []

    for pi in order:
        best_iou = 0.0
        best_ti: int | None = None
        for ti in range(n_truths):
            if ti in matched_truths:
                continue
            iou = iou_matrix[pi][ti]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_ti = ti
        if best_ti is not None:
            matched_truths.add(best_ti)
            matches.append(
                Match(
                    pred_index=pi,
                    truth_index=best_ti,
                    iou=best_iou,
                    confidence=confidences[pi],
                    is_tp=True,
                )
            )
        else:
            matches.append(
                Match(
                    pred_index=pi,
                    truth_index=None,
                    iou=0.0,
                    confidence=confidences[pi],
                    is_tp=False,
                )
            )
    return matches, matched_truths


__all__ = ["Match", "greedy_match"]
