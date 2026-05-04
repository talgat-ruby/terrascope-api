"""Minimal helpers needed for SAM-Road inference.

Lifted (and trimmed) from the upstream `dataset.py`, `graph_extraction.py`,
and `graph_utils.py`. Drops the training/visualization code paths and their
heavy deps (cv2, networkx, igraph, sklearn, addict, tcod, skimage).
"""

from __future__ import annotations

import numpy as np
import scipy.spatial


def get_patch_info_one_img(
    image_index: int,
    image_size: int,
    sample_margin: int,
    patch_size: int,
    patches_per_edge: int,
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    """Return ``patches_per_edge × patches_per_edge`` evenly spaced patches.

    Each entry is ``(image_index, (x0, y0), (x1, y1))`` with patches sized
    ``patch_size × patch_size`` and offset by at least ``sample_margin``
    pixels from the image border.
    """
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


def get_points_and_scores_from_mask(
    mask: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    rcs = np.column_stack(np.where(mask > threshold))
    xys = rcs[:, ::-1]
    scores = mask[mask > threshold]
    return xys, scores


def nms_points(
    points: np.ndarray, scores: np.ndarray, radius: float
) -> np.ndarray:
    """Greedy radius-NMS. Scores > 1.0 are forced kept regardless."""
    if points.shape[0] == 0:
        return points
    sorted_indices = np.argsort(scores)[::-1]
    sorted_points = points[sorted_indices, :]
    sorted_scores = scores[sorted_indices]
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = scipy.spatial.KDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        neighbor_indices = tree.query_ball_point(p, r=radius)
        neighbor_scores = sorted_scores[neighbor_indices]
        keep_nbr = np.greater(neighbor_scores, 1.0)
        kept[neighbor_indices] = keep_nbr
        kept[idx] = True
    return sorted_points[kept]


def extract_graph_points(
    keypoint_mask: np.ndarray, road_mask: np.ndarray, config
) -> np.ndarray:
    """Combine keypoint and road masks into a deduped (x, y) point set."""
    kp_candidates, kp_scores = get_points_and_scores_from_mask(
        keypoint_mask, config.ITSC_THRESHOLD * 255
    )
    kps_0 = nms_points(kp_candidates, kp_scores, config.ITSC_NMS_RADIUS)
    kp_candidates, kp_scores = get_points_and_scores_from_mask(
        road_mask, config.ROAD_THRESHOLD * 255
    )
    kps_1 = nms_points(kp_candidates, kp_scores, config.ROAD_NMS_RADIUS)
    # Prioritize intersection points (force-keep with score=1.0+epsilon-style flag).
    if kps_0.shape[0] == 0 and kps_1.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    kp_candidates = np.concatenate([kps_0, kps_1], axis=0)
    kp_scores = np.concatenate(
        [np.ones((kps_0.shape[0],)), np.zeros((kps_1.shape[0],))], axis=0
    )
    return nms_points(kp_candidates, kp_scores, config.ROAD_NMS_RADIUS)
