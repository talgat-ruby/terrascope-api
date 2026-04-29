"""Tests for SegformerLandscapeDetector with a stub model."""

from unittest.mock import MagicMock

import numpy as np
import torch
from rasterio.transform import Affine

from core.detection.segformer_landscape import SegformerLandscapeDetector
from core.detection.types import Raster


def _make_detector_with_stub(label_map: np.ndarray, id2label: dict[int, str]):
    """Build a SegformerLandscapeDetector whose forward pass returns logits
    matching the supplied (H, W) label_map (one-hot)."""
    h, w = label_map.shape
    n_classes = max(id2label.keys()) + 1
    logits = np.zeros((1, n_classes, h, w), dtype=np.float32)
    for c in range(n_classes):
        logits[0, c][label_map == c] = 10.0  # peak at the correct class

    fake_outputs = MagicMock()
    fake_outputs.logits = torch.from_numpy(logits)  # type: ignore[attr-defined]

    fake_model = MagicMock()
    fake_model.return_value = fake_outputs
    fake_model.config.id2label = id2label
    fake_model.to.return_value = fake_model

    fake_processor = MagicMock()
    fake_processor.return_value = {
        "pixel_values": torch.zeros(  # type: ignore[attr-defined]
            (1, 3, h, w), dtype=torch.float32  # type: ignore[attr-defined]
        )
    }

    detector = SegformerLandscapeDetector(min_pixels=1, max_dim=10_000)
    detector._model = fake_model  # type: ignore[assignment]
    detector._processor = fake_processor  # type: ignore[assignment]
    detector._id2name = id2label
    return detector


def _raster(h: int = 32, w: int = 32) -> Raster:
    return Raster(
        data=np.zeros((h, w, 3), dtype=np.uint8),
        transform=Affine(1, 0, 0, 0, -1, h),
        crs="EPSG:4326",
    )


def test_emits_landscape_class_for_known_label():
    label_map = np.zeros((32, 32), dtype=np.int32)
    label_map[10:20, 10:20] = 1  # class 1 = "building"
    id2label = {0: "wall", 1: "building"}

    detector = _make_detector_with_stub(label_map, id2label)
    out = detector.detect(_raster())

    assert len(out) == 1
    assert out[0].class_name == "building"


def test_skips_classes_not_in_landscape_map():
    label_map = np.full((32, 32), 0, dtype=np.int32)  # all "wall"
    id2label = {0: "wall"}

    detector = _make_detector_with_stub(label_map, id2label)
    out = detector.detect(_raster())
    assert out == []


def test_separates_disjoint_components_of_same_class():
    label_map = np.zeros((32, 32), dtype=np.int32)
    label_map[2:8, 2:8] = 1
    label_map[20:28, 20:28] = 1
    id2label = {0: "wall", 1: "building"}

    detector = _make_detector_with_stub(label_map, id2label)
    out = detector.detect(_raster())
    assert len(out) == 2
    assert all(d.class_name == "building" for d in out)


def test_drops_components_below_min_pixels():
    label_map = np.zeros((32, 32), dtype=np.int32)
    label_map[0, 0] = 1  # 1-pixel speck
    id2label = {0: "wall", 1: "building"}

    detector = SegformerLandscapeDetector(min_pixels=10, max_dim=10_000)
    h, w = label_map.shape
    n_classes = max(id2label.keys()) + 1
    logits = np.zeros((1, n_classes, h, w), dtype=np.float32)
    for c in range(n_classes):
        logits[0, c][label_map == c] = 10.0
    fake_outputs = MagicMock()
    fake_outputs.logits = torch.from_numpy(logits)  # type: ignore[attr-defined]
    fake_model = MagicMock(return_value=fake_outputs)
    fake_model.config.id2label = id2label
    fake_model.to.return_value = fake_model
    fake_processor = MagicMock(
        return_value={
            "pixel_values": torch.zeros(  # type: ignore[attr-defined]
                (1, 3, h, w), dtype=torch.float32  # type: ignore[attr-defined]
            )
        }
    )
    detector._model = fake_model  # type: ignore[assignment]
    detector._processor = fake_processor  # type: ignore[assignment]
    detector._id2name = id2label

    out = detector.detect(_raster())
    assert out == []


def test_maps_synonyms_to_canonical_names():
    """ADE20K 'sea' should be reported as 'water'."""
    label_map = np.zeros((32, 32), dtype=np.int32)
    label_map[10:20, 10:20] = 1
    id2label = {0: "wall", 1: "sea"}

    detector = _make_detector_with_stub(label_map, id2label)
    out = detector.detect(_raster())
    assert len(out) == 1
    assert out[0].class_name == "water"
