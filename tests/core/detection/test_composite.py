import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection.composite import CompositeDetector
from core.detection.types import Detection, Raster


def _raster() -> Raster:
    return Raster(
        data=np.zeros((100, 100, 3), dtype=np.uint8),
        transform=Affine(1, 0, 0, 0, -1, 100),
        crs="EPSG:4326",
    )


def _det(id: int, class_name: str) -> Detection:
    bnds = (0.0, 0.0, 1.0, 1.0)
    return Detection(
        id=id,
        class_name=class_name,
        confidence=0.9,
        bbox=bnds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bnds).centroid,
    )


class _Stub:
    def __init__(self, dets: list[Detection], name: str = "stub") -> None:
        self.dets = dets
        self.name = name

    def detect(self, raster: Raster) -> list[Detection]:
        return list(self.dets)


def test_composite_concatenates_children():
    a = _Stub([_det(0, "car"), _det(1, "ship")])
    b = _Stub([_det(0, "building"), _det(1, "grass")])
    composite = CompositeDetector(children=[a, b])
    out = composite.detect(_raster())
    assert len(out) == 4
    assert {d.class_name for d in out} == {"car", "ship", "building", "grass"}


def test_composite_renumbers_ids_globally():
    a = _Stub([_det(0, "car"), _det(1, "ship")])
    b = _Stub([_det(0, "building")])
    composite = CompositeDetector(children=[a, b])
    out = composite.detect(_raster())
    assert [d.id for d in out] == [0, 1, 2]


def test_composite_empty_child():
    a = _Stub([])
    b = _Stub([_det(0, "building")])
    composite = CompositeDetector(children=[a, b])
    out = composite.detect(_raster())
    assert len(out) == 1
    assert out[0].class_name == "building"


def test_composite_requires_children():
    with pytest.raises(ValueError, match="at least one"):
        CompositeDetector(children=[])
