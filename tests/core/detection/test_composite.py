import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection.composite import CompositeDetector
from core.detection.spec import DetectorSpec
from core.detection.types import Detection, Raster


def _raster() -> Raster:
    return Raster(
        data=np.zeros((100, 100, 3), dtype=np.uint8),
        transform=Affine(1, 0, 0, 0, -1, 100),
        crs="EPSG:4326",
    )


def _det(
    id: int, class_name: str, source_model: str = "stub", confidence: float = 0.9
) -> Detection:
    bnds = (0.0, 0.0, 1.0, 1.0)
    return Detection(
        id=id,
        class_name=class_name,
        confidence=confidence,
        bbox=bnds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bnds).centroid,
        source_model=source_model,
    )


class _Stub:
    def __init__(self, dets: list[Detection], name: str = "stub") -> None:
        self.dets = dets
        self.name = name

    def detect(self, raster: Raster) -> list[Detection]:
        return list(self.dets)


def test_composite_concatenates_children():
    a = _Stub([_det(0, "car", "a"), _det(1, "ship", "a")], name="a")
    b = _Stub([_det(0, "building", "b"), _det(1, "grass", "b")], name="b")
    composite = CompositeDetector(
        pairs=[(a, DetectorSpec(name="a")), (b, DetectorSpec(name="b"))]
    )
    out = composite.detect(_raster())
    assert len(out) == 4
    assert {d.class_name for d in out} == {"car", "ship", "building", "grass"}


def test_composite_renumbers_ids_globally():
    a = _Stub([_det(0, "car", "a"), _det(1, "ship", "a")], name="a")
    b = _Stub([_det(0, "building", "b")], name="b")
    composite = CompositeDetector(
        pairs=[(a, DetectorSpec(name="a")), (b, DetectorSpec(name="b"))]
    )
    out = composite.detect(_raster())
    assert [d.id for d in out] == [0, 1, 2]


def test_composite_class_allowlist():
    a = _Stub([_det(0, "car", "a"), _det(1, "ship", "a")], name="a")
    b = _Stub([_det(0, "building", "b"), _det(1, "grass", "b")], name="b")
    composite = CompositeDetector(
        pairs=[
            (a, DetectorSpec(name="a", classes=("ship",))),
            (b, DetectorSpec(name="b", classes=("building",))),
        ]
    )
    out = composite.detect(_raster())
    assert {d.class_name for d in out} == {"ship", "building"}


def test_composite_per_spec_min_confidence():
    a = _Stub(
        [_det(0, "car", "a", confidence=0.2), _det(1, "ship", "a", confidence=0.9)],
        name="a",
    )
    composite = CompositeDetector(
        pairs=[(a, DetectorSpec(name="a", min_confidence=0.5))]
    )
    out = composite.detect(_raster())
    assert len(out) == 1
    assert out[0].class_name == "ship"


def test_composite_stamps_source_model():
    a = _Stub([_det(0, "car", source_model="wrong")], name="a")
    composite = CompositeDetector(pairs=[(a, DetectorSpec(name="a"))])
    out = composite.detect(_raster())
    assert out[0].source_model == "a"


def test_composite_requires_pairs():
    with pytest.raises(ValueError, match="at least one"):
        CompositeDetector(pairs=[])
