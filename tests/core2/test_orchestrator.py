import numpy as np
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection.types import Detection, Raster

from core2 import ProcessSpec, run_processes


def _raster() -> Raster:
    return Raster(
        data=np.zeros((50, 50, 3), dtype=np.uint8),
        transform=Affine(1, 0, 0, 0, -1, 50),
        crs="EPSG:4326",
    )


def _det(class_name: str, confidence: float, source_model: str = "stub") -> Detection:
    bnds = (0.0, 0.0, 1.0, 1.0)
    return Detection(
        id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=bnds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bnds).centroid,
        source_model=source_model,
    )


class _StubProcess:
    def __init__(self, spec: ProcessSpec, dets: list[Detection]) -> None:
        self.spec = spec
        self.name = spec.name
        self._dets = dets

    def run(self, raster: Raster) -> list[Detection]:
        return list(self._dets)


def test_concatenates_processes_and_renumbers():
    a = _StubProcess(
        ProcessSpec(name="a"), [_det("car", 0.9), _det("ship", 0.8)]
    )
    b = _StubProcess(ProcessSpec(name="b"), [_det("building", 0.95)])
    out = run_processes([a, b], _raster())
    assert [d.id for d in out] == [0, 1, 2]
    assert [d.class_name for d in out] == ["car", "ship", "building"]


def test_class_allowlist_filters():
    a = _StubProcess(
        ProcessSpec(name="a", classes=("ship",)),
        [_det("car", 0.9), _det("ship", 0.8)],
    )
    out = run_processes([a], _raster())
    assert [d.class_name for d in out] == ["ship"]


def test_per_spec_min_confidence_drops_below():
    a = _StubProcess(
        ProcessSpec(name="a", min_confidence=0.5),
        [_det("car", 0.2), _det("ship", 0.8)],
    )
    out = run_processes([a], _raster())
    assert [d.class_name for d in out] == ["ship"]


def test_stamps_source_model_from_process_name():
    a = _StubProcess(
        ProcessSpec(name="a"),
        [_det("car", 0.9, source_model="wrong")],
    )
    out = run_processes([a], _raster())
    assert out[0].source_model == "a"


def test_empty_processes_raises():
    import pytest
    with pytest.raises(ValueError, match="at least one"):
        run_processes([], _raster())
