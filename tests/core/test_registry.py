import pytest

from core import ProcessSpec, build, register, registered_names


def test_built_in_processes_registered():
    names = registered_names()
    assert "msft-buildings" in names
    assert "yolov8-satellite-vehicle" in names


def test_yolo26_presets_registered():
    names = registered_names()
    for key in ("yolo26n-sahi", "yolo26m-sahi", "yolo26-obb", "yolo26-seg"):
        assert key in names


def test_unknown_process_raises():
    with pytest.raises(ValueError, match="Unknown process"):
        build(ProcessSpec(name="does-not-exist"))


def test_register_custom_process():
    class _Stub:
        def __init__(self, spec):
            self.spec = spec
            self.name = spec.name

        def run(self, raster):  # pragma: no cover - not exercised
            return []

    register("test-stub", _Stub)
    proc = build(ProcessSpec(name="test-stub"))
    assert proc.name == "test-stub"
