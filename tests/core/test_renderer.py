"""Tests for core's translucent overlay renderer."""

from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection.types import Detection, Raster

from core import render_overlay
from core.renderer import _color_for


def _raster(h: int = 100, w: int = 100) -> Raster:
    return Raster(
        data=np.full((h, w, 3), 80, dtype=np.uint8),  # mid-grey background
        transform=Affine(1, 0, 0, 0, -1, h),
        crs="EPSG:4326",
    )


def _det(
    class_name: str,
    pixel_bbox: tuple[int, int, int, int],
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=(0.0, 0.0, 1.0, 1.0),
        pixel_bbox=pixel_bbox,
        centroid=box(0, 0, 1, 1).centroid,
        source_model="stub",
    )


def test_render_overlay_writes_png(tmp_path: Path):
    out = tmp_path / "overlay.png"
    render_overlay(_raster(), [_det("car", (10, 10, 30, 30))], out)
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (100, 100)
        assert im.mode == "RGB"


def test_fill_blends_with_background(tmp_path: Path):
    """Pixels inside the bbox should differ from outside (translucent fill)
    but still keep some of the original grey (alpha < 1)."""
    out = tmp_path / "overlay.png"
    render_overlay(_raster(), [_det("car", (40, 40, 60, 60))], out)
    arr = np.array(Image.open(out))

    inside = arr[50, 50]
    outside = arr[5, 5]
    assert not np.array_equal(inside, outside), "fill should tint inside pixels"
    # Background was uniform grey 80; with ~30% fill it should still carry
    # a meaningful contribution from the original (not be the pure palette
    # color).
    assert inside.min() > 20, "alpha-composited inside should retain some background"


def test_color_stable_per_class():
    a = _color_for("car")
    b = _color_for("car")
    assert a == b
    assert _color_for("car") != _color_for("building")


def test_label_drawn_inside_when_box_at_top(tmp_path: Path):
    """Boxes hugging y=0 must place labels inside (not clipped above)."""
    out = tmp_path / "overlay.png"
    render_overlay(_raster(), [_det("ship", (5, 0, 25, 18))], out)
    assert out.exists()


def test_handles_empty_detections(tmp_path: Path):
    out = tmp_path / "overlay.png"
    render_overlay(_raster(), [], out)
    arr = np.array(Image.open(out))
    # No detections → image equals the source raster.
    assert (arr == 80).all()
