import numpy as np
from PIL import Image
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection import render_overlay
from core.detection.types import Detection, Raster


def _raster(h: int = 100, w: int = 100) -> Raster:
    return Raster(
        data=np.full((h, w, 3), 128, dtype=np.uint8),
        transform=Affine(1, 0, 0, 0, -1, h),
        crs="EPSG:4326",
    )


def _det(pixel_bbox: tuple[int, int, int, int], class_name: str = "car") -> Detection:
    return Detection(
        id=0,
        class_name=class_name,
        confidence=0.91,
        bbox=(0, 0, 1, 1),
        pixel_bbox=pixel_bbox,
        centroid=box(0, 0, 1, 1).centroid,
    )


def test_render_overlay_writes_png(tmp_path):
    raster = _raster()
    out = render_overlay(raster, [_det((10, 10, 30, 30))], tmp_path / "out.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_overlay_dims_match_raster(tmp_path):
    raster = _raster(h=120, w=80)
    out = render_overlay(raster, [_det((5, 5, 20, 20))], tmp_path / "out.png")
    img = Image.open(out)
    assert img.size == (80, 120)


def test_render_overlay_handles_no_detections(tmp_path):
    raster = _raster()
    out = render_overlay(raster, [], tmp_path / "empty.png")
    assert out.exists()


def test_render_overlay_handles_label_clipping_at_top(tmp_path):
    """Detection touching y=0 gets its label inside the box, not above."""
    raster = _raster()
    det = _det((10, 0, 30, 30))  # box starts at the top
    out = render_overlay(raster, [det], tmp_path / "out.png")
    assert out.exists()


def test_render_overlay_distinct_colors_per_class(tmp_path):
    raster = _raster()
    dets = [_det((10, 10, 30, 30), "car"), _det((50, 50, 70, 70), "person")]
    out = render_overlay(raster, dets, tmp_path / "out.png")
    assert out.exists()
