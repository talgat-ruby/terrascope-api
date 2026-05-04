"""Tests for unet_roads pixel→world georeferencing.

We don't load the model (needs a torch + smp install + weights). We test
the pure transform plumbing in `_mask_to_detections` against a synthetic
mask, asserting that pixel coords on the mask map to the expected world
coords through the raster's affine transform.
"""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine

from core import ProcessSpec
from core.detection.types import Raster
from core.processes.unet_roads import UnetRoadsProcess


def _raster(h: int, w: int) -> Raster:
    # 1° per pixel, origin at (0, h) so row=0 sits on top.
    return Raster(
        data=np.zeros((h, w, 3), dtype=np.uint8),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, h),
        crs="EPSG:4326",
    )


def test_centroid_xy_matches_affine_transform() -> None:
    """A blob centered at pixel (col=20, row=10) on a 100×100 raster with
    Affine(1, 0, 0, 0, -1, 100) must georeference to world (x=20.5, y=89.5).
    """
    spec = ProcessSpec(name="unet-roads")
    proc = UnetRoadsProcess.from_spec(spec)

    h, w = 100, 100
    raster = _raster(h, w)

    # 10×10 square blob centered around (col=20, row=10).
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:15, 15:25] = 1

    proc.min_area_px = 10  # let the small blob through
    proc.simplify_px = 0.0  # disable simplification noise
    dets = proc._mask_to_detections(mask, raster)
    assert len(dets) == 1
    det = dets[0]

    # Centroid in world coords. With Affine(1, 0, 0, 0, -1, h):
    # x = col + 0.5; y = h - (row + 0.5) = 100 - 10.5 = 89.5.
    cx, cy = det.centroid.x, det.centroid.y
    assert abs(cx - 20.0) <= 0.5, f"x off: {cx}"
    assert abs(cy - 90.0) <= 0.5, f"y off: {cy}"

    # bbox should bracket the centroid in world coords.
    minx, miny, maxx, maxy = det.bbox
    assert minx < cx < maxx
    assert miny < cy < maxy
