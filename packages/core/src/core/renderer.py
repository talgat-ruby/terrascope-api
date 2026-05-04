"""PNG overlay renderer for cli.

Bolder, more saturated palette than core's renderer, with translucent box
fills (~30% opacity) so a viewer can see both the source pixels and the
detected items at once. Labels keep full opacity for legibility.

Pillow-only, pixel coordinates only — same surface as
`core.detection.render_overlay` so call sites can swap implementations.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rasterio.transform import rowcol
from rasterio.warp import transform_geom
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection, Raster

_BOX_WIDTH = 3
_LINE_WIDTH = 1  # thinner stroke for linear features (roads, etc.)
_LABEL_PAD = 3
_FILL_ALPHA = 77  # ~30% of 255


# Hand-picked, high-saturation palette. Indexed deterministically by class
# name so colors stay stable across runs and across processes.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 56, 96),    # crimson
    (0, 200, 255),    # cyan
    (255, 215, 0),    # gold
    (124, 252, 0),    # lawn green
    (255, 105, 30),   # vermilion
    (148, 0, 211),    # violet
    (0, 255, 127),    # spring green
    (255, 20, 147),   # magenta
    (30, 144, 255),   # dodger blue
    (255, 140, 0),    # dark orange
    (0, 255, 200),    # aqua
    (220, 20, 60),    # red
    (138, 43, 226),   # blue-violet
    (255, 255, 0),    # yellow
    (255, 69, 0),     # red-orange
    (50, 205, 50),    # lime
)


def _color_for(class_name: str) -> tuple[int, int, int]:
    """Pick a palette color deterministically from the class name."""
    h = int(hashlib.md5(class_name.encode()).hexdigest(), 16)
    return _PALETTE[h % len(_PALETTE)]


def _project_to_pixels(
    coords: list[tuple[float, float]], raster: Raster,
) -> list[tuple[float, float]]:
    """Convert (x, y) in WGS84 to (col, row) in raster pixel space."""
    if raster.crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        line = {"type": "LineString", "coordinates": coords}
        reprojected = transform_geom("EPSG:4326", raster.crs, line)
        coords = [(float(x), float(y)) for x, y in reprojected["coordinates"]]
    out: list[tuple[float, float]] = []
    for x, y in coords:
        r, c = rowcol(raster.transform, x, y)
        out.append((float(c), float(r)))
    return out


def _draw_geometry(
    draw: ImageDraw.ImageDraw,
    geom: BaseGeometry,
    raster: Raster,
    color: tuple[int, int, int],
    fill_alpha: int,
    box_width: int,
    line_width: int,
) -> None:
    """Draw a shapely geometry onto `draw` in pixel space."""
    if isinstance(geom, (MultiLineString, MultiPolygon)):
        for part in geom.geoms:
            _draw_geometry(
                draw, part, raster, color, fill_alpha, box_width, line_width,
            )
        return
    if isinstance(geom, LineString):
        pts = _project_to_pixels(list(geom.coords), raster)
        if len(pts) >= 2:
            draw.line(pts, fill=color + (255,), width=line_width)
        return
    if isinstance(geom, Polygon):
        # Use the minimum rotated rectangle (oriented bbox) instead of
        # the raw polygon outline — gives a cleaner, parallelogram-like
        # quad that conveys orientation without raster-noise jitter.
        # GEOS' rotating calipers logs a RuntimeWarning on perfectly
        # axis-aligned or collinear rings (common for polygons coming
        # out of rasterio.features.shapes); suppress it — we filter
        # NaN results below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obb = geom.minimum_rotated_rectangle
        if not isinstance(obb, Polygon) or obb.is_empty:
            return  # degenerate (line/point) — nothing to fill
        ext = _project_to_pixels(list(obb.exterior.coords), raster)
        if any(not math.isfinite(c) for pt in obb.exterior.coords for c in pt):
            return  # NaN-laden OBB; skip rather than crash PIL
        if len(ext) >= 3:
            draw.polygon(
                ext,
                fill=color + (fill_alpha,),
                outline=color + (255,),
            )
        return


def render_overlay(
    raster: Raster,
    detections: list[Detection],
    output_path: str | Path,
    *,
    fill_alpha: int = _FILL_ALPHA,
    box_width: int = _BOX_WIDTH,
    line_width: int = _LINE_WIDTH,
) -> Path:
    """Draw translucent filled bboxes + labels and save as PNG.

    Each detection is drawn twice on a separate RGBA layer:
    once as a translucent rectangle fill (`fill_alpha`, default ~30%),
    and once as an opaque outline. The layer is alpha-composited onto the
    source raster, then a second pass writes labels on top so they are
    never washed out by overlapping fills.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = Image.fromarray(raster.data, mode="RGB").convert("RGBA")
    fills = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fills, mode="RGBA")

    for det in detections:
        color = _color_for(det.class_name)
        if det.geometry is not None:
            _draw_geometry(
                fill_draw, det.geometry, raster, color,
                fill_alpha, box_width, line_width,
            )
        else:
            c0, r0, c1, r1 = det.pixel_bbox
            fill_draw.rectangle(
                (c0, r0, c1, r1),
                fill=color + (fill_alpha,),
                outline=color + (255,),
                width=box_width,
            )

    composed = Image.alpha_composite(base, fills)
    label_draw = ImageDraw.Draw(composed, mode="RGBA")
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    for det in detections:
        # Linear features (roads, etc.) don't get per-segment labels — they
        # would tile the image with redundant text. Polygonal/point detections
        # still get a label anchored at the bbox top-left.
        if isinstance(det.geometry, (LineString, MultiLineString)):
            continue

        c0, r0, _c1, _r1 = det.pixel_bbox
        color = _color_for(det.class_name)

        label = f"{det.class_name} {det.confidence:.2f}"
        tbox = label_draw.textbbox((0, 0), label, font=font)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]

        ly1 = r0 - 1
        ly0 = ly1 - th - 2 * _LABEL_PAD
        if ly0 < 0:
            ly0 = r0 + 1
            ly1 = ly0 + th + 2 * _LABEL_PAD
        lx0 = c0
        lx1 = lx0 + tw + 2 * _LABEL_PAD

        label_draw.rectangle((lx0, ly0, lx1, ly1), fill=color + (235,))
        label_draw.text(
            (lx0 + _LABEL_PAD, ly0 + _LABEL_PAD),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    composed.convert("RGB").save(output_path, format="PNG", optimize=True)
    return output_path


__all__ = ["render_overlay"]
