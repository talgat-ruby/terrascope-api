"""PNG overlay renderer.

The headline deliverable: an annotated PNG of the source raster with each
detection drawn as a labeled rectangle. Pillow-only (no matplotlib), pixel
coordinates only (no reprojection), portable default font.
"""

from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.detection.types import Detection, Raster

_BOX_WIDTH = 2
_LABEL_PAD = 2


def _color_for(class_name: str) -> tuple[int, int, int]:
    """Stable, distinct color per class via hash -> HSV -> RGB."""
    h = int(hashlib.md5(class_name.encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def render_overlay(
    raster: Raster,
    detections: list[Detection],
    output_path: str | Path,
) -> Path:
    """Draw detection bboxes + labels onto the raster and save as PNG.

    `raster.data` is expected to be HWC uint8 RGB; `ImageryLoaderService`
    handles the percentile stretch upstream. Labels are placed above each
    box (or inside, near the top, when the box hugs the top edge).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(raster.data, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    for det in detections:
        c0, r0, c1, r1 = det.pixel_bbox
        color = _color_for(det.class_name)
        draw.rectangle((c0, r0, c1, r1), outline=color + (255,), width=_BOX_WIDTH)

        label = f"{det.class_name} {det.confidence:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Place label above the box; if it would clip off-image, place inside.
        ly1 = r0 - 1
        ly0 = ly1 - th - 2 * _LABEL_PAD
        if ly0 < 0:
            ly0 = r0 + 1
            ly1 = ly0 + th + 2 * _LABEL_PAD
        lx0 = c0
        lx1 = lx0 + tw + 2 * _LABEL_PAD

        draw.rectangle((lx0, ly0, lx1, ly1), fill=color + (220,))
        draw.text(
            (lx0 + _LABEL_PAD, ly0 + _LABEL_PAD),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    img.convert("RGB").save(output_path, format="PNG", optimize=True)
    return output_path
