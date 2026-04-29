"""Landscape (land-cover) detector via HuggingFace SegFormer + ADE20K.

The default model `nvidia/segformer-b0-finetuned-ade-512-512` is trained on
ADE20K, which contains 150 classes including the landscape labels users
typically ask for: building, road, grass, water, tree, sea, lake, river,
field, sand, sidewalk, path, earth/ground, mountain.

We run a single full-image inference (downsampling first if the raster is
larger than `landscape_max_dim` on the long side -- ADE20K models were
trained at modest resolutions and chunking adds little for region classes).
The argmax label map is then converted to per-class connected components,
each emitted as a `Detection` with a bbox covering the component.

Trade-offs vs. an aerial-trained segmenter (OEM, DeepGlobe):
- Pro: weights are first-party HF, load via standard `transformers` API,
  no custom training code.
- Con: ADE20K is street-level / oblique, not nadir. Predictions on truly
  top-down imagery are sometimes noisy, but the user's stated tolerance
  ("precision is not important") makes this acceptable for v1.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection.types import Detection, Raster

# ADE20K classes worth surfacing on aerial/landscape imagery. Maps the raw
# ADE20K label string to the simpler name we expose in the output.
_ADE_LANDSCAPE_LABELS: dict[str, str] = {
    "building": "building",
    "house": "building",
    "skyscraper": "building",
    "hovel": "building",
    "tower": "building",
    "road": "road",
    "sidewalk": "road",
    "path": "road",
    "runway": "road",
    "grass": "grass",
    "field": "grass",
    "tree": "tree",
    "plant": "tree",
    "palm": "tree",
    "water": "water",
    "river": "water",
    "lake": "water",
    "sea": "water",
    "pool": "water",
    "fountain": "water",
    "earth": "earth",
    "ground": "earth",
    "sand": "sand",
    "mountain": "mountain",
    "rock": "mountain",
}


class SegformerLandscapeDetector:
    """Pluggable Detector implementation: SegFormer-ADE20K -> bbox per region."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cpu",
        max_dim: int = 1024,
        min_pixels: int = 200,
        confidence_threshold: float = 0.0,
        name: str = "segformer-landscape",
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.device = device
        self.max_dim = max_dim
        self.min_pixels = min_pixels
        self.confidence_threshold = confidence_threshold
        self._model = None  # type: ignore[assignment]
        self._processor = None  # type: ignore[assignment]
        self._id2name: dict[int, str] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import (  # type: ignore[import-untyped]
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
        )

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._id2name = self._model.config.id2label

    def detect(self, raster: Raster) -> list[Detection]:
        import torch

        self._load()

        # Downsample so SegFormer's 512x512 training res matches roughly.
        h, w = raster.height, raster.width
        long_side = max(h, w)
        scale = max(1, long_side // self.max_dim) if long_side > self.max_dim else 1
        if scale > 1:
            ds = raster.data[::scale, ::scale]
        else:
            ds = raster.data

        assert self._processor is not None and self._model is not None
        inputs = self._processor(images=ds, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            logits = outputs.logits  # (1, num_classes, H', W')

        # Upsample to the downsampled image's spatial size, then argmax.
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(ds.shape[0], ds.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        label_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        # Per-pixel softmax peak as a confidence proxy.
        probs = torch.softmax(upsampled, dim=1).squeeze(0).cpu().numpy()  # type: ignore[attr-defined]

        return self._regions_to_detections(label_map, probs, raster.transform, scale)

    def _regions_to_detections(
        self,
        label_map: NDArray[np.int32],
        probs: NDArray[np.float32],
        transform: Affine,
        scale: int,
    ) -> list[Detection]:
        detections: list[Detection] = []
        counter = 0
        unique_ids = np.unique(label_map).tolist()

        for class_id in unique_ids:
            ade_name = (
                self._id2name.get(int(class_id), "").strip().lower().split(",")[0]
            )
            mapped = _ADE_LANDSCAPE_LABELS.get(ade_name)
            if mapped is None:
                continue

            mask = label_map == class_id
            if not mask.any():
                continue

            # Light morphological closing to merge speckle.
            mask = scipy.ndimage.binary_closing(mask, structure=np.ones((3, 3)))

            labeled_components, n_components = scipy.ndimage.label(mask)  # type: ignore[arg-type]
            if n_components == 0:
                continue

            slices = scipy.ndimage.find_objects(labeled_components)
            for comp_id, sl in enumerate(slices, start=1):
                if sl is None:
                    continue
                row_slice, col_slice = sl  # type: ignore[misc]
                comp_mask = labeled_components[row_slice, col_slice] == comp_id
                n_pix = int(comp_mask.sum())
                if n_pix < self.min_pixels:
                    continue
                conf = float(
                    probs[int(class_id), row_slice, col_slice][comp_mask].mean()
                )
                if conf < self.confidence_threshold:
                    continue

                # Pixel bbox in downsampled space; multiply by scale for source raster.
                c0_ds = int(col_slice.start)
                r0_ds = int(row_slice.start)
                c1_ds = int(col_slice.stop)
                r1_ds = int(row_slice.stop)
                c0, r0 = c0_ds * scale, r0_ds * scale
                c1, r1 = c1_ds * scale, r1_ds * scale

                x0, y0 = transform * (c0, r1)  # type: ignore[misc]
                x1, y1 = transform * (c1, r0)  # type: ignore[misc]
                minx, maxx = sorted((x0, x1))
                miny, maxy = sorted((y0, y1))
                geo_bbox = (minx, miny, maxx, maxy)
                centroid = box(*geo_bbox).centroid

                detections.append(
                    Detection(
                        id=counter,
                        class_name=mapped,
                        confidence=conf,
                        bbox=geo_bbox,
                        pixel_bbox=(c0, r0, c1, r1),
                        centroid=centroid,
                    )
                )
                counter += 1

        return detections
