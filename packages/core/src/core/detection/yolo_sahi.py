"""Ultralytics YOLO + SAHI sliced inference detector.

Uses SAHI (Slicing Aided Hyper Inference) for tiled prediction so large
rasters don't have to fit a single forward pass and so objects on tile
boundaries are merged via Greedy-NMM rather than fragmented.

Slice size is auto-picked: for rasters <= 1024 px on the long side we run a
single full-image inference (faster, no merging needed). Larger rasters are
sliced at 640 (YOLO's training resolution) with 0.2 overlap.
"""

from __future__ import annotations

from shapely.geometry import box

from core.detection.types import Detection, Raster


class YoloSahiDetector:
    """Pluggable Detector implementation: pretrained YOLO + SAHI slicing.

    `weights` accepts any Ultralytics-loadable checkpoint (`yolov8n.pt` for
    generic COCO objects, `yolov8s-obb.pt` for DOTA aerial classes, or any
    path to a fine-tuned `.pt` file).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        full_image_threshold: int = 1024,
        name: str = "yolov8n-sahi",
    ) -> None:
        self.name = name
        self.weights = weights
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.full_image_threshold = full_image_threshold
        self._model = None  # type: ignore[assignment]

    def _load(self) -> None:
        if self._model is not None:
            return
        from sahi import AutoDetectionModel  # type: ignore[import-untyped]

        self._model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.weights,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )

    def detect(self, raster: Raster) -> list[Detection]:
        self._load()
        long_side = max(raster.height, raster.width)

        if long_side <= self.full_image_threshold:
            from sahi.predict import get_prediction  # type: ignore[import-untyped]

            result = get_prediction(
                image=raster.data,
                detection_model=self._model,
            )
        else:
            from sahi.predict import get_sliced_prediction  # type: ignore[import-untyped]

            result = get_sliced_prediction(
                image=raster.data,
                detection_model=self._model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
            )

        return self._to_detections(result.object_prediction_list, raster)

    def _to_detections(self, predictions: list, raster: Raster) -> list[Detection]:
        detections: list[Detection] = []
        for i, pred in enumerate(predictions):
            bbox = pred.bbox  # SAHI BoundingBox: minx, miny, maxx, maxy in pixels
            c0, r0, c1, r1 = (
                int(bbox.minx),
                int(bbox.miny),
                int(bbox.maxx),
                int(bbox.maxy),
            )
            # Pixel rows grow south; map to geographic via affine.
            x0, y0 = raster.transform * (c0, r1)  # type: ignore[misc]
            x1, y1 = raster.transform * (c1, r0)  # type: ignore[misc]
            minx, maxx = sorted((x0, x1))
            miny, maxy = sorted((y0, y1))
            geo_bbox = (minx, miny, maxx, maxy)
            centroid = box(*geo_bbox).centroid

            detections.append(
                Detection(
                    id=i,
                    class_name=str(pred.category.name),
                    confidence=float(pred.score.value),
                    bbox=geo_bbox,
                    pixel_bbox=(c0, r0, c1, r1),
                    centroid=centroid,
                )
            )
        return detections
