"""PostprocessorService -- merge tiles, NMS, filter, simplify detections."""

from dataclasses import dataclass, replace

from pyproj import Geod
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from core.services.detector import RawDetection


@dataclass
class PostprocessingConfig:
    """Configuration for the postprocessing pipeline."""

    iou_threshold: float = 0.5
    confidence_threshold: int = 50
    min_area_m2: float = 10.0
    max_area_m2: float = 1_000_000.0
    simplify_tolerance_m: float = 1.0


class PostprocessorService:
    """Merges tile detections, applies NMS, filters, and simplifies."""

    def merge_tile_detections(
        self, tile_detections: list[list[RawDetection]]
    ) -> list[RawDetection]:
        """Flatten detections from multiple tiles into a single list.

        Args:
            tile_detections: List of per-tile detection lists.

        Returns:
            Flat list of all detections.
        """
        merged: list[RawDetection] = []
        for detections in tile_detections:
            merged.extend(detections)
        return merged

    def apply_nms(
        self, detections: list[RawDetection], iou_threshold: float = 0.5
    ) -> list[RawDetection]:
        """Apply Non-Maximum Suppression using STRtree spatial index.

        For overlapping detections of the same class, keeps the one with
        higher confidence.

        Args:
            detections: List of detections to filter.
            iou_threshold: IoU threshold above which to suppress.

        Returns:
            Filtered list of detections.
        """
        if not detections:
            return []

        # Group by class
        by_class: dict[str, list[RawDetection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name, []).append(d)

        result: list[RawDetection] = []
        for class_detections in by_class.values():
            result.extend(self._nms_for_class(class_detections, iou_threshold))

        return result

    def _nms_for_class(
        self, detections: list[RawDetection], iou_threshold: float
    ) -> list[RawDetection]:
        """NMS within a single class."""
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        geometries = [d.geometry for d in sorted_dets]
        tree = STRtree(geometries)

        suppressed: set[int] = set()
        kept: list[RawDetection] = []

        for i, det in enumerate(sorted_dets):
            if i in suppressed:
                continue
            kept.append(det)

            # Find candidates that intersect this detection's bounds
            candidates = tree.query(det.geometry)
            for j in candidates:
                if j <= i or j in suppressed:
                    continue
                iou = self._compute_iou(det.geometry, sorted_dets[j].geometry)
                if iou >= iou_threshold:
                    suppressed.add(j)

        return kept

    def _compute_iou(self, geom_a: BaseGeometry, geom_b: BaseGeometry) -> float:
        """Compute Intersection over Union between two geometries."""
        if not geom_a.intersects(geom_b):
            return 0.0
        intersection = geom_a.intersection(geom_b).area
        union = geom_a.area + geom_b.area - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def filter_by_confidence(
        self, detections: list[RawDetection], threshold: int = 50
    ) -> list[RawDetection]:
        """Remove detections below the confidence threshold."""
        return [d for d in detections if d.confidence >= threshold]

    def filter_by_size(
        self,
        detections: list[RawDetection],
        min_area_m2: float = 10.0,
        max_area_m2: float = 1_000_000.0,
    ) -> list[RawDetection]:
        """Remove detections outside the area range (geodesic m2).

        Uses WGS84 ellipsoid for accurate area computation from
        geographic coordinates.
        """
        geod = Geod(ellps="WGS84")
        result: list[RawDetection] = []
        for d in detections:
            area_m2 = abs(geod.geometry_area_perimeter(d.geometry)[0])
            if area_m2 > 0 and min_area_m2 <= area_m2 <= max_area_m2:
                result.append(d)
        return result

    def filter_by_shape(self, detections: list[RawDetection]) -> list[RawDetection]:
        """Remove invalid or degenerate geometries."""
        result: list[RawDetection] = []
        for d in detections:
            if d.geometry.is_empty or not d.geometry.is_valid:
                continue
            # Skip very thin slivers (width/height ratio)
            minx, miny, maxx, maxy = d.geometry.bounds
            width = maxx - minx
            height = maxy - miny
            if width == 0 or height == 0:
                continue
            aspect = max(width, height) / min(width, height)
            if aspect > 100:
                continue
            result.append(d)
        return result

    def simplify_geometries(
        self, detections: list[RawDetection], tolerance_m: float = 1.0
    ) -> list[RawDetection]:
        """Simplify detection geometries using Douglas-Peucker.

        Args:
            detections: List of detections.
            tolerance_m: Simplification tolerance in meters. Converted to
                approximate degrees for geographic CRS (1m ~ 1/111320 degrees).

        Returns:
            Detections with simplified geometries.
        """
        tolerance_deg = tolerance_m / 111_320.0
        result: list[RawDetection] = []
        for d in detections:
            simplified = d.geometry.simplify(tolerance_deg, preserve_topology=True)
            if simplified.is_empty:
                continue
            result.append(replace(d, geometry=simplified))
        return result

    def clip_to_aoi(
        self, detections: list[RawDetection], aoi: BaseGeometry
    ) -> list[RawDetection]:
        """Clip detections to the AOI boundary.

        Detections fully outside are removed. Partially overlapping
        detections are clipped to the AOI.

        Args:
            detections: List of detections.
            aoi: Area of interest geometry.

        Returns:
            Clipped detections.
        """
        result: list[RawDetection] = []
        for d in detections:
            if not d.geometry.intersects(aoi):
                continue
            clipped = d.geometry.intersection(aoi)
            if clipped.is_empty:
                continue
            result.append(replace(d, geometry=clipped))
        return result

    def run(
        self,
        detections: list[RawDetection],
        config: PostprocessingConfig | None = None,
        aoi: BaseGeometry | None = None,
    ) -> tuple[list[RawDetection], dict]:
        """Run the full postprocessing pipeline.

        Args:
            detections: Raw detections from all tiles.
            config: Postprocessing configuration.
            aoi: Optional AOI to clip results to.

        Returns:
            Tuple of (processed detections, stats dict).
        """
        cfg = config or PostprocessingConfig()
        stats: dict = {"input_count": len(detections)}

        result = self.apply_nms(detections, cfg.iou_threshold)
        stats["after_nms"] = len(result)

        result = self.filter_by_confidence(result, cfg.confidence_threshold)
        stats["after_confidence_filter"] = len(result)

        result = self.filter_by_size(result, cfg.min_area_m2, cfg.max_area_m2)
        stats["after_size_filter"] = len(result)

        result = self.filter_by_shape(result)
        stats["after_shape_filter"] = len(result)

        result = self.simplify_geometries(result, cfg.simplify_tolerance_m)
        stats["after_simplify"] = len(result)

        if aoi is not None:
            result = self.clip_to_aoi(result, aoi)
            stats["after_clip"] = len(result)

        stats["output_count"] = len(result)
        return result, stats
