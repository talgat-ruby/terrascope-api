"""DetectorService -- ML inference on tiles with mask-to-vector conversion."""

from dataclasses import dataclass

import numpy as np
import rasterio.features
import scipy.ndimage
from numpy.typing import NDArray
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from core.models.detection import CLASS_REGISTRY
from core.models.tile import Tile
from core.services.models.samgeo_model import SamGeoModel
from core.services.models.torchgeo_model import TorchGeoModel


@dataclass
class RawDetection:
    """Intermediate detection before DB persistence."""

    class_name: str
    confidence: int
    geometry: BaseGeometry
    source: str


class DetectorService:
    """Runs ML inference on tiles and converts masks to vector detections."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._torchgeo: TorchGeoModel | None = None
        self._samgeo: SamGeoModel | None = None

    def load_models(self, in_channels: int = 3) -> None:
        """Load all ML models."""
        self._torchgeo = TorchGeoModel(device=self.device)
        self._torchgeo.load(in_channels=in_channels)

        self._samgeo = SamGeoModel(device=self.device)
        self._samgeo.load()

    def predict_tile(self, tile: Tile) -> list[RawDetection]:
        """Run inference on a tile, returning detected features as polygons.

        Uses TorchGeoModel for vegetation/road/water and SamGeoModel for buildings.
        Masks are converted to polygons via rasterio.features.shapes.

        Args:
            tile: A Tile with data and transform.

        Returns:
            List of RawDetection with class, confidence, and geometry.
        """
        detections: list[RawDetection] = []

        # TorchGeo: semantic segmentation for vegetation, road, water
        if self._torchgeo is not None:
            probs = self._torchgeo.predict(tile.data)
            for class_name, prob_mask in probs.items():
                if class_name in CLASS_REGISTRY:
                    masked = prob_mask * tile.valid_mask
                    polygons = self._mask_to_polygons(
                        masked, tile.transform, class_name, source="torchgeo"
                    )
                    detections.extend(polygons)

        # SAMGeo: instance segmentation for buildings
        if self._samgeo is not None:
            probs = self._samgeo.predict(tile.data, tile.transform, tile.crs)
            for class_name, prob_mask in probs.items():
                if class_name in CLASS_REGISTRY:
                    masked = prob_mask * tile.valid_mask
                    polygons = self._mask_to_polygons(
                        masked, tile.transform, class_name, source="samgeo"
                    )
                    detections.extend(polygons)

        return detections

    def predict_tile_with_masks(
        self,
        tile: Tile,
        class_masks: dict[str, NDArray[np.float32]],
        source: str = "mock",
    ) -> list[RawDetection]:
        """Convert pre-computed probability masks to detections.

        Useful for testing or when masks come from external sources.

        Args:
            tile: A Tile with valid_mask and transform.
            class_masks: Dict mapping class name to probability mask.
            source: Source identifier for the detections.

        Returns:
            List of RawDetection.
        """
        detections: list[RawDetection] = []
        for class_name, prob_mask in class_masks.items():
            if class_name in CLASS_REGISTRY:
                masked = prob_mask * tile.valid_mask
                polygons = self._mask_to_polygons(
                    masked, tile.transform, class_name, source=source
                )
                detections.extend(polygons)
        return detections

    def _mask_to_polygons(
        self,
        prob_mask: NDArray[np.float32],
        transform: Affine,
        class_name: str,
        source: str,
        threshold: float = 0.5,
    ) -> list[RawDetection]:
        """Convert a probability mask to polygon detections.

        Args:
            prob_mask: Probability mask of shape (height, width), values in [0, 1].
            transform: Affine transform mapping pixels to geographic coords.
            class_name: Detection class name.
            source: Model source identifier.
            threshold: Probability threshold for binary classification.

        Returns:
            List of RawDetection with confidence and geometry.
        """
        binary_mask = (prob_mask >= threshold).astype(np.uint8)

        if not binary_mask.any():
            return []

        # Fill in small holes and smooth edges before vectorization
        binary_mask = scipy.ndimage.binary_closing(
            binary_mask, structure=np.ones((3, 3))
        ).astype(np.uint8)

        detections: list[RawDetection] = []
        for geom_dict, value in rasterio.features.shapes(
            binary_mask, transform=transform
        ):
            if value == 0:
                continue

            polygon = shape(geom_dict)
            if not polygon.is_valid or polygon.is_empty:
                continue
            if polygon.geom_type not in ("Polygon", "MultiPolygon"):
                continue

            confidence = self._assign_confidence(polygon, prob_mask, transform)
            detections.append(
                RawDetection(
                    class_name=class_name,
                    confidence=confidence,
                    geometry=polygon,
                    source=source,
                )
            )

        return detections

    def _assign_confidence(
        self,
        polygon: BaseGeometry,
        prob_mask: NDArray[np.float32],
        transform: Affine,
    ) -> int:
        """Compute confidence score as mean probability within the polygon.

        Args:
            polygon: Detection polygon in geographic coordinates.
            prob_mask: Probability mask.
            transform: Affine transform.

        Returns:
            Confidence score as integer 0-100.
        """
        # Rasterize the polygon back to get pixel indices
        minx, miny, maxx, maxy = polygon.bounds
        inv_transform = ~transform
        col_min, row_min = inv_transform * (minx, maxy)  # type: ignore[misc]
        col_max, row_max = inv_transform * (maxx, miny)  # type: ignore[misc]

        r0 = max(0, int(row_min))
        r1 = min(prob_mask.shape[0], int(row_max) + 1)
        c0 = max(0, int(col_min))
        c1 = min(prob_mask.shape[1], int(col_max) + 1)

        if r0 >= r1 or c0 >= c1:
            return 50

        region = prob_mask[r0:r1, c0:c1]
        mean_prob = float(np.mean(region[region > 0])) if np.any(region > 0) else 0.5

        return max(0, min(100, int(mean_prob * 100)))
