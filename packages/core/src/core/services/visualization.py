"""VisualizationService -- render detection overlays on imagery."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from core.services.detector import RawDetection

# Color map per class
CLASS_COLORS: dict[str, str] = {
    "building": "#e74c3c",
    "road": "#3498db",
    "vegetation": "#2ecc71",
    "water": "#9b59b6",
}


class VisualizationService:
    """Renders detection overlays on satellite imagery."""

    def render_overlay(
        self,
        imagery_path: str | Path,
        detections: list[RawDetection],
        output_path: str | Path,
        figsize: tuple[int, int] = (16, 16),
        alpha: float = 0.4,
    ) -> Path:
        """Render detections overlaid on the source imagery.

        Args:
            imagery_path: Path to the GeoTIFF file.
            detections: Detections to overlay.
            output_path: Path to save the PNG output.
            figsize: Figure size in inches.
            alpha: Overlay transparency.

        Returns:
            Path to the saved PNG.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(str(imagery_path)) as dataset:
            # Read RGB bands (assume first 3 bands)
            bands = min(dataset.count, 3)
            data = dataset.read(list(range(1, bands + 1)))
            transform = dataset.transform

        # Normalize for display
        img = self._normalize_rgb(data)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        extent = (
            transform.c,
            transform.c + transform.a * img.shape[1],
            transform.f + transform.e * img.shape[0],
            transform.f,
        )
        ax.imshow(img, extent=extent, origin="upper")

        # Draw detections
        for det in detections:
            color = CLASS_COLORS.get(det.class_name, "#ffffff")
            self._draw_geometry(ax, det.geometry, color, alpha)

        # Legend
        handles = []
        labels_seen: set[str] = set()
        for det in detections:
            if det.class_name not in labels_seen:
                labels_seen.add(det.class_name)
                color = CLASS_COLORS.get(det.class_name, "#ffffff")
                handles.append(
                    Rectangle(
                        (0, 0), 1, 1, fc=color, alpha=alpha, label=det.class_name
                    )
                )
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=10)

        ax.set_title("Detection Overlay", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def _normalize_rgb(self, data: np.ndarray) -> np.ndarray:
        """Normalize raster bands to 0-1 for display.

        Handles both uint8 (0-255) and float imagery.
        """
        # data shape: (bands, height, width) -> (height, width, bands)
        if data.shape[0] == 1:
            # Single band: replicate to 3 channels
            img = np.stack([data[0]] * 3, axis=-1)
        else:
            img = np.transpose(data[:3], (1, 2, 0))

        img = img.astype(np.float32)
        if img.max() > 1.0:
            # Scale per-band to 0-1
            for i in range(img.shape[2]):
                band = img[:, :, i]
                bmin, bmax = (
                    np.percentile(band[band > 0], [2, 98])
                    if np.any(band > 0)
                    else (0, 1)
                )
                if bmax > bmin:
                    img[:, :, i] = np.clip((band - bmin) / (bmax - bmin), 0, 1)
                else:
                    img[:, :, i] = 0

        return img

    def _draw_geometry(
        self,
        ax: Axes,
        geometry: BaseGeometry,
        color: str,
        alpha: float,
    ) -> None:
        """Draw a shapely geometry on a matplotlib axes."""
        if isinstance(geometry, Polygon):
            self._draw_polygon(ax, geometry, color, alpha)
        elif isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms:
                self._draw_polygon(ax, poly, color, alpha)

    def _draw_polygon(
        self,
        ax: Axes,
        polygon: Polygon,
        color: str,
        alpha: float,
    ) -> None:
        """Draw a single polygon."""
        if polygon.is_empty:
            return
        exterior = list(polygon.exterior.coords)
        patch = MplPolygon(
            exterior, closed=True, fc=color, ec=color, alpha=alpha, linewidth=1
        )
        ax.add_patch(patch)
