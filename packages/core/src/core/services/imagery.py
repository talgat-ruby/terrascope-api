"""ImageryLoaderService -- load, validate, and clip GeoTIFF imagery."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import rasterio.mask
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely import ops
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from core.detection.types import Raster


class ImageryLoaderService:
    """Loads GeoTIFF files, validates georeferencing, and clips to AOI."""

    def load(self, path: str | Path) -> rasterio.DatasetReader:
        """Open a GeoTIFF and validate that it has a CRS.

        Args:
            path: Path to the GeoTIFF file.

        Returns:
            An open rasterio DatasetReader. Caller must close it.

        Raises:
            ValueError: If the file has no CRS.
            rasterio.errors.RasterioIOError: If the file cannot be opened.
        """
        dataset = rasterio.open(str(path))
        if dataset.crs is None:
            dataset.close()
            raise ValueError(f"GeoTIFF has no CRS: {path}")
        return dataset

    def clip_to_aoi(
        self,
        dataset: rasterio.DatasetReader,
        aoi: BaseGeometry,
        aoi_crs: str = "EPSG:4326",
    ) -> tuple[NDArray[np.float32], Affine, str]:
        """Clip raster to AOI bounds, reprojecting AOI if CRS differs.

        Args:
            dataset: An open rasterio DatasetReader.
            aoi: Shapely geometry defining the area of interest.
            aoi_crs: CRS of the AOI geometry.

        Returns:
            Tuple of (clipped data as float32, updated transform, CRS string).

        Raises:
            ValueError: If AOI does not intersect the raster.
        """
        dataset_crs = CRS.from_user_input(dataset.crs)
        aoi_crs_obj = CRS.from_user_input(aoi_crs)

        if dataset_crs != aoi_crs_obj:
            transformer = Transformer.from_crs(aoi_crs_obj, dataset_crs, always_xy=True)
            aoi = ops.transform(transformer.transform, aoi)

        dataset_bbox = shapely_box(*dataset.bounds)
        if not dataset_bbox.intersects(aoi):
            raise ValueError("AOI does not intersect raster")

        out_image, out_transform = rasterio.mask.mask(
            dataset, [aoi], crop=True, all_touched=True, filled=True, nodata=0
        )

        if out_image.size == 0:
            raise ValueError("AOI does not intersect raster")

        return out_image.astype(np.float32), out_transform, str(dataset.crs)

    def load_clipped(
        self,
        path: str | Path,
        aoi: BaseGeometry | None = None,
        aoi_crs: str = "EPSG:4326",
    ) -> "Raster":
        """Load a GeoTIFF, clip to AOI (or full extent), return a Raster.

        The returned `data` is HWC uint8 RGB (the first 3 bands, percentile-
        stretched). This is the format detectors and renderer both consume.
        """
        from core.detection.types import Raster

        dataset = self.load(path)
        try:
            if aoi is None:
                aoi_used = self.get_bounds_geometry(dataset)
                data, transform, crs = self.clip_to_aoi(
                    dataset, aoi_used, aoi_crs=str(dataset.crs)
                )
            else:
                aoi_used = aoi
                data, transform, crs = self.clip_to_aoi(dataset, aoi, aoi_crs)
        finally:
            dataset.close()

        rgb_uint8 = self._to_rgb_uint8(data)
        return Raster(data=rgb_uint8, transform=transform, crs=crs, aoi_geom=aoi_used)

    @staticmethod
    def _to_rgb_uint8(data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Convert (bands, H, W) float to (H, W, 3) uint8 with percentile stretch."""
        bands = min(data.shape[0], 3)
        rgb = data[:bands]
        if bands < 3:
            # Replicate single band into RGB
            rgb = np.repeat(rgb[:1], 3, axis=0)

        out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = rgb[i]
            valid = band[band > 0]
            if valid.size == 0:
                continue
            lo, hi = np.percentile(valid, [2, 98])
            if hi > lo:
                stretched = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
                out[..., i] = (stretched * 255).astype(np.uint8)
        return out

    def get_bounds_geometry(self, dataset: rasterio.DatasetReader) -> BaseGeometry:
        """Return the full extent of the raster as a Shapely polygon in the dataset's CRS."""
        return shapely_box(*dataset.bounds)

    def get_metadata(self, dataset: rasterio.DatasetReader) -> dict:
        """Extract raster metadata.

        Returns:
            Dict with band_count, crs, bounds, and resolution.
        """
        return {
            "band_count": dataset.count,
            "crs": str(dataset.crs),
            "bounds": {
                "left": dataset.bounds.left,
                "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right,
                "top": dataset.bounds.top,
            },
            "resolution": (dataset.res[0], dataset.res[1]),
        }
