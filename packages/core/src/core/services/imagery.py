"""ImageryLoaderService -- load, validate, and clip GeoTIFF imagery."""

from pathlib import Path

import numpy as np
import rasterio
import rasterio.mask
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely import ops
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry


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
