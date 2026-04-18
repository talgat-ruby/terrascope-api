"""TilerService -- split raster data into overlapping tiles for ML inference."""

from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine

from core.models.tile import Tile


class TilerService:
    """Splits raster data into fixed-size overlapping tiles."""

    def __init__(self, tile_size: int = 512, overlap: int = 64) -> None:
        self.tile_size = tile_size
        self.overlap = overlap

    def generate_tiles(
        self,
        data: NDArray[np.float32],
        transform: Affine,
        tile_size: int | None = None,
        overlap: int | None = None,
    ) -> Generator[Tile, None, None]:
        """Split raster into overlapping tiles, padding edges with zeros.

        Args:
            data: Raster array of shape (bands, height, width).
            transform: Affine transform for the raster.
            tile_size: Override instance tile_size.
            overlap: Override instance overlap.

        Yields:
            Tile dataclass instances with padded data and valid_mask.
        """
        ts = tile_size if tile_size is not None else self.tile_size
        ov = overlap if overlap is not None else self.overlap
        stride = ts - ov

        bands, height, width = data.shape

        for row_idx, row_off in enumerate(range(0, height, stride)):
            for col_idx, col_off in enumerate(range(0, width, stride)):
                actual_h = min(ts, height - row_off)
                actual_w = min(ts, width - col_off)

                tile_data = np.zeros((bands, ts, ts), dtype=np.float32)
                tile_data[:, :actual_h, :actual_w] = data[
                    :, row_off : row_off + actual_h, col_off : col_off + actual_w
                ]

                valid_mask = np.zeros((ts, ts), dtype=np.bool_)
                valid_mask[:actual_h, :actual_w] = True

                tile_transform = transform * Affine.translation(col_off, row_off)

                yield Tile(
                    index=(row_idx, col_idx),
                    pixel_window=(col_off, row_off, actual_w, actual_h),
                    transform=tile_transform,
                    data=tile_data,
                    valid_mask=valid_mask,
                )

    def tile_bounds(self, tile: Tile) -> tuple[float, float, float, float]:
        """Return geographic bbox (west, south, east, north) from tile.

        Args:
            tile: A Tile instance with transform and data.

        Returns:
            Tuple of (west, south, east, north).
        """
        _, height, width = tile.data.shape
        west, north = tile.transform * (0, 0)  # type: ignore[misc]
        east, south = tile.transform * (width, height)  # type: ignore[misc]
        return (west, south, east, north)
