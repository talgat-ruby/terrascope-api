from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine


@dataclass
class Tile:
    index: tuple[int, int]
    pixel_window: tuple[int, int, int, int]  # (col_off, row_off, width, height)
    transform: Affine
    data: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
