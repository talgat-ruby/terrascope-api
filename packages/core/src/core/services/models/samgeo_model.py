"""SAMGeo model wrapper for instance segmentation (buildings)."""

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine


class SamGeoModel:
    """Wraps segment-geospatial (SAMGeo) for building instance segmentation.

    Produces a binary mask and per-instance probability map for buildings.
    """

    CLASSES: list[str] = ["building"]

    def __init__(self, device: str = "cpu", model_type: str = "vit_h") -> None:
        self.device = device
        self.model_type = model_type
        self._model = None  # type: ignore[assignment]

    def load(self) -> None:
        """Load the SAM model."""
        from samgeo import SamGeo

        self._model = SamGeo(
            model_type=self.model_type,
            device=self.device,
            automatic=True,
        )

    def predict(
        self,
        tile_data: NDArray[np.float32],
        transform: Affine,
        crs: str = "EPSG:4326",
    ) -> dict[str, NDArray[np.float32]]:
        """Run SAM inference on a tile, returning building probability mask.

        SAMGeo requires a GeoTIFF file as input, so we write the tile to a
        temporary file, run inference, and read back the mask.

        Args:
            tile_data: Array of shape (bands, height, width), float32.
            transform: Affine transform for the tile.

        Returns:
            Dict mapping "building" to a probability mask of shape (height, width),
            values in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        height, width = tile_data.shape[1], tile_data.shape[2]
        bands = min(tile_data.shape[0], 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "tile.tif"
            output_path = Path(tmpdir) / "mask.tif"

            # Write tile as GeoTIFF (SAMGeo needs file input)
            with rasterio.open(
                input_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=bands,
                dtype="float32",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(tile_data[:bands])

            self._model.generate(str(input_path), output=str(output_path))

            if output_path.exists():
                with rasterio.open(output_path) as src:
                    mask = src.read(1).astype(np.float32)
                    # Normalize to [0, 1]
                    if mask.max() > 1:
                        mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((height, width), dtype=np.float32)

        return {"building": mask}
