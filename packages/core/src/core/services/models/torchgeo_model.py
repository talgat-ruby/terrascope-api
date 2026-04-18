"""TorchGeo model wrapper for semantic segmentation (vegetation, roads, water)."""

import numpy as np
import torch
from numpy.typing import NDArray


class TorchGeoModel:
    """Wraps a torchgeo FCN semantic segmentation model.

    Produces per-pixel class probabilities for land cover classes:
    vegetation, road, water. Channel 0 is background.
    """

    CLASSES: list[str] = ["vegetation", "road", "water"]

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._model: torch.nn.Module | None = None

    def load(self, in_channels: int = 3) -> None:
        """Load the FCN segmentation model.

        Uses torchgeo's FCN which produces per-pixel class logits
        of shape (batch, classes, H, W). No pretrained weights exist
        for this 3-class task -- model requires fine-tuning.

        Args:
            in_channels: Number of input bands (default 3 for RGB).
        """
        from torchgeo.models import FCN

        num_classes = len(self.CLASSES) + 1  # +1 for background at index 0
        self._model = FCN(in_channels=in_channels, classes=num_classes)
        self._model.to(self.device)
        self._model.eval()

    def predict(self, tile_data: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
        """Run inference on a tile, returning per-class probability masks.

        Args:
            tile_data: Array of shape (bands, height, width), float32.

        Returns:
            Dict mapping class name to probability mask of shape (height, width),
            values in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            tensor = torch.from_numpy(tile_data).unsqueeze(0).to(self.device)
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        results: dict[str, NDArray[np.float32]] = {}
        n_output_channels = probs.shape[0]
        for i, class_name in enumerate(self.CLASSES):
            class_idx = i + 1  # skip background channel at index 0
            if class_idx < n_output_channels:
                results[class_name] = probs[class_idx].astype(np.float32)
            else:
                results[class_name] = np.zeros(
                    (tile_data.shape[1], tile_data.shape[2]), dtype=np.float32
                )

        return results
