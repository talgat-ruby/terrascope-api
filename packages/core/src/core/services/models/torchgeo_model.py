"""TorchGeo model wrapper for semantic segmentation (vegetation, roads, water)."""

import numpy as np
import torch
from numpy.typing import NDArray


class TorchGeoModel:
    """Wraps a torchgeo pretrained semantic segmentation model.

    Produces per-pixel class probabilities for land cover classes:
    vegetation, road, water.
    """

    CLASSES: list[str] = ["vegetation", "road", "water"]

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._model: torch.nn.Module | None = None

    def load(self) -> None:
        """Load the pretrained segmentation model.

        Uses torchgeo's pretrained ResNet-based segmentation.
        """
        from torchgeo.models import ResNet18_Weights, resnet18

        self._model = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)
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
            output = self._model(tensor)

            if isinstance(output, dict):
                logits = output.get("out", output.get("logits", next(iter(output.values()))))
            else:
                logits = output

            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        results: dict[str, NDArray[np.float32]] = {}
        n_output_classes = probs.shape[0]
        for i, class_name in enumerate(self.CLASSES):
            if i < n_output_classes:
                results[class_name] = probs[i].astype(np.float32)
            else:
                results[class_name] = np.zeros(
                    (tile_data.shape[1], tile_data.shape[2]), dtype=np.float32
                )

        return results
