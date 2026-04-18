"""ML model wrappers for object detection in satellite imagery."""

from core.services.models.samgeo_model import SamGeoModel
from core.services.models.torchgeo_model import TorchGeoModel

__all__ = [
    "SamGeoModel",
    "TorchGeoModel",
]
