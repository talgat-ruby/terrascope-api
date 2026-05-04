"""Per-class/per-model processes.

Each process is a self-contained pipeline keyed by `ProcessSpec.name`.
Built-ins register themselves on import via the side-effect modules
imported at the bottom of this file.
"""

from core.processes.base import Process, ProcessSpec
from core.processes.registry import build, register, registered_names

# Side-effect imports register built-in processes.
from core.processes import msft_footprints  # noqa: F401
from core.processes import osm_roads  # noqa: F401
from core.processes import sam_road  # noqa: F401
from core.processes import unet_roads  # noqa: F401
from core.processes import yolo_sahi  # noqa: F401

__all__ = [
    "Process",
    "ProcessSpec",
    "build",
    "register",
    "registered_names",
]
