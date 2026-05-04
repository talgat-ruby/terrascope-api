"""Per-class/per-model processes.

Each process is a self-contained pipeline keyed by `ProcessSpec.name`.
Built-ins register themselves on import via the side-effect modules
imported at the bottom of this file.
"""

from core2.processes.base import Process, ProcessSpec
from core2.processes.registry import build, register, registered_names

# Side-effect imports register built-in processes.
from core2.processes import msft_footprints  # noqa: F401
from core2.processes import osm_roads  # noqa: F401
from core2.processes import sam_road  # noqa: F401
from core2.processes import yolo_sahi  # noqa: F401

__all__ = [
    "Process",
    "ProcessSpec",
    "build",
    "register",
    "registered_names",
]
