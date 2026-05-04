"""Terrascope core — per-process pipeline (experimental).

Each `Process` owns its full pipeline (download, preprocess, infer,
postprocess) and emits canonical `Detection` objects (reused from `core`).
The orchestrator runs a list of processes and merges their output into a
single GeoJSON.

Public surface:

- `ProcessSpec`        declarative job-side config for one process.
- `Process` protocol   `run(raster) -> list[Detection]`.
- `register` / `build` factory keyed by `ProcessSpec.name`.
- `run_processes`      orchestrator: run + merge + sequential id renumber.
"""

from core.io import compute_indicators, export_geojson, load_raster, write_indicators
from core.orchestrator import run_processes
from core.processes.base import Process, ProcessSpec
from core.processes.registry import build, register, registered_names
from core.renderer import render_overlay

# Trigger side-effect registration of all built-in processes.
# Without this import the registry stays empty and build() always fails.
import core.processes  # noqa: F401

__all__ = [
    "Process",
    "ProcessSpec",
    "build",
    "compute_indicators",
    "export_geojson",
    "load_raster",
    "register",
    "registered_names",
    "render_overlay",
    "run_processes",
    "write_indicators",
]
