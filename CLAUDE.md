# CLAUDE.md

## Project Overview

Terrascope is a satellite imagery analysis tool that detects objects (buildings, roads, vehicles, etc.) in GeoTIFF imagery and exports the results as a GeoJSON layer plus a PNG overlay. Each detector is implemented as a self-contained `Process`; the orchestrator runs a list of processes against one raster and merges their output.

**Tech stack:** Python 3.14, PyTorch + SAMGeo + Ultralytics + SAHI, rasterio, shapely, pyproj, Typer CLI, uv workspaces monorepo. Linting with ruff, type-checking with pyright.

## Repository Structure

```
packages/
  core/              # Detection types, processes, orchestrator, IO, geo helpers
    src/core/
      detection/     # Detection / Raster dataclasses + filter helpers
      io/            # load_raster (rasterio) + export_geojson
      geo/           # CRS / bbox-to-pixel helpers shared by processes
      processes/     # One Process per (model, dataset) pair
      vendor/sam_road/  # Vendored upstream sam_road inference code
      orchestrator.py
      renderer.py    # PNG overlay
      config.py      # Env-driven cache + weights paths
  cli/               # Typer CLI tool (`terrascope`)
tests/               # pytest, mirrors packages/ structure
```

There is **no database, no ORM, no async runtime.** A previous iteration used SQLModel + PostGIS + Alembic + Temporal; that layer was removed in favor of the simpler process-only pipeline. Don't reintroduce it without an explicit ask.

## Setup

```bash
uv sync
cp .env.example .env  # optional — only env vars are TERRASCOPE_CACHE_DIR/WEIGHTS_DIR/LOG_LEVEL
```

## Common Commands

```bash
# Run a job
uv run terrascope --config packages/cli/examples/job.example.json

# List registered processes
uv run terrascope --list-processes

# Tests
uv run pytest

# Linting
uvx ruff check .
uvx ruff format .

# Type checking
uvx pyright
```

## Architecture

- **`Process` protocol** (`core.processes.base.Process`): each implementation owns its full pipeline — download external data, preprocess, run inference, post-process — and emits a `list[Detection]`. Built-in processes register themselves on import via `core.processes.registry.register`.
- **`ProcessSpec`** is the declarative job-side config: `name`, optional class allowlist, optional `min_confidence`, and a process-specific `kwargs` dict. CLI configs are lists of these.
- **`run_processes(procs, raster)`** runs each process in order, applies the spec's allowlist + min_confidence, stamps `source_model`, and renumbers ids 0..N.
- **`Raster`** is HWC uint8 data + a rasterio `Affine` + a CRS string + an optional WGS84 AOI. `Detection` carries both world-CRS `bbox`/`geometry` and pixel-frame `pixel_bbox` so downstream code never has to re-project.
- **CRS propagation** is explicit: `load_raster()` returns a `Raster` carrying its native CRS; processes convert WGS84 vector data into the raster's CRS via the shared helpers in `core.geo.raster_utils`.
- **Geodesic accuracy:** use `pyproj.Geod(ellps="WGS84")` for area/length on geographic coordinates — never raw `geom.area`. See `msft_footprints.py` and `sam_road.py` for examples.

## Code Style

- Type hints on every public function. Use built-in generics (`list`, `dict`, `tuple`) — not `typing.List`.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- No legacy compatibility code — Python 3.14 only.
- Don't write multi-line comment blocks or docstrings explaining what code does — only the why when non-obvious.
- Default to writing no comments. The bar is "would removing this confuse a future reader?"

## Caching

- OSM Overpass responses → `$TERRASCOPE_CACHE_DIR/osm_roads/`
- Microsoft Building Footprint shards → `$TERRASCOPE_CACHE_DIR/msft_buildings/`
- Model checkpoints (HF Hub, SAM) → `$TERRASCOPE_WEIGHTS_DIR` (default `$TERRASCOPE_CACHE_DIR/weights`)

All paths can also be overridden per-process via `ProcessSpec.kwargs.cache_dir` / `weights_dir`.

## Testing

- pytest, no async mode needed.
- Tests are pure: no database, no live network. OSM and HuggingFace are patched with `unittest.mock.patch`. SAM-Road tests exercise pure-numpy helpers only.
- Test files mirror source structure under `tests/`.
