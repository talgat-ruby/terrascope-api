# Terrascope

Detect objects on geo-referenced satellite/aerial imagery and produce an
annotated PNG plus a GeoJSON of the detections.

Given a GeoTIFF, Terrascope:

1. Loads the raster and (optionally) clips to an AOI.
2. Runs a pluggable `Detector` (default: pretrained Ultralytics YOLO with
   SAHI sliced inference, so large rasters and edge-spanning objects work
   without manual tiling).
3. Renders an `overlay.png` with each detection drawn as a labeled bbox on
   the source image.
4. Writes a `detections.geojson` (point centroid + bbox WKT per feature) for
   GIS use.
5. Optionally computes per-zone indicators (count, density, bbox area).

Precision is explicitly **not** a goal — the design lets you swap in a
better-suited model when accuracy matters. The default pipeline ships
something runnable on day zero with zero training.

## Tech stack

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) workspaces (monorepo)
- FastAPI + SQLModel + AsyncPG
- PostgreSQL + PostGIS
- Temporal workflows
- Ultralytics YOLO + [SAHI](https://github.com/obss/sahi) for sliced inference
- Pillow for the PNG overlay
- GeoPandas / Shapely / rasterio / pyproj
- Typer CLI

## Repository layout

```
packages/
  core/
    src/core/
      detection/   Pluggable detector module:
                     types.py        Detection / Raster / Detector protocol
                     yolo_sahi.py    Default detector (YOLO + SAHI)
                     filter.py       Confidence + AOI + id renumber
                     renderer.py     Pillow PNG overlay
                     factory.py      Registry-based build_detector()
      services/    Imagery loader, GeoJSON exporter, indicator calculator,
                   STAC client
      models/      SQLModel tables (Detection, ProcessingJob, Territory, ...)
      schemas/     Pydantic API schemas
      alembic/     DB migrations
  api/             FastAPI application
  worker/          Temporal workflow + activities
                   (load -> detect -> export -> indicators -> finalize)
  cli/             Typer CLI (process, stac, worker, db)
infra/compose/     Postgres, Temporal, Elasticsearch
inputs/            Sample GeoTIFFs
outputs/           Per-job artifacts
tests/             Mirrors packages/
```

## Architecture

```
Load GeoTIFF -> Detect (Detector + filter) -> Render PNG + Export GeoJSON
                                          -> Compute indicators
```

There is no separate tiling step. SAHI slices large rasters and merges
predictions internally; the orchestrator does not see tiles.

The Temporal flow is the same five activities (`load_imagery → detect →
export_results → compute_indicators → finalize_job`); the `detect` activity
also renders the PNG so pixel-space bboxes don't need to be persisted.

## Setup

Prerequisites: Python 3.14+, `uv`, Docker + Docker Compose.

```bash
uv sync
cp .env.example .env
docker compose -f infra/compose/compose.yml --env-file .env up -d
uv run alembic -c packages/core/src/core/alembic.ini upgrade head
```

## Quick start (local CLI)

```bash
uv run terrascope process \
  --input inputs/Astana_1.tif \
  --output outputs/astana
```

Produces:

- `outputs/astana/overlay.png` — annotated source image
- `outputs/astana/detections.geojson` — point centroids + `bbox_wkt` attribute
- `outputs/astana/indicators/` — CSV + JSON per-zone stats (when an AOI is
  supplied or implied)

Optional flags:

- `--aoi path/to/aoi.geojson` — clip to a polygon AOI
- `--detector <name>` — pick a registered detector (default `yolov8n-sahi`)
- `--use-temporal` — submit the job to a running Temporal worker instead of
  running locally

Other CLI subcommands:

```bash
# STAC catalog search
uv run terrascope stac search --bbox 10.0,49.0,11.0,50.0 --datetime 2024-01-01/2024-06-01

# Start the Temporal worker (alternative to `python -m worker.main`)
uv run terrascope worker

# Run alembic from the CLI
uv run terrascope db upgrade
```

## API + Temporal worker

Each in its own terminal:

```bash
uv run python -m api.main      # FastAPI on $API_PORT (default 30001)
uv run python -m worker.main   # Temporal worker
```

Submit a job:

```bash
curl -X POST http://localhost:30001/processing/start \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/abs/path/to/raster.tif"}'
```

Poll, then download:

```bash
curl http://localhost:30001/processing/<job_id>/status
curl -OJ "http://localhost:30001/results/<job_id>/download?format=png"
curl -OJ "http://localhost:30001/results/<job_id>/download?format=geojson"
```

API docs at <http://localhost:30001/docs>.

## How detection works

The detection layer is a single small protocol:

```python
class Detector(Protocol):
    name: str
    def detect(self, raster: Raster) -> list[Detection]: ...
```

The default implementation, `YoloSahiDetector`:

- Loads an Ultralytics YOLO checkpoint (default `yolov8n.pt`, configurable
  via `settings.yolo_weights`).
- For rasters ≤ 1024 px on the long side, runs a single forward pass.
- For larger rasters, uses SAHI's `get_sliced_prediction` (slice 640,
  overlap 0.2). SAHI handles tiling, per-slice inference, and merging
  across slice boundaries via Greedy-NMM.
- Maps each prediction to a `Detection`: pixel bbox, geographic bbox via the
  raster's affine, COCO class name, score in [0, 1].

Switching detectors:

```python
# packages/core/src/core/detection/factory.py
_BUILDERS = {
    "yolov8n-sahi": _build_yolo_sahi,
    # add your detector here, e.g.:
    # "rtdetr": _build_rtdetr,
    # "grounding-dino": _build_grounding_dino,
}
```

Adding a new detector is one entry in `_BUILDERS` plus a class that
implements the `Detector` protocol. No changes to the orchestrator, worker,
exporter, or DB schema.

Postprocessing is one function — `filter_detections(...)`: confidence
threshold, optional AOI centroid containment, sequential id renumbering. NMS
/ stitching / size filters are intentionally absent: SAHI does the merging
during inference.

## Configuration

Settings live in `core/config.py` (pydantic-settings). Common knobs:

| Setting          | Default        | Purpose                       |
|------------------|----------------|-------------------------------|
| `detector_name`  | `yolov8n-sahi` | Which `Detector` to build     |
| `yolo_weights`   | `yolov8n.pt`   | Ultralytics checkpoint path   |
| `min_confidence` | `0.25`         | Postprocess threshold (0–1)   |
| `device`         | autodetect     | `cuda` / `mps` / `cpu`        |
| `output_dir`     | `output`       | Where job artifacts land      |

Override via `.env` or environment variables.

## Output schema

`detections.geojson` is a `FeatureCollection` of `Point` features
(centroids). Each feature carries:

| Field        | Type    | Meaning                                              |
|--------------|---------|------------------------------------------------------|
| `id`         | int     | 0..N, assigned post-filter                           |
| `class_name` | string  | Whatever label the detector emitted (e.g., `"car"`)  |
| `confidence` | float   | Score in [0, 1]                                      |
| `bbox_wkt`   | string  | WKT of the bbox polygon in EPSG:4326                 |

The `detections` table mirrors this: composite PK `(job_id, id)`, geometry
columns for centroid (`POINT`) and `bbox` (`POLYGON`), all SRID 4326.

## Common commands

```bash
# Tests
uv run pytest

# Lint + format
uvx ruff check .
uvx ruff format .

# Type check
uvx pyright

# Migrations
uv run alembic -c packages/core/src/core/alembic.ini revision --autogenerate -m "description"
uv run alembic -c packages/core/src/core/alembic.ini upgrade head
```

## Local service ports

| Service       | Port  |
|---------------|-------|
| API           | 30001 |
| PostgreSQL    | 35432 |
| Temporal      | 37233 |
| Temporal UI   | 38080 |
| Elasticsearch | 39200 |

## Caveats

- The default YOLO checkpoint is COCO-pretrained. On nadir aerial imagery
  with small ground sample distance you'll get noisy labels (`tv`,
  `clock`, etc.) because the model was trained on a different distribution.
  For real use, swap in aerial-fine-tuned weights via `yolo_weights` or
  register a different detector via the factory.
- The `quality_metrics` table and `/quality` endpoint are leftovers from
  the previous design; nothing currently populates them.

## Documentation

- `docs/assignment.md` — original requirements
- `docs/plan.md` — phase-by-phase development log
- `~/.claude/plans/i-created-new-branch-cached-kite.md` — current rewrite plan
