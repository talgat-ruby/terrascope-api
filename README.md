# Terrascope

Detect objects on geo-referenced satellite/aerial imagery and produce an
annotated PNG plus a GeoJSON of the detections.

Given a GeoTIFF, Terrascope:

1. Loads the raster and (optionally) clips to an AOI.
2. Runs one or more pluggable `Detector`s declared in the job config. Each
   detector can be scoped to a specific class allowlist with its own
   confidence threshold; results are merged with a `source_model` provenance
   tag. Defaults to YOLO + SAHI sliced inference so large rasters and
   edge-spanning objects work without manual tiling.
3. Renders an `overlay.png` with each detection drawn as a labeled bbox on
   the source image.
4. Writes a `detections.geojson` (point centroid + bbox WKT per feature) for
   GIS use.
5. Optionally computes per-zone indicators (count, density, bbox area).

Precision is explicitly **not** a goal — the design lets you compose
specialist models per class (e.g., one for vehicles, one for buildings)
without rewriting the pipeline.

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
                     types.py             Detection / Raster / Detector protocol
                     spec.py              DetectorSpec — per-detector job config
                     yolo_sahi.py         YOLO + SAHI detector
                     segformer_landscape.py  SegFormer-ADE20K land-cover
                     composite.py         Multi-detector merger w/ class allowlist
                     filter.py            Confidence + AOI + id renumber
                     renderer.py          Pillow PNG overlay
                     factory.py           Registry + build_from_specs()
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
  --output outputs/astana \
  --detectors '[{"name":"yolov8-obb-aerial","classes":["ship","large vehicle"]},
                {"name":"segformer-landscape","classes":["building","road"],"min_confidence":0.5}]'
```

Produces:

- `outputs/astana/overlay.png` — annotated source image
- `outputs/astana/detections.geojson` — point centroids + `bbox_wkt` attribute
- `outputs/astana/indicators/` — CSV + JSON per-zone stats (when an AOI is
  supplied or implied)

Flags:

- `--detectors '<json>'` (required) — JSON list of detector specs. Each spec
  has `name` (factory key), optional `classes` (allowlist), optional
  `min_confidence`, optional `kwargs`.
- `--aoi path/to/aoi.geojson` — clip to a polygon AOI
- `--use-temporal` — submit the job to a running Temporal worker instead of
  running locally

Registered detector names: `yolov8n-sahi`, `yolov8-obb-aerial`,
`yolov8-obb-dota-v2`, `yolov8-satellite-vehicle`, `segformer-landscape`,
`beit-ade`.

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
  -d '{
        "input_path": "/abs/path/to/raster.tif",
        "config": {
          "detectors": [
            {"name": "yolov8-obb-aerial", "classes": ["ship", "large vehicle"]},
            {"name": "segformer-landscape", "classes": ["building", "road"]}
          ]
        }
      }'
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

A job declares which detectors to run via a list of `DetectorSpec`s:

```python
@dataclass(frozen=True)
class DetectorSpec:
    name: str                              # factory key
    classes: tuple[str, ...] | None = None # allowlist; None = accept all
    min_confidence: float | None = None    # per-detector override
    kwargs: dict[str, Any] = field(default_factory=dict)
```

`build_from_specs(specs)` always wraps the leaves in a `CompositeDetector`
(single- and multi-spec jobs go through the same path), which:

1. Runs each child against the same raster.
2. Drops detections whose `class_name` isn't in the spec's allowlist.
3. Drops detections below the spec's `min_confidence`.
4. Stamps each detection with `source_model = child.name`.
5. Concatenates all surviving detections (final id renumbering happens
   in `filter_detections` after the global confidence + AOI filters).

Built-in detectors:

| Name                       | Backend                              | Classes                                                       |
|----------------------------|--------------------------------------|---------------------------------------------------------------|
| `yolov8n-sahi`             | Ultralytics YOLO + SAHI              | COCO 80 (generic objects)                                     |
| `yolov8-obb-aerial`        | YOLOv8s-OBB (DOTA v1)                | 15 aerial classes (ship, plane, vehicle, harbor, bridge, ...) |
| `yolov8-obb-dota-v2`       | YOLOv8x-OBB (DOTA v2)                | DOTA v2 incl. sports fields (tennis, basketball, soccer)      |
| `yolov8-satellite-vehicle` | `keremberke/yolov8m-satellite-...`   | car (HF-hosted finetune for satellite imagery)                |
| `segformer-landscape`      | SegFormer-ADE20K                     | building, road, grass, tree, water, earth, sand, mountain     |
| `beit-ade`                 | `microsoft/beit-large-finetuned-ade` | Same ADE20K classes; higher-capacity backbone                 |

Adding a new detector is one entry in `_BUILDERS` plus a class implementing
the `Detector` protocol. The orchestrator, worker, exporter, and DB schema
do not change.

Postprocessing is one function — `filter_detections(...)`: global
confidence floor, optional AOI centroid containment, sequential id
renumbering. NMS / stitching / size filters are absent: SAHI does the
merging during inference, per-detector confidence is handled in the spec.

## Configuration

Settings live in `core/config.py` (pydantic-settings). Common knobs:

| Setting               | Default        | Purpose                                         |
|-----------------------|----------------|-------------------------------------------------|
| `yolo_weights`        | `yolov8n.pt`   | Default Ultralytics checkpoint                  |
| `landscape_model`     | SegFormer-b0   | HF model id for SegformerLandscapeDetector      |
| `landscape_max_dim`   | `1024`         | Downsample raster long-side cap for segmenter   |
| `landscape_min_pixels`| `200`          | Drop CC regions smaller than this               |
| `min_confidence`      | `0.25`         | Global postprocess threshold (0–1)              |
| `device`              | autodetect     | `cuda` / `mps` / `cpu`                          |
| `output_dir`          | `output`       | Where job artifacts land                        |

Detectors are selected per-job via the `detectors` field in
`ProcessingJob.config` (or `--detectors` on the CLI), not via a setting.

Override via `.env` or environment variables.

## Output schema

`detections.geojson` is a `FeatureCollection` of `Point` features
(centroids). Each feature carries:

| Field          | Type    | Meaning                                              |
|----------------|---------|------------------------------------------------------|
| `id`           | int     | 0..N, assigned post-filter                           |
| `class_name`   | string  | Whatever label the detector emitted (e.g., `"car"`)  |
| `confidence`   | float   | Score in [0, 1]                                      |
| `source_model` | string  | Detector name that produced this detection           |
| `bbox_wkt`     | string  | WKT of the bbox polygon in EPSG:4326                 |

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
