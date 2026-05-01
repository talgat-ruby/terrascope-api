# Terrascope — Architecture

## Goal

Take a geo-referenced raster (GeoTIFF), run one or more pluggable
detectors against it, and emit:

- `detections.geojson` — per-detection bbox polygons + metadata
- `overlay.png` — annotated source image
- `indicators/` — per-zone counts/density/area when an AOI is supplied

The system is multi-model first-class: a job declares a list of
`DetectorSpec`s, each scoped to a class allowlist with its own confidence
threshold. Outputs from every detector are merged with a `source_model`
provenance tag and persisted together.

## High-level pipeline

```
            ┌─────────────────────────────────────────────────────────┐
GeoTIFF ──▶ │  load_imagery   detect   export   indicators   finalize │ ──▶ outputs/
            └─────────────────────────────────────────────────────────┘
                                ▲
                                │
                       DetectorSpec[]  (from job.config.detectors)
```

Five sequential Temporal activities, idempotent via a checkpoint stored on
`ProcessingJob.checkpoint_data`. Each activity reads what it needs from the
prior step's checkpoint slot.

## Code layout

```
packages/
  core/      shared domain — models, schemas, services, detection module
  api/       FastAPI: POST /processing/start, GET status, GET results
  worker/    Temporal workflow + activities
  cli/       Typer CLI (`terrascope process …`) — runs locally or submits
             a job to the worker
infra/       Docker Compose (Postgres+PostGIS, Temporal, Elasticsearch)
scripts/     Sample bash invocations
docs/        This document, plan, class analysis
```

## Detection layer

The detection contract is one Protocol:

```python
class Detector(Protocol):
    name: str
    def detect(self, raster: Raster) -> list[Detection]: ...
```

Implementations live in `packages/core/src/core/detection/` and never leak
model-specific structures upward — only `Detection` objects.

### DetectorSpec — declarative job config

```python
@dataclass(frozen=True)
class DetectorSpec:
    name: str                              # factory key
    classes: tuple[str, ...] | None = None # allowlist; None = accept all
    min_confidence: float | None = None    # per-detector override
    kwargs: dict[str, Any] = field(default_factory=dict)
```

A job's config carries `detectors: list[DetectorSpec dict]`. The detection
activity parses these via `DetectorSpec.list_from_config()` and hands them
to `build_from_specs()`.

### Factory (`detection/factory.py`)

`_BUILDERS: dict[str, Callable[..., Detector]]` — one entry per registered
preset. Adding a detector = one entry plus a class implementing the
`Detector` Protocol. Currently registered:

| Key                        | Backend / weights                                      |
|----------------------------|--------------------------------------------------------|
| `yolov8n-sahi`             | `yolov8n.pt` (COCO 80 generic)                         |
| `yolov8-obb-aerial`        | `yolov8s-obb.pt` (DOTA v1 — 15 aerial)                 |
| `yolov8-obb-dota-v2`       | `yolov8x-obb.pt` (DOTA v2 — sports + bridge + cars)    |
| `yolov8-satellite-vehicle` | `keremberke/yolov8m-satellite-vehicle-detection` (HF)  |
| `segformer-landscape`      | `nvidia/segformer-b0-finetuned-ade-512-512`            |
| `beit-ade`                 | `microsoft/beit-large-finetuned-ade-640-640`           |
| `aerial-road-segmenter`    | user-supplied HF checkpoint trained on nadir roads     |

`build_from_specs(specs)`:
- 1 spec, no class/conf overrides → returns the leaf detector directly.
- otherwise → wraps leaves in a `CompositeDetector(pairs=[(leaf, spec), …])`.

### CompositeDetector

```python
for child, spec in self.pairs:
    for det in child.detect(raster):
        if spec.classes is not None and det.class_name not in spec.classes:
            continue
        if spec.min_confidence is not None and det.confidence < spec.min_confidence:
            continue
        det = replace(det, source_model=child.name)
        merged.append(det)
return [replace(d, id=i) for i, d in enumerate(merged)]
```

No cross-model NMS — class-level deduplication is left to the model that
owns the class via the allowlist.

### Backends

- **YoloSahiDetector** — Ultralytics + SAHI. Auto-picks single-pass
  inference for rasters ≤1024 px on the long side; otherwise SAHI sliced
  prediction (slice 640, overlap 0.2) with Greedy-NMM merge.
- **SegformerLandscapeDetector** — HuggingFace
  `AutoModelForSemanticSegmentation`. Downsamples to `max_dim` (default
  1024), argmax → connected components → bbox per component. Reused for
  both `segformer-landscape` and `beit-ade` (same ADE20K label space, just
  different `model_name`).

### Postprocessing

Two stages, both intentionally minimal:

1. **CompositeDetector**: per-spec class allowlist + per-spec confidence
   floor + provenance stamping.
2. **`filter_detections()`**: global confidence floor (`min_confidence`,
   default 0.25) + AOI centroid containment (when an AOI is provided) +
   sequential id renumber 0..N. SAHI handles slice merging during
   inference, so no separate NMS step.

## Temporal workflow

`ProcessingWorkflow.run(job_id)` chains 5 activities sequentially, each
with its own retry policy (`_ml_retry` for `detect`, `_default_retry` for
the rest), 10-minute timeouts (30 min for `detect`):

| # | Activity            | Reads from checkpoint | Writes to checkpoint                          |
|---|---------------------|-----------------------|-----------------------------------------------|
| 1 | `load_imagery`      | —                     | `load`: `clipped_path`, `transform`, `crs`, `aoi_wkt` |
| 2 | `detect`            | `load`                | `detect`: `detector`, `detectors`, `detection_count`, `overlay_path`; persists rows to `detections` |
| 3 | `export_results`    | `detect` (DB rows)    | `export`: `geojson_path`                      |
| 4 | `compute_indicators`| `detect` + AOI        | `indicators`: paths to CSV/JSON               |
| 5 | `finalize_job`      | —                     | sets `status=COMPLETED`, `completed_at`       |

Idempotency: each activity checks its slot and skips work if already
populated. Failures retry per the policy; non-retryable types
(`FileNotFoundError`, `ValueError`) abort.

## Data model

Postgres + PostGIS, all geometry SRID 4326:

```
territories         — registered AOIs
processing_jobs     — id (UUID), status, input_path, config (JSON),
                      checkpoint_data (JSON), error_message,
                      created/updated/completed timestamps
detections          — composite PK (job_id, id), class_name, confidence,
                      source_model, geometry (POLYGON bbox)
zone_indicators     — per-zone count, density_per_km2, total_area_m2
quality_metrics     — placeholder, currently unused
```

Migrations are sequential Alembic versions in
`packages/core/src/core/alembic/versions/`. Latest: `007_detection_source_model`.

## Async + concurrency model

- FastAPI handlers are async; SQLAlchemy uses `asyncpg`.
- ML/torch/rasterio code is blocking. Activities wrap blocking calls in
  `asyncio.to_thread()` at the async boundary
  (e.g. `await asyncio.to_thread(build_from_specs, specs)`,
  `await asyncio.to_thread(detector.detect, raster)`).
- Temporal activity workers run blocking work without starving the FastAPI
  event loop because they're separate processes.

## CRS handling

CRS is passed explicitly through the pipeline rather than inferred:
`ImageryLoaderService.load_clipped()` returns it in `Raster.crs`; the
`load` checkpoint persists `crs` as a string; downstream activities
hydrate it back into `Raster` instances. Geographic computations (area,
density) use `pyproj.Geod(ellps="WGS84")` — never raw `geometry.area`,
which is meaningless on lon/lat.

## Output schema

`detections.geojson` is a `FeatureCollection` of polygon (bbox) features:

| Field          | Type    |                                                           |
|----------------|---------|-----------------------------------------------------------|
| `id`           | int     | 0..N, post-filter sequential                              |
| `class_name`   | string  | model-native label                                        |
| `confidence`   | float   | [0, 1]                                                    |
| `source_model` | string  | which detector emitted this (e.g. `segformer-landscape`)  |
| geometry       | Polygon | bbox in EPSG:4326                                         |

## Adding a new detector

1. Implement a class with `name: str` and `detect(raster) -> list[Detection]`
   stamping `source_model=self.name` on each emission.
2. Add a builder function and a `_BUILDERS["my-key"] = _build_my`
   registration in `factory.py`.
3. Submit jobs with `{"detectors": [{"name": "my-key", "classes": [...]}]}`.

No changes to workflow, worker, exporter, schema, or DB.

## Known limitations

- ADE20K segmentation models (`segformer-landscape`, `beit-ade`) were
  trained on street-level oblique imagery. On true nadir aerial they
  under-segment and miss small features — known fit issue, not a config
  problem. Tunable knobs:
  `kwargs.max_dim` (raise to keep small features after downsampling),
  `kwargs.min_pixels` (lower to keep small CCs).
- `quality_metrics` table is unused.
- No cross-model NMS; overlapping detections from different models are
  both kept. Use disjoint `classes` allowlists per spec to avoid this.
