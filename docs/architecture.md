# Terrascope — Architecture

## Goal

Take a geo-referenced raster (GeoTIFF), run one or more pluggable
processes against it, and emit:

- `detections.geojson` — per-detection geometries + §5 metadata, in EPSG:4326
- `overlay.png` — annotated source image
- `indicators.json` + `indicators.csv` — per-class counts/density/area for the AOI

The system is multi-process first-class: a job declares a list of
`ProcessSpec`s, each scoped to a class allowlist with its own confidence
threshold. Outputs from every process are merged with a `source_model`
provenance tag and written together.

## High-level pipeline

```
            ┌────────────────────────────────────────────────────────────────┐
GeoTIFF ──▶ │ load_raster   run_processes   render_overlay   export   indicators │ ──▶ output/
            └────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                     ProcessSpec[]  (from job.config.processes)
```

A single in-process pipeline. There is **no database, no queue, no
async runtime**: the CLI walks the raster through the five steps in
order. An earlier iteration used FastAPI + Temporal + Postgres/PostGIS;
that layer was removed in favor of the simpler local pipeline.

## Code layout

```
packages/
  core/
    src/core/
      detection/     Detection / Raster dataclasses + filter_detections helper
      io/            load_raster (rasterio), export_geojson, write_indicators
      geo/           CRS / bbox helpers shared across processes
      processes/     One Process per (model, dataset) pair, plus registry
      vendor/sam_road/  Vendored sam_road inference code
      orchestrator.py
      renderer.py    PNG overlay
      config.py      Env-driven cache + weights paths
  cli/               Typer CLI (`terrascope`)
infra/               Docker Compose (legacy; not required for the CLI)
docs/                This document, plan, assignment spec
```

## Process layer

The contract is one Protocol:

```python
class Process(Protocol):
    name: str
    spec: ProcessSpec
    def run(self, raster: Raster) -> list[Detection]: ...
```

Implementations live in `packages/core/src/core/processes/` and never
leak model-specific structures upward — only `Detection` objects. Unlike
a pure inference contract, a `Process` owns its full pipeline: it may
download external data (OSM, MS Building Footprints), preprocess to a
target GSD, run inference, and post-process — all inside `run()`.

### ProcessSpec — declarative job config

```python
@dataclass(frozen=True)
class ProcessSpec:
    name: str                              # registry key
    classes: tuple[str, ...] | None = None # allowlist; None = accept all
    min_confidence: float | None = None    # per-process override
    kwargs: dict[str, Any] = field(default_factory=dict)
```

A job's config carries `processes: list[ProcessSpec dict]`. The CLI
parses each via `ProcessSpec.from_dict()` and resolves them through
`build()`.

### Registry (`processes/registry.py`)

`_BUILDERS: dict[str, _Entry]` — one entry per registered process. Each
entry pairs a builder callable with an optional Pydantic config model.
Adding a process = one `register("my-key", MyProcess.from_spec, config_model=…)`
call (the `config_model` is optional). Currently registered:

| Key                        | Backend / data source                                    |
|----------------------------|----------------------------------------------------------|
| `msft-buildings`           | Microsoft Global Building Footprints (vector, no model)  |
| `osm-roads`                | OpenStreetMap via Overpass API (vector, no model)        |
| `unet-roads`               | Local U-Net checkpoint (`road_unet.pth`, segmentation)   |
| `sam-road`                 | `congrui/sam_road` cityscale ViT-B 512 (graph extraction)|
| `sam-road-spacenet`        | sam-road SpaceNet ViT-B 256 preset                       |
| `yolov8n-sahi`             | `yolov8n.pt` (COCO 80 generic) via SAHI                  |
| `yolov8-obb-aerial`        | `yolov8s-obb.pt` (DOTA v1, 15 aerial classes)            |
| `yolov8-obb-dota-v2`       | `yolov8x-obb.pt` (DOTA v2)                               |
| `yolov8-satellite-vehicle` | `keremberke/yolov8m-satellite-vehicle-detection` (HF)    |
| `yolo26{n,m}-sahi`, `yolo26-obb`, `yolo26-seg` | YOLO26 (Ultralytics late-2025 successor) |

#### Optional Pydantic kwargs validation

Builders may pass `config_model=SomeBaseModel` to `register()`. When
present, `build(spec)` validates `spec.kwargs` against the model and
raises `ValueError("Invalid kwargs for process X: …")` on unknown keys
or wrong types — opt-in, fully backwards compatible.

`msft-buildings` is the reference implementation
(`MsftFootprintConfig`, forbids extra keys); the rest still accept the
raw dict and self-validate inside `from_spec`.

### Orchestrator

`run_processes(procs, raster) -> list[Detection]` walks each process in
spec order:

```python
for proc in processes:
    spec = proc.spec
    allow = set(spec.classes) if spec.classes is not None else None
    for det in proc.run(raster):
        if allow is not None and det.class_name not in allow:
            continue
        if spec.min_confidence is not None and det.confidence < spec.min_confidence:
            continue
        if det.source_model != proc.name:
            det = replace(det, source_model=proc.name)
        merged.append(det)
return [replace(d, id=i) for i, d in enumerate(merged)]
```

Per-spec class allowlist + per-spec confidence floor + provenance
stamping + sequential id renumber. There is **no cross-process NMS** —
class-level deduplication is left to the process that owns the class
via the allowlist. Processes that need internal NMS (sam-road, SAHI)
handle it themselves.

### Backends

- **MsftFootprintProcess / OsmRoadsProcess** — vector-only. Fetch from
  CDN / Overpass, cache shards on disk under `$TERRASCOPE_CACHE_DIR`,
  clip to the AOI, emit detections. OSM tile fetches fan out across a
  thread pool (`max_workers=8`); a DNS / connection-refused error
  latches the rest of the run into cache-only mode.
- **UnetRoadsProcess** — sliding 512×512 patches, sigmoid-averaged
  probability map, threshold + `rasterio.features.shapes` →
  Polygon detections.
- **SamRoadProcess** — graph extraction via the vendored sam-road
  network. Tiles the raster on top of the model's internal patch grid;
  stitches tile graphs by collapsing nodes within `ROAD_NMS_RADIUS`.
  Resamples to the checkpoint's training GSD when `source_gsd_m_per_px`
  is set, so high-res inputs don't see microscope-scale imagery.
- **YoloSahiProcess** — Ultralytics + SAHI. Single-pass inference for
  long-side ≤ `full_image_threshold` (default 1024); otherwise SAHI
  sliced prediction. HuggingFace checkpoints are downloaded once into
  `$TERRASCOPE_WEIGHTS_DIR`; gated/missing repos surface as a clean
  `FileNotFoundError` with a `huggingface-cli login` hint.

## CRS handling

CRS is propagated explicitly through the pipeline:

- `load_raster()` returns a `Raster` carrying its native CRS string and
  the optional WGS84 AOI used for clipping.
- Process internals work in the raster's CRS; vector-data fetches
  (OSM, MSFT) project from WGS84 into the raster CRS using the shared
  helpers in `core/geo/raster_utils.py`
  (`raster_roi_wgs84`, `bbox_to_pixels`).
- The exporter reprojects to **EPSG:4326** on the way out so the layer
  opens cleanly in QGIS aligned to any web base map (§5, §10).
- Geographic computations (area, length, density) use
  `pyproj.Geod(ellps="WGS84")` — never raw `geometry.area`, which is
  meaningless on lon/lat.

## Output schema (assignment §5)

`detections.geojson` is a `FeatureCollection` in **EPSG:4326**. Each
feature carries:

| Field          | Type    |                                                           |
|----------------|---------|-----------------------------------------------------------|
| `id`           | int     | 0..N, post-filter sequential                              |
| `class`        | string  | model-native label (renamed from internal `class_name`)   |
| `confidence`   | float   | **0–100** (scaled at export from internal 0–1)            |
| `source`       | string  | scene identifier (input GeoTIFF stem)                     |
| `source_model` | string  | which process emitted this (e.g. `osm-roads`)             |
| `area_m2`      | float   | optional, polygons only — geodesic                        |
| `length_m`     | float   | optional, lines only — geodesic                           |
| `centroid_wkt` | string  | for tools that don't read GeoJSON geometry directly       |
| `bbox`         | list    | raster-CRS bbox `[minx, miny, maxx, maxy]`                |
| `pixel_bbox`   | list    | raster pixel frame `[col_min, row_min, col_max, row_max]` |
| `geometry`     | Geometry| Point / LineString / Polygon / Multi*, EPSG:4326          |

`indicators.json` (and a flat `indicators.csv` mirror) contains:

```json
{
  "zone_area_m2": <number>,
  "total_detections": <int>,
  "by_class": [
    {"class": "building", "count": 42,
     "total_area_m2": 1234.5, "total_length_m": 0.0,
     "density_per_km2": 0.84, "area_fraction": 0.012},
    ...
  ]
}
```

The zone is taken from `Raster.aoi_geom` when an AOI was supplied,
otherwise from the raster footprint. All areas / lengths are geodesic.

## Configuration

Three optional environment variables, all with sane defaults:

| Variable                  | Default                              | Used for                              |
|---------------------------|--------------------------------------|---------------------------------------|
| `TERRASCOPE_CACHE_DIR`    | `~/.cache/terrascope`                | OSM tiles, MSFT shards, dataset CSVs  |
| `TERRASCOPE_WEIGHTS_DIR`  | `$TERRASCOPE_CACHE_DIR/weights`      | HF / SAM checkpoint cache             |
| `TERRASCOPE_LOG_LEVEL`    | `WARNING`                            | Root log level                        |

Per-process overrides via `ProcessSpec.kwargs.cache_dir` /
`weights_dir` always win over the env defaults.

## Adding a new process

1. Implement a `@dataclass` with:
   - `spec: ProcessSpec`, `name: str`
   - `from_spec(cls, spec) -> Self` classmethod
   - `run(self, raster) -> list[Detection]`
2. (Optional) Define a `pydantic.BaseModel` with `extra="forbid"` for
   strict kwargs validation.
3. `register("my-key", MyProcess.from_spec, config_model=MyConfig)`.
4. Submit jobs with `{"processes": [{"name": "my-key", ...}]}`.

No changes to the orchestrator, exporter, indicators, or CLI.

## Tiling and false-positive control (assignment §6)

- **Tiling**: `osm-roads` splits the AOI into a grid of `tile_size_deg`
  tiles, fetched concurrently and merged by element id. `sam-road`
  tiles the raster on top of the model's internal patch grid and
  stitches graphs across seams. `unet-roads` slides patches with
  `patch_size // 2` stride and averages overlapping probabilities.
- **False-positive reduction**: per-process size filters
  (`min_area_px`, `min_area_m2`, `min_edge_len_m`), AOI clipping,
  oriented-envelope simplification in the renderer, NMS inside
  sam-road and SAHI, and the spec-level `classes` allowlist + global
  `min_confidence` applied by the orchestrator.

## Known limitations

- No cross-process NMS — overlapping detections from different
  processes are both kept. Disjoint `classes` allowlists per spec
  remain the recommended way to prevent double-counting.
- `unet-roads` checkpoint must be supplied locally; there's no auto
  download (it's a project-trained model, not a published one).
- No quality metrics computation in code — assignment §7 metrics
  (precision/recall/IoU) belong in the report deliverable, evaluated
  externally against a labeled control sample.
