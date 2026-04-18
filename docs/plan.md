# Terrascope -- Implementation Plan

## Context

Build a prototype for extracting objects/features from satellite imagery and generating GIS layers. Detects multiple classes (buildings, roads, vegetation/land cover) using pretrained models, exports as GeoJSON/GeoPackage, computes zone-level indicators.

### Architecture Decisions

- **Python 3.14** (no downgrade)
- **Monorepo**: uv workspaces with 4 packages (`core`, `api`, `worker`, `cli`)
- **Workflow orchestration**: Temporal (full resumability per activity)
- **Database**: PostgreSQL + PostGIS (port 35432 local)
- **ORM**: SQLModel
- **ML**: Pretrained models (torchgeo + samgeo)
- **Dev infra**: Docker Compose split configs in `infra/compose/` with `.env.example`

---

## Phase 1: Infrastructure & Foundation -- COMPLETED

- uv workspace with 4 packages (`core`, `api`, `worker`, `cli`)
- Docker Compose: split into `infra/compose/` (postgres, temporal with ES, admin tools, namespace creation, UI)
- `.env.example` with all env vars; non-standard ports (35432, 37233, 38080, 39200)
- Config: `Settings` uses `SettingsConfigDict(env_file=".env")`, computed fields for `database_url` and `temporal_address` from individual `POSTGRES_*`/`TEMPORAL_*` env vars
- API: `api_port` setting (30001), `uvicorn.run` in `__main__` block
- Core models: `Detection`, `Territory`, `ProcessingJob`, `ZoneIndicator`, `QualityMetrics`, `Tile`
- Core schemas: detection, territory, processing, indicator, quality, export
- `CLASS_REGISTRY`: building, road, vegetation, water
- Alembic: env.py reads DB URL from env vars via `dotenv`; migrations use pure `sa.String()` (no sqlmodel dependency)
- All SQLModel tables use `# type: ignore[assignment]` on `__tablename__` for pyright
- Pyright configured with `extraPaths = ["packages/core/src"]`
- API stubs: health, imagery upload, processing start/status/log/retry, results endpoints
- Worker stubs: `ProcessingWorkflow` with 6 activity stubs, connects via `settings.temporal_address`
- CLI stubs: process, stac, evaluate, worker commands
- 12 tests passing

---

## Phase 2: Imagery Processing Pipeline -- COMPLETED

- `ImageryLoaderService` (`services/imagery.py`): `load()` validates CRS, `clip_to_aoi()` reprojects AOI via pyproj+shapely if CRS mismatch then clips with `rasterio.mask.mask(crop=True, nodata=0)`, `get_metadata()` returns band_count/crs/bounds/resolution
- `TilerService` (`services/tiler.py`): `generate_tiles()` splits raster into overlapping tiles (stride = tile_size - overlap), pads edge tiles with zeros, sets `valid_mask=False` for padded pixels; `tile_bounds()` returns geographic bbox from tile transform
- `StacService` (`services/stac.py`): sync `search()` via pystac-client with configurable collection/bbox/datetime, async `download()` streams COG via httpx with configurable `asset_key` (default "visual")
- Services are sync (rasterio is blocking C lib); Temporal activities will wrap in `asyncio.to_thread()` in Phase 7
- CRS comparison uses `pyproj.CRS` objects for robustness
- Affine tuple unpacking uses `# type: ignore[misc]` for pyright (known Affine typing limitation)
- `services/__init__.py` exports all three services
- 7 imagery tests + 8 tiler tests = 15 new tests (27 total passing)

---

## Phase 3: ML Detection Engine -- COMPLETED

- `DetectorService` (`services/detector.py`): `predict_tile()` dispatches to TorchGeo (vegetation/road/water) and SAMGeo (buildings), `predict_tile_with_masks()` for testing/external masks, `_mask_to_polygons()` converts probability masks to vector polygons via `rasterio.features.shapes` + shapely, `_assign_confidence()` computes mean probability in bounding region as 0-100 score
- `RawDetection` dataclass: intermediate detection format (class_name, confidence, geometry as `BaseGeometry`, source)
- `TorchGeoModel` (`services/models/torchgeo_model.py`): wraps torchgeo ResNet18 with Sentinel-2 MOCO weights for semantic segmentation; produces per-class probability masks for vegetation, road, water
- `SamGeoModel` (`services/models/samgeo_model.py`): wraps segment-geospatial for building instance segmentation; writes temp GeoTIFF (SAMGeo requires file input), reads back binary mask
- Both models follow same pattern: `load()` initializes weights, `predict()` returns `dict[str, NDArray[float32]]` mapping class names to probability masks
- 9 detector tests covering mask-to-polygon, empty/below-threshold masks, confidence scoring, valid_mask respect, unknown class filtering, multi-region detection (36 total passing)

---

## Phase 4: Post-Processing

**Goal**: Merge tiles, NMS, filter false positives, simplify.

### Files to create

- `packages/core/src/core/services/postprocessor.py` -- `PostprocessorService`
  - `merge_tile_detections(tile_detections)`
  - `apply_nms(detections, iou_threshold)` -- STRtree spatial index
  - `filter_by_size`, `filter_by_confidence`, `filter_by_shape`
  - `simplify_geometries(detections, tolerance)`
  - `clip_to_aoi(detections, aoi)`
  - `run(detections, config) -> (list[Detection], stats_dict)`

### Tests

- `tests/core/test_postprocessor.py`

---

## Phase 5: GIS Export

**Goal**: GeoJSON (mandatory), GeoPackage, Shapefile export with proper attributes.

### Files to create

- `packages/core/src/core/services/exporter.py` -- `GISExporterService`
  - `to_geodataframe(detections, crs)` -- id, class, confidence, source, geometry, area_m2, length_m, date, change_flag
  - `export_geojson(gdf, path)` -- WGS84 for RFC 7946
  - `export_geopackage(gdf, path)` / `export_shapefile(gdf, path)`
  - `export_all(detections, crs, output_dir)`
  - `generate_qgis_project(layers, imagery_path, output_path)` -- .qgs XML

### Tests

- `tests/core/test_exporter.py` -- round-trip, CRS, attributes

---

## Phase 6: Analytics/Indicators

**Goal**: Per-zone quantitative metrics, persist to DB.

### Files to create

- `packages/core/src/core/services/indicators.py` -- `IndicatorCalculatorService`
  - `compute(detections, zones)` -- spatial intersection, count/density/area per class
  - `export_csv` / `export_json`
  - `generate_summary_table(indicators)`

### Tests

- `tests/core/test_indicators.py`

---

## Phase 7: Temporal Workflows & API Endpoints

**Goal**: Wire up Temporal workflows with real activities + finalize REST API.

### Worker package (`packages/worker/src/worker/`)

- `workflows/processing.py` -- `ProcessingWorkflow` with 6 activities:
  1. `load_imagery` -- validate GeoTIFF, save metadata to DB
  2. `tile_imagery` -- generate tiles, store manifest as checkpoint
  3. `detect_objects` -- ML inference, save raw detections to DB
  4. `postprocess` -- NMS + filtering, update detections
  5. `export_results` -- generate GeoJSON/GeoPackage files
  6. `compute_indicators` -- zone stats, save to DB
- Each updates `ProcessingJob.current_step` + `checkpoint_data`
- Resumes from last completed activity on restart

### API package (`packages/api/src/api/`)

- Finalize all router implementations (currently stubs)
- Wire `POST /processing/start` to trigger Temporal workflow
- Wire `GET /results/{job_id}/detections` to query PostGIS with `ST_AsGeoJSON`

### Tests

- `tests/api/test_api_*.py`
- `tests/worker/test_workflows.py`, `test_activities.py`

---

## Phase 8: CLI Tool

**Goal**: Typer CLI -- direct processing or via Temporal.

### Files to update (`packages/cli/src/cli/commands/`)

- `process.py` -- implement using core services directly or via Temporal client
- `stac.py` -- implement using `StacService`
- `evaluate.py` -- implement using `QualityEvaluatorService`
- `worker.py` -- already wired to `worker.main`

### Tests

- `tests/cli/test_cli.py`

---

## Phase 9: Quality Evaluation

**Goal**: Precision, Recall, F1, IoU, mAP + visualization.

### Files to create (`packages/core/src/core/services/`)

- `evaluator.py` -- `QualityEvaluatorService`
  - `evaluate(predictions, ground_truth, iou_threshold)` -- per-class metrics
  - `generate_report(metrics, error_examples)`
  - `create_control_sample(detections, sample_size)`
- `visualization.py` -- `VisualizationService`
  - `render_overlay(imagery_path, detections, output_path)` -- matplotlib PNG

### Tests

- `tests/core/test_evaluator.py`

---

## Phase 10: Testing, Docker & Documentation

### Files to create

- `tests/conftest.py` -- synthetic GeoTIFF, sample AOI, mock detector, test DB (testcontainers)
- `tests/integration/test_pipeline.py` -- end-to-end with mocked detector
- `Dockerfile.api` -- multi-stage (builder + runtime for API)
- `Dockerfile.worker` -- multi-stage (builder + runtime for worker)
- `docs/technical_report.md` -- data description, approach, metrics, conclusions, limitations
- Update root `README.md`

---

## Verification Plan

1. `uv sync` -- all workspace packages resolve
2. `docker compose -f infra/compose/compose.yml up` -- postgres, temporal, ES, UI start
3. `alembic upgrade head` -- PostGIS tables created
4. `pytest` after each phase
5. Upload GeoTIFF via API -> Temporal workflow completes -> PostGIS has detections
6. Download GeoJSON -> open in QGIS -> CRS aligns with basemap
7. CLI: `terrascope process --input sample.tif --aoi aoi.geojson --output ./out/`
8. Kill workflow mid-step -> restart -> resumes from checkpoint

---

## Key Dependencies by Package

| Package  | Dependencies                                                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `core`   | sqlmodel, geoalchemy2, asyncpg, rasterio, geopandas, shapely, pyogrio, pyproj, torch, torchvision, torchgeo, segment-geospatial, pystac-client, pydantic-settings, matplotlib |
| `api`    | core (workspace), fastapi[standard], temporalio                                                                                  |
| `worker` | core (workspace), temporalio                                                                                                     |
| `cli`    | core (workspace), typer, temporalio                                                                                              |
| dev      | pytest, pytest-cov, pytest-asyncio, httpx, alembic, python-dotenv                                                                |
