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

- `ImageryLoaderService` (`services/imagery.py`): `load()` validates CRS, `clip_to_aoi()` returns 3-tuple `(data, transform, crs_string)`, reprojects AOI via pyproj+shapely if CRS mismatch, explicit bbox intersection check before `rasterio.mask.mask(crop=True, nodata=0)`, `get_metadata()` returns band_count/crs/bounds/resolution
- `TilerService` (`services/tiler.py`): `generate_tiles()` accepts `crs` param and passes it to Tile, splits raster into overlapping tiles (stride = tile_size - overlap), pads edge tiles with zeros, sets `valid_mask=False` for padded pixels; `tile_bounds()` returns geographic bbox from tile transform
- `Tile` dataclass: added `crs: str = "EPSG:4326"` field for CRS propagation through pipeline
- `StacService` (`services/stac.py`): `search()` is `async def` wrapping blocking pystac-client in `asyncio.to_thread()`, async `download()` streams COG via httpx with configurable `asset_key` (default "visual")
- CRS comparison uses `pyproj.CRS` objects for robustness
- Affine tuple unpacking uses `# type: ignore[misc]` for pyright (known Affine typing limitation)
- `services/__init__.py` exports all three services
- 7 imagery tests + 8 tiler tests = 15 new tests (27 total passing)

---

## Phase 3: ML Detection Engine -- COMPLETED

- `DetectorService` (`services/detector.py`): `predict_tile()` dispatches to TorchGeo (vegetation/road/water) and SAMGeo (buildings), `predict_tile_with_masks()` for testing/external masks, `_mask_to_polygons()` converts probability masks to vector polygons via `rasterio.features.shapes` + shapely, `_assign_confidence()` computes mean probability in bounding region as 0-100 score
- `RawDetection` dataclass: intermediate detection format (class_name, confidence, geometry as `BaseGeometry`, source)
- `TorchGeoModel` (`services/models/torchgeo_model.py`): wraps torchgeo FCN for semantic segmentation; `load(in_channels)` creates model with background channel at index 0 + 3 class channels; produces per-class probability masks for vegetation, road, water
- `SamGeoModel` (`services/models/samgeo_model.py`): wraps segment-geospatial for building instance segmentation; `predict()` accepts `crs` param, writes temp GeoTIFF (SAMGeo requires file input), reads back binary mask
- Both models follow same pattern: `load()` initializes model, `predict()` returns `dict[str, NDArray[float32]]` mapping class names to probability masks
- 9 detector tests covering mask-to-polygon, empty/below-threshold masks, confidence scoring, valid_mask respect, unknown class filtering, multi-region detection (36 total passing)

---

## Phase 4: Post-Processing -- COMPLETED

- `PostprocessorService` (`services/postprocessor.py`): `merge_tile_detections()` flattens per-tile lists, `apply_nms()` uses STRtree spatial index with per-class IoU suppression (higher confidence wins), `filter_by_confidence()`, `filter_by_size()` uses `Geod(ellps="WGS84")` for geodesic area in m2, `filter_by_shape()` (removes slivers with aspect > 100), `simplify_geometries(tolerance_m)` converts meters to approximate degrees (1m ≈ 1/111320°), `clip_to_aoi()`
- `PostprocessingConfig` dataclass: iou_threshold, confidence_threshold, min/max_area_m2, simplify_tolerance_m
- `run()` orchestrator: NMS → confidence filter → size filter → shape filter → simplify → clip (optional), returns `(detections, stats_dict)` with counts at each step
- Uses `dataclasses.replace()` for immutable detection updates
- 15 postprocessor tests (51 total passing)

---

## Phase 5: GIS Export -- COMPLETED

- `GISExporterService` (`services/exporter.py`): `to_geodataframe()` converts `RawDetection` list to GeoDataFrame with class_name/confidence/source/geometry/area_m2 columns; area_m2 computed via `Geod(ellps="WGS84")` for geodesic accuracy, `export_geojson()` always reprojects to WGS84 (RFC 7946), `export_geopackage()`, `export_shapefile()`, `export_all()` writes all three formats and returns format→path mapping
- `generate_qgis_project` deferred -- can be added later as XML generation
- area_m2 uses raw geometry area (geographic CRS); accurate m2 requires reprojection to projected CRS downstream
- 8 exporter tests covering round-trip read-back, CRS enforcement, empty input (59 total passing)

---

## Phase 6: Analytics/Indicators -- COMPLETED

- `IndicatorCalculatorService` (`services/indicators.py`): `compute(detections, zones)` performs spatial intersection per zone/class, computes count, density_per_km2, total_area_m2 using `Geod(ellps="WGS84")` for geodesic accuracy; partial intersections are clipped to zone boundary
- `ZoneIndicatorResult` dataclass: zone_id, class_name, count, density_per_km2, total_area_m2
- `export_csv()` / `export_json()` write indicators to file
- `generate_summary_table()` groups by zone with per-class breakdowns and totals
- 8 indicator tests covering basic compute, outside-zone filtering, multi-zone, partial intersection, empty input, CSV/JSON export, summary table (67 total passing)

---

## Phase 7: Temporal Workflows & API Endpoints -- COMPLETED

- All 6 activity stubs implemented with real core service integration + DB persistence
- `_helpers.py` module: `get_job()`, `update_job()`, `fail_job()`, `detections_to_raw()`, `raw_to_detections()`
- `load_imagery`: loads GeoTIFF via `ImageryLoaderService`, clips to AOI, saves `.npy` to disk, stores metadata in `checkpoint_data`
- `tile_imagery`: loads clipped data, tiles via `TilerService`, saves tiles + manifest to disk
- `detect_objects`: loads tiles from manifest, runs `DetectorService.predict_tile()`, bulk inserts `Detection` rows with PostGIS geometry
- `postprocess`: queries detections from DB, runs `PostprocessorService.run()` with config, deletes raw detections and inserts filtered ones
- `export_results`: queries detections, exports via `GISExporterService.export_all()` to GeoJSON/GPKG/Shapefile, stores paths in checkpoint
- `compute_indicators`: queries detections + territories, runs `IndicatorCalculatorService.compute()`, inserts `ZoneIndicator` rows, marks job COMPLETED
- Each activity updates `ProcessingJob.status`, `current_step`, `checkpoint_data`; catches exceptions and calls `fail_job()`
- `POST /processing/start` triggers `ProcessingWorkflow` via Temporal client
- `POST /processing/{job_id}/retry` validates FAILED status, resets job, starts new workflow
- `GET /results/{job_id}/download` serves export files via `FileResponse` with format-specific media types
- STAC endpoints (`POST /stac/search`, `POST /stac/download`) implemented using `StacService`
- 8 activity tests + 5 API processing tests + 4 API results tests = 17 new tests (84 total passing)

---

## Phase 8: CLI Tool -- COMPLETED

- `process.py`: full local pipeline (load → tile → detect → postprocess → export → indicators) with `--use-temporal` flag for Temporal submission
- `stac.py`: search and download commands using `StacService`, bbox parsing, async wrappers
- `evaluate.py`: stub awaiting Phase 9 `QualityEvaluatorService`
- `worker.py`: already complete (imports `worker.main`)
- 7 CLI tests covering local processing, temporal flag, STAC search/download, evaluate stub, worker help (91 total passing)

---

## Phase 9: Quality Evaluation -- COMPLETED

- `QualityEvaluatorService` (`services/evaluator.py`): `evaluate()` with greedy IoU-based matching, per-class precision/recall/F1/IoU/mAP, `_compute_ap()` using 11-point interpolation, `generate_report()` JSON output, `create_control_sample()` for manual review
- `ClassMetrics` and `EvaluationResult` dataclasses for structured results
- Error examples (FP/FN) collected with geometry WKT for debugging
- `VisualizationService` (`services/visualization.py`): `render_overlay()` draws detection polygons on GeoTIFF imagery with per-class colors, 2-98th percentile normalization, matplotlib legend
- CLI `evaluate` command wired: loads GeoJSON predictions/GT via geopandas, prints metrics table, optional JSON report and control sample
- Services exported via `__init__.py`
- 16 new tests: 11 evaluate (perfect match, no overlap, partial, multi-class, empty inputs, greedy matching, error examples, threshold sensitivity), 2 report/sample, 2 visualization, 1 CLI evaluate (107 total passing)

---

## Phase 10: Testing, Docker & Documentation -- COMPLETED

### Files to create

- `tests/conftest.py` -- synthetic GeoTIFF, sample AOI, mock detector, test DB (testcontainers)
- `tests/integration/test_pipeline.py` -- end-to-end with mocked detector
- `Dockerfile.api` -- multi-stage (builder + runtime for API)
- `Dockerfile.worker` -- multi-stage (builder + runtime for worker)
- `docs/technical_report.md` -- data description, approach, metrics, conclusions, limitations
- Update root `README.md`

---

## Post-Phase: Optional AOI

- `ImageryLoaderService.get_bounds_geometry()`: returns full raster extent as Shapely polygon
- `ProcessingRequest.aoi` is now optional (`None` by default); when omitted, the full raster extent is used
- CLI `--aoi` flag is optional; prints "No AOI provided, using full raster extent" when omitted
- Worker `load_imagery` activity derives AOI from dataset bounds when `aoi` is missing from job config
- API `/processing/start` only includes `aoi`/`aoi_crs` in job config when explicitly provided

---

## Verification Plan

1. `uv sync` -- all workspace packages resolve
2. `docker compose -f infra/compose/compose.yml up` -- postgres, temporal, ES, UI start
3. `uv run alembic -c packages/core/src/core/alembic.ini upgrade head` -- PostGIS tables created
4. `pytest` after each phase
5. Upload GeoTIFF via API -> Temporal workflow completes -> PostGIS has detections
6. Download GeoJSON -> open in QGIS -> CRS aligns with basemap
7. CLI: `terrascope process --input sample.tif --output ./out/` (AOI optional)
8. CLI: `terrascope process --input sample.tif --aoi aoi.geojson --output ./out/`
9. Kill workflow mid-step -> restart -> resumes from checkpoint

---

## Key Dependencies by Package

| Package  | Dependencies                                                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `core`   | sqlmodel, geoalchemy2, asyncpg, rasterio, geopandas, shapely, pyogrio, pyproj, torch, torchvision, torchgeo, segment-geospatial, pystac-client, pydantic-settings, matplotlib |
| `api`    | core (workspace), fastapi[standard], temporalio                                                                                  |
| `worker` | core (workspace), temporalio                                                                                                     |
| `cli`    | core (workspace), typer, temporalio                                                                                              |
| dev      | pytest, pytest-cov, pytest-asyncio, httpx, alembic, python-dotenv                                                                |
