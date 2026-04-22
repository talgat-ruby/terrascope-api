# Terrascope API

Satellite imagery analysis system that detects objects (buildings, roads, vegetation, water) in GeoTIFF imagery and exports results as GIS layers (GeoJSON, GeoPackage) with zone-level indicators and quality metrics.

## Tech Stack

- **Language:** Python 3.14+
- **Package manager:** [uv](https://docs.astral.sh/uv/) (workspaces monorepo)
- **API:** FastAPI
- **Database:** PostgreSQL + PostGIS, SQLModel ORM
- **Workflow orchestration:** Temporal
- **ML:** PyTorch, TorchGeo, SAMGeo
- **Geospatial:** Rasterio, GeoPandas, Shapely, GeoAlchemy2
- **Linting:** Ruff, Pyright

## Project Structure

```
packages/
  core/    # Shared models, schemas, services, config, database
  api/     # FastAPI application and routers (+ Dockerfile)
  worker/  # Temporal workflows and activities (+ Dockerfile)
  cli/     # Typer CLI tool
  core/alembic/  # Database migrations (inside core package)
infra/     # Docker Compose (Postgres, Temporal, Elasticsearch)
tests/     # Test suite (mirrors packages/ structure)
docs/      # Technical report, assignment spec, implementation plan
```

## Architecture

The system implements a 6-step processing pipeline:

```
Load GeoTIFF -> Tile -> Detect Objects -> Post-process -> Export GIS -> Compute Indicators
```

- **Detection:** TorchGeo FCN for vegetation/road/water, SAMGeo for buildings
- **Post-processing:** NMS, confidence/size/shape filtering, geometry simplification
- **Export:** GeoJSON (WGS84), GeoPackage, Shapefile with geodesic area_m2
- **Orchestration:** Temporal workflows with per-activity checkpointing, or local CLI

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- Docker & Docker Compose

## Setup

```bash
# Install dependencies
uv sync

# Copy environment config
cp .env.example .env

# Start infrastructure (Postgres, Temporal, Elasticsearch)
docker compose -f infra/compose/compose.yml --env-file .env up -d

# Run database migrations
uv run alembic -c packages/core/src/core/alembic.ini upgrade head
```

## Development

```bash
# API server with auto-reload (port from API_PORT in .env)
uv run python -m api.main

# Temporal worker
uv run python -m worker.main
```

API docs available at <http://localhost:30001/docs>

## CLI

```bash
# Local processing pipeline (AOI is optional -- uses full raster extent if omitted)
uv run terrascope process --input image.tif --output ./results

# Local processing with explicit AOI
uv run terrascope process --input image.tif --aoi aoi.geojson --output ./results

# Resume interrupted pipeline (checkpoints are saved automatically)
uv run terrascope process --input image.tif --output ./results

# Force fresh run, ignoring existing checkpoints
uv run terrascope process --input image.tif --output ./results --no-resume

# Submit to Temporal workflow
uv run terrascope process --input image.tif --aoi aoi.geojson --use-temporal

# STAC catalog search
uv run terrascope stac search --bbox 10.0,49.0,11.0,50.0 --datetime 2024-01-01/2024-06-01

# Download STAC asset
uv run terrascope stac download --bbox 10.0,49.0,11.0,50.0 --output ./data

# Quality evaluation against ground truth
uv run terrascope evaluate --predictions preds.geojson --ground-truth gt.geojson --report report.json
```

### Checkpoint / Resume

The local processing pipeline saves intermediate results to `{output_dir}/.checkpoints/` after each step. If the pipeline is interrupted (crash, Ctrl+C, etc.), re-running the same command resumes from the last completed step -- skipping expensive ML inference.

Checkpoints are automatically invalidated when input file or processing settings change. Use `--no-resume` to force a fresh run.

Pipeline steps with individual checkpoints:

1. **Load imagery** -- raster data + transform + CRS
2. **Tile** -- all tile arrays + metadata
3. **Detect** -- raw detections as GeoJSON
4. **Post-process** (6 substeps, each checkpointed):
   - NMS, confidence filter, size filter, shape filter, simplify, clip to AOI
5. **Export** -- GeoJSON, GeoPackage, Shapefile
6. **Compute indicators** -- CSV + JSON

## Docker

Each service has its own Dockerfile inside its package directory. Build from the repo root (required for uv workspace resolution):

```bash
# Build images
docker build -f packages/api/Dockerfile -t terrascope-api .
docker build -f packages/worker/Dockerfile -t terrascope-worker .

# Run API with infrastructure
docker run --rm --network terrascope-network \
  -e POSTGRES_HOST=terrascope-postgres \
  -e TEMPORAL_HOST=terrascope-temporal \
  -p 8080:8080 terrascope-api

# Run worker with infrastructure
docker run --rm --network terrascope-network \
  -e POSTGRES_HOST=terrascope-postgres \
  -e TEMPORAL_HOST=terrascope-temporal \
  -v model_cache:/app/model_cache \
  terrascope-worker
```

Per-service compose files are also available:

```bash
docker compose -f packages/api/compose.yaml up --build
docker compose -f packages/worker/compose.yaml up --build
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run integration tests only
uv run pytest tests/integration/
```

## Linting

```bash
uvx ruff check .
uvx ruff format .
uvx pyright
```

## Local Development Ports

| Service       | Port  |
|---------------|-------|
| API           | 30001 |
| PostgreSQL    | 35432 |
| Temporal      | 37233 |
| Temporal UI   | 38080 |
| Elasticsearch | 39200 |

## Documentation

- [Technical Report](docs/technical_report.md) -- data description, approach, metrics, conclusions, limitations
- [Assignment Spec](docs/assignment.md) -- original requirements
- [Implementation Plan](docs/plan.md) -- phase-by-phase development log
