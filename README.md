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
  api/     # FastAPI application and routers
  worker/  # Temporal workflows and activities
  cli/     # Typer CLI tool
alembic/   # Database migrations
infra/     # Docker Compose (Postgres, Temporal, Elasticsearch)
tests/     # Test suite (mirrors packages/ structure)
docs/      # Assignment spec and implementation plan
```

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
docker compose -f infra/compose/compose.yml up -d

# Run database migrations
alembic upgrade head
```

## Development

```bash
# API server with auto-reload (port from API_PORT in .env)
uv run python -m api.main

# Temporal worker
uv run python -m worker.main
```

API docs available at http://localhost:30001/docs

## Testing

```bash
uv run pytest
uv run pytest --cov
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
