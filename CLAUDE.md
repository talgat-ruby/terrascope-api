# CLAUDE.md

## Project Overview

Terrascope is a satellite imagery analysis system that detects objects (buildings, roads, vegetation, water) in GeoTIFF imagery and exports results as GIS layers (GeoJSON, GeoPackage). It computes zone-level indicators and quality metrics.

**Tech stack:** Python 3.14, SQLModel, PostgreSQL + PostGIS, PyTorch + TorchGeo + SAMGeo, uv workspaces monorepo. Linting with ruff and pyright.

## Repository Structure

```
packages/
  core/    # Shared models, schemas, services, config, database
  cli/     # Typer CLI tool
  core/alembic/  # Database migrations (inside core package)
infra/     # Docker Compose configs (Postgres)
tests/     # Mirrors packages/ structure
docs/      # Assignment spec and implementation plan
```

## Setup

```bash
uv sync
cp .env.example .env
docker compose -f infra/compose/compose.yml --env-file .env up -d
uv run alembic -c packages/core/src/core/alembic.ini upgrade head
```

## Common Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Linting
uvx ruff check .
uvx ruff format .

# Type checking
uvx pyright

# Database migration
uv run alembic -c packages/core/src/core/alembic.ini revision --autogenerate -m "description"
uv run alembic -c packages/core/src/core/alembic.ini upgrade head
```

## Architecture & Conventions

- **Monorepo via uv workspaces** - each package under `packages/` has its own `pyproject.toml`; root `pyproject.toml` declares the workspace
- **Async everywhere** - AsyncPG, async SQLAlchemy sessions
- **SQLModel** - combined ORM + Pydantic validation in a single class. Models live in `packages/core/src/core/models/`
- **PostGIS** - all geometry stored with SRID=4326 using GeoAlchemy2
- **Config** - pydantic-settings `Settings` class in `core/config.py`, reads from `.env`
- **CRS propagation** - CRS is passed explicitly through the pipeline: `clip_to_aoi()` returns CRS string, `Tile` has `crs` field, model wrappers accept `crs` param
- **Geodesic accuracy** - use `pyproj.Geod(ellps="WGS84")` for area/distance computations on geographic coordinates, not raw `geometry.area`
- **Blocking I/O in async** - wrap blocking libs (rasterio, pystac-client) in `asyncio.to_thread()` at the async boundary

## Code Style

- Type hints on all function signatures (Python 3.14+ — use built-in generics: `list`, `dict`, `tuple`, not `typing.List`)
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- No legacy compatibility code — target Python 3.14 only
- Prefer `async def` for I/O-bound operations
- Keep models in `core/models/`, schemas in `core/schemas/`, business logic in `core/services/`

## Database

- PostgreSQL with PostGIS extension
- Migrations via Alembic (sequential numbering: `001_`, `002_`, ...)
- Tables: `territories`, `processing_jobs`, `detections`, `zone_indicators`, `quality_metrics`
- UUID primary keys, JSON columns for flexible config/checkpoint data
- Foreign keys have `index=True` for query performance

## Testing

- pytest with `asyncio_mode=auto`
- Test files mirror source structure under `tests/`
- Integration tests go in `tests/integration/`

## Key Ports (Local Development)

| Service    | Port  |
|------------|-------|
| PostgreSQL | 35432 |

## Implementation Plan

The full implementation plan with phases, steps, and status is in `docs/plan.md`. Always refer to it before starting work on a phase or step.

Phases 1-10 are complete. AOI is now optional in both CLI and API (falls back to full raster extent).
