# Terrascope API

FastAPI application for Terrascope.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

## Run

Development server with auto-reload:

```bash
uv run fastapi dev src/app/main.py
```

Production server:

```bash
uv run fastapi run src/app/main.py
```

API docs available at http://localhost:8000/docs

## Tests

```bash
uv run pytest
```
