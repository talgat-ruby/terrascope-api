# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Build stage: install dependencies and workspace packages using uv.
# This is a uv workspace monorepo, so we need the full repo as build context.
# Build with: docker build -f packages/api/Dockerfile -t terrascope-api .
FROM ghcr.io/astral-sh/uv:python3.14-bookworm AS builder

# Install system build dependencies for native extensions (GDAL, asyncpg, pyproj).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libpq-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY packages/api/pyproject.toml packages/api/pyproject.toml
COPY packages/worker/pyproject.toml packages/worker/pyproject.toml
COPY packages/cli/pyproject.toml packages/cli/pyproject.toml

# Install third-party dependencies only (cached unless lockfile changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace

# Copy source code and install workspace packages.
COPY packages/ packages/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

################################################################################
# Runtime stage: slim image with only the runtime libraries.
FROM python:3.14-slim-bookworm AS final

# Install runtime-only system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal34 \
    libpq5 \
    libproj25 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

WORKDIR /app

# Copy the virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Create writable directories for uploads and output.
RUN mkdir -p /app/uploads /app/output && chown -R appuser:appuser /app/uploads /app/output

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose the API port.
EXPOSE 8080

USER appuser

CMD ["python", "-m", "api.main"]
