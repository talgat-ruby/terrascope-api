"""Process-wide configuration: env-driven cache + weights paths.

Each process owns its own kwargs (see `ProcessSpec.kwargs`); this module
only resolves the *defaults* so a fresh checkout doesn't write to a
CWD-relative `./tmp/weights` (which breaks every time the user `cd`s
elsewhere) and doesn't shard its caches across user installs.

Environment variables (all optional):

- ``TERRASCOPE_CACHE_DIR``     base cache root. Defaults to
                               ``~/.cache/terrascope``.
- ``TERRASCOPE_WEIGHTS_DIR``   model weights cache. Defaults to
                               ``$TERRASCOPE_CACHE_DIR/weights``.
- ``TERRASCOPE_LOG_LEVEL``     root log level (DEBUG/INFO/WARNING/...).
                               Default WARNING.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser()
    return default


CACHE_DIR: Path = _env_path(
    "TERRASCOPE_CACHE_DIR", Path.home() / ".cache" / "terrascope"
)

WEIGHTS_DIR: Path = _env_path("TERRASCOPE_WEIGHTS_DIR", CACHE_DIR / "weights")


def msft_buildings_cache() -> Path:
    return CACHE_DIR / "msft_buildings"


def osm_roads_cache() -> Path:
    return CACHE_DIR / "osm_roads"


def log_level() -> str:
    return os.environ.get("TERRASCOPE_LOG_LEVEL", "WARNING").upper()


__all__ = [
    "CACHE_DIR",
    "WEIGHTS_DIR",
    "log_level",
    "msft_buildings_cache",
    "osm_roads_cache",
]
