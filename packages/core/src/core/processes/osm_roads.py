"""OSM Roads — fetch road network from OpenStreetMap via Overpass API.

Vector-only — no model inference. Given a Raster (already loaded +
clipped), determines the WGS84 bounding box of the AOI, queries the
Overpass API for all ways tagged with ``highway``, and emits one
``Detection`` per road segment whose pixel footprint is non-trivial.

Mirrors ``MsftFootprintProcess`` in structure: responses are cached on
disk so repeat runs (same AOI, same process) don't re-query.

Highway tags are mapped to three broad class names by default:

    major_road  — motorway, trunk, primary (and _link variants)
    road        — secondary, tertiary, unclassified
    local_road  — residential, service, living_street

Override via ``kwargs.class_map`` (partial override — only the keys you
supply are changed; everything else uses the default).

``kwargs`` knobs:
- ``overpass_url``   (str) — Overpass API endpoint. Default is the
                     public de instance. Self-host or use
                     ``https://overpass.kumi.systems/api/interpreter``
                     for rate-limit relief.
- ``cache_dir``      (str | Path) — local cache root. Default
                     ``~/.cache/terrascope/osm_roads``.
- ``class_map``      (dict[str, str]) — partial override of the OSM
                     highway value → class_name mapping.
- ``timeout``        (int, default 220) — HTTP socket timeout in seconds.
                     Keep larger than ``server_timeout``.
- ``server_timeout`` (int, default 180) — Overpass server-side processing
                     timeout (the ``[timeout:N]`` knob in the QL query).
                     Raise for very large AOIs.
- ``tile_size_deg``  (float, default 0.1) — split the AOI into a grid of
                     sub-bboxes of at most this many degrees per side and
                     query each tile separately. Keeps individual queries
                     under Overpass's processing budget for large rasters
                     and lets a partial failure leave most tiles cached.
- ``max_retries``    (int, default 3) — retry transient 5xx / timeouts
                     per tile with exponential backoff before giving up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, MultiLineString, Point, box

from core.config import osm_roads_cache
from core.detection.types import Detection, Raster
from core.geo.raster_utils import bbox_to_pixels, raster_roi_wgs84

from core.processes.base import ProcessSpec
from core.processes.registry import register

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())

_DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM ``highway`` tag value → broad class name used in Detection.
# Only tags present in this map are emitted; everything else is dropped.
# Excludes pedestrian/cycling infrastructure by design.
_DEFAULT_CLASS_MAP: dict[str, str] = {
    "motorway": "major_road",
    "motorway_link": "major_road",
    "trunk": "major_road",
    "trunk_link": "major_road",
    "primary": "major_road",
    "primary_link": "major_road",
    "secondary": "road",
    "secondary_link": "road",
    "tertiary": "road",
    "tertiary_link": "road",
    "unclassified": "local_road",
    "residential": "local_road",
    "service": "local_road",
    "living_street": "local_road",
    "road": "road",
}

# Overpass QL: fetch only the highway types present in the default class_map
# plus their member nodes in one round-trip. Restricting to named tags is
# dramatically faster than `way["highway"]` with no value filter.
_HIGHWAY_TYPES = "|".join([
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential",
    "service", "living_street", "road",
])

_OVERPASS_QUERY = """\
[out:json][timeout:{server_timeout}];
(
  way["highway"~"^({highway_types})$"]({s},{w},{n},{e});
);
(._;>;);
out body;
"""


@dataclass
class OsmRoadsProcess:
    """Pull OSM road network via Overpass API for the raster's AOI.

    Caches each Overpass response under ``cache_dir`` keyed by a hash
    of the endpoint URL + bounding box, so subsequent runs with the
    same AOI are instant.
    """

    spec: ProcessSpec
    name: str = "osm-roads"
    overpass_url: str = _DEFAULT_OVERPASS_URL
    cache_dir: Path = field(default_factory=osm_roads_cache)
    class_map: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_CLASS_MAP))
    server_timeout: int = 180   # QL [timeout:N] — server-side processing budget
    timeout: int = 220          # HTTP socket timeout; must exceed server_timeout
    tile_size_deg: float = 0.1  # split AOI into ≤ this-deg-per-side tiles
    max_retries: int = 3        # per-tile retries on transient 5xx / timeouts
    max_workers: int = 8        # cap on concurrent Overpass tile fetches
    # Latched true when a DNS / connection-refused error is seen on any
    # tile. Subsequent tiles in the same run skip the network and serve
    # only from cache. Reset at the start of every `_fetch_or_load`.
    _offline: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_spec(cls, spec: ProcessSpec) -> "OsmRoadsProcess":
        kw = spec.kwargs
        # Allow partial override of the class map.
        class_map = dict(_DEFAULT_CLASS_MAP)
        user_map = kw.get("class_map") or {}
        if not isinstance(user_map, dict):
            raise ValueError("kwargs.class_map must be a dict[str, str]")
        class_map.update({str(k): str(v) for k, v in user_map.items()})
        return cls(
            spec=spec,
            overpass_url=str(kw.get("overpass_url", _DEFAULT_OVERPASS_URL)),
            cache_dir=Path(kw.get("cache_dir", osm_roads_cache())).expanduser(),
            class_map=class_map,
            server_timeout=int(kw.get("server_timeout", 180)),
            timeout=int(kw.get("timeout", 220)),
            tile_size_deg=float(kw.get("tile_size_deg", 0.1)),
            max_retries=int(kw.get("max_retries", 3)),
            max_workers=int(kw.get("max_workers", 8)),
        )

    def run(self, raster: Raster) -> list[Detection]:
        lon_min, lat_min, lon_max, lat_max = raster_roi_wgs84(raster)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        elements = self._fetch_or_load(lon_min, lat_min, lon_max, lat_max)

        # Build a node-id → (lon, lat) lookup from the flat element list.
        nodes: dict[int, tuple[float, float]] = {}
        ways: list[dict[str, Any]] = []
        for el in elements:
            t = el.get("type")
            if t == "node":
                nodes[el["id"]] = (float(el["lon"]), float(el["lat"]))
            elif t == "way":
                ways.append(el)

        _log.info(
            "osm-roads: %d nodes, %d ways in response", len(nodes), len(ways)
        )

        allow = set(self.spec.classes) if self.spec.classes is not None else None
        aoi_box = box(lon_min, lat_min, lon_max, lat_max)
        detections: list[Detection] = []
        idx = 0

        for way in ways:
            tags = way.get("tags") or {}
            hw = tags.get("highway", "")
            class_name = self.class_map.get(hw)
            if class_name is None:
                continue  # not in our map (footway, cycleway, steps, etc.)
            if allow is not None and class_name not in allow:
                continue

            coords = [
                nodes[nid]
                for nid in way.get("nodes", [])
                if nid in nodes
            ]
            if len(coords) < 2:
                continue

            # One detection per OSM way (the natural human unit of "a
            # road"). The way's polyline is clipped to the AOI bbox so
            # nothing renders off-raster; if the clip splits the way
            # into multiple disjoint pieces we emit a MultiLineString.
            way_line = LineString(coords)
            clipped = way_line.intersection(aoi_box)
            if clipped.is_empty:
                continue
            if not isinstance(clipped, (LineString, MultiLineString)):
                continue  # touches AOI at a point — not a road segment
            if clipped.length == 0:
                continue

            geo_bbox = clipped.bounds
            pixel_bbox = bbox_to_pixels(geo_bbox, raster)
            c0, r0, c1, r1 = pixel_bbox
            if c1 < c0 or r1 < r0 or (c1 == c0 and r1 == r0):
                continue

            centroid = clipped.centroid
            detections.append(
                Detection(
                    id=idx,
                    class_name=class_name,
                    confidence=1.0,  # vector data — deterministic
                    bbox=geo_bbox,
                    pixel_bbox=pixel_bbox,
                    centroid=Point(centroid.x, centroid.y),
                    source_model=self.name,
                    geometry=clipped,
                )
            )
            idx += 1

        _log.info("osm-roads: emitted %d detections", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_or_load(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
    ) -> list[dict[str, Any]]:
        """Return merged ``elements`` for the AOI, tiled and deduped by id."""
        tiles = _tile_bbox(
            lon_min, lat_min, lon_max, lat_max, self.tile_size_deg,
        )
        _log.info(
            "osm-roads: AOI split into %d tile(s) at %.3f° per side",
            len(tiles), self.tile_size_deg,
        )

        # Reset offline latch per run so a transient outage in a previous
        # run() doesn't permanently disable the network for this one.
        self._offline = False
        cache_hits = 0
        cache_misses = 0

        # Run cache-hit tiles inline first (no IO, no thread cost) so an all-
        # cached run stays single-threaded; remaining cache misses fan out
        # over a small thread pool. Order is preserved when deduping by id.
        merged: dict[tuple[str, int], dict[str, Any]] = {}
        cached_tiles: list[tuple[int, tuple[float, float, float, float]]] = []
        miss_tiles: list[tuple[int, tuple[float, float, float, float]]] = []
        for i, tile in enumerate(tiles, start=1):
            if self._tile_cache_path(*tile).exists():
                cached_tiles.append((i, tile))
            else:
                miss_tiles.append((i, tile))

        for i, (s, w, n, e) in cached_tiles:
            elements = self._fetch_or_load_tile(s, w, n, e, i, len(tiles))
            cache_hits += 1
            for el in elements:
                key = (el.get("type", ""), int(el.get("id", 0)))
                merged.setdefault(key, el)

        if miss_tiles:
            workers = max(1, min(self.max_workers, len(miss_tiles)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._fetch_or_load_tile, s, w, n, e, i, len(tiles),
                    ): i
                    for i, (s, w, n, e) in miss_tiles
                }
                for fut in as_completed(futures):
                    elements = fut.result()
                    if elements:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                    for el in elements:
                        key = (el.get("type", ""), int(el.get("id", 0)))
                        merged.setdefault(key, el)

        if self._offline:
            _log.info(
                "osm-roads: offline — served %d/%d tiles from cache, %d missing",
                cache_hits, len(tiles), cache_misses,
            )
        return list(merged.values())

    def _tile_cache_path(
        self, lat_min: float, lon_min: float, lat_max: float, lon_max: float,
    ) -> Path:
        bbox_str = f"{lat_min:.6f},{lon_min:.6f},{lat_max:.6f},{lon_max:.6f}"
        cache_key = hashlib.md5(
            f"{self.overpass_url}|{bbox_str}".encode()
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

    def _fetch_or_load_tile(
        self,
        lat_min: float,
        lon_min: float,
        lat_max: float,
        lon_max: float,
        tile_idx: int,
        tile_total: int,
    ) -> list[dict[str, Any]]:
        """Fetch (or read cache for) a single tile."""
        bbox_str = f"{lat_min:.6f},{lon_min:.6f},{lat_max:.6f},{lon_max:.6f}"
        cache_path = self._tile_cache_path(lat_min, lon_min, lat_max, lon_max)

        if cache_path.exists():
            _log.info(
                "osm-roads: [%d/%d] cache hit %s",
                tile_idx, tile_total, cache_path.name,
            )
            return json.loads(cache_path.read_text()).get("elements", [])

        # Network already known unreachable for this run — skip the call,
        # accept partial coverage. Logged at warning so the gap is visible
        # without making every miss a separate alarm.
        if self._offline:
            _log.warning(
                "osm-roads: [%d/%d] offline — cache miss for tile %s, skipping",
                tile_idx, tile_total, bbox_str,
            )
            return []

        query = _OVERPASS_QUERY.format(
            s=lat_min,
            w=lon_min,
            n=lat_max,
            e=lon_max,
            server_timeout=self.server_timeout,
            highway_types=_HIGHWAY_TYPES,
        )
        body = urllib.parse.urlencode({"data": query}).encode("utf-8")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            _log.info(
                "osm-roads: [%d/%d] query attempt %d (bbox %s)",
                tile_idx, tile_total, attempt, bbox_str,
            )
            req = urllib.request.Request(
                self.overpass_url,
                data=body,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "terrascope-cli/0.1 (research)",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                try:
                    cache_path.write_bytes(raw)
                except OSError as cache_exc:
                    _log.warning(
                        "osm-roads: failed to write tile cache %s: %s — "
                        "continuing without cache",
                        cache_path, cache_exc,
                    )
                result = json.loads(raw)
                elements: list[dict[str, Any]] = result.get("elements", [])
                _log.info(
                    "osm-roads: [%d/%d] fetched %d elements",
                    tile_idx, tile_total, len(elements),
                )
                return elements
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                # 4xx (other than 429) is not worth retrying.
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500 and exc.code != 429:
                    raise
                # DNS / connection-refused: the network is unreachable.
                # Latch offline mode and fall back to cache-only for the
                # rest of the run instead of waiting through retries.
                if _is_offline_error(exc):
                    self._offline = True
                    _log.warning(
                        "osm-roads: [%d/%d] network unreachable (%s) — "
                        "falling back to cache-only for the rest of this run",
                        tile_idx, tile_total, exc,
                    )
                    return []
                last_err = exc
                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1) * 5  # 5s, 10s, 20s, ...
                    _log.warning(
                        "osm-roads: [%d/%d] %s — retrying in %ds",
                        tile_idx, tile_total, exc, backoff,
                    )
                    time.sleep(backoff)
        assert last_err is not None
        raise last_err


def _is_offline_error(exc: BaseException) -> bool:
    """True for errors that mean "the network is unreachable" (vs transient).

    DNS failures and connection refusal are not going to recover within
    a 35-second backoff window — there's no point retrying. 5xx, 429,
    and generic timeouts are still treated as transient by the caller.
    """
    if isinstance(exc, (socket.gaierror, socket.herror, ConnectionRefusedError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (socket.gaierror, socket.herror, ConnectionRefusedError)):
            return True
    return False


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _tile_bbox(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    tile_size_deg: float,
) -> list[tuple[float, float, float, float]]:
    """Split a WGS84 bbox into ``(s, w, n, e)`` tiles ≤ ``tile_size_deg`` per side.

    Returns at least one tile (the full bbox) when the AOI is smaller than
    ``tile_size_deg``. Tile size is the *maximum* side length; the actual
    grid step is computed so all tiles are equal-sized.
    """
    if tile_size_deg <= 0:
        raise ValueError("tile_size_deg must be > 0")
    width = lon_max - lon_min
    height = lat_max - lat_min
    if width <= 0 or height <= 0:
        return [(lat_min, lon_min, lat_max, lon_max)]
    nx = max(1, math.ceil(width / tile_size_deg))
    ny = max(1, math.ceil(height / tile_size_deg))
    dx = width / nx
    dy = height / ny
    tiles: list[tuple[float, float, float, float]] = []
    for j in range(ny):
        s = lat_min + j * dy
        n = lat_min + (j + 1) * dy if j < ny - 1 else lat_max
        for i in range(nx):
            w = lon_min + i * dx
            e = lon_min + (i + 1) * dx if i < nx - 1 else lon_max
            tiles.append((s, w, n, e))
    return tiles


register("osm-roads", OsmRoadsProcess.from_spec)

__all__ = ["OsmRoadsProcess"]
