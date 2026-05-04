"""Tests for OsmRoadsProcess offline-fallback behavior.

We don't hit the real Overpass API. Instead we either:
- pre-populate the cache_dir with synthetic tile JSONs (cache-hit path), or
- patch `urllib.request.urlopen` to raise a chosen exception (network path),
and verify the process latches into "offline" mode on DNS / connection
errors instead of waiting through exponential backoff.
"""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rasterio.transform import Affine

from core.detection.types import Raster

from core2 import ProcessSpec
from core2.processes.osm_roads import OsmRoadsProcess, _is_offline_error


def _raster_two_tiles() -> Raster:
    """Raster spanning ~2x1 tiles at the default tile_size_deg=0.1.

    Footprint 0..0.21 lon × 50..50.05 lat → splits into 3 columns x 1 row
    when tile_size_deg=0.1, but we override tile_size_deg to 0.1 below
    so the math stays predictable.
    """
    deg_lon = 0.21
    deg_lat = 0.05
    px_w, px_h = 210, 50
    transform = Affine(deg_lon / px_w, 0, 0.0, 0, -deg_lat / px_h, 50.05)
    return Raster(
        data=np.zeros((px_h, px_w, 3), dtype=np.uint8),
        transform=transform,
        crs="EPSG:4326",
    )


def _populate_tile(
    cache_dir: Path,
    overpass_url: str,
    s: float, w: float, n: float, e: float,
    elements: list[dict],
) -> Path:
    """Write a fake Overpass response into the cache."""
    bbox_str = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    key = hashlib.md5(f"{overpass_url}|{bbox_str}".encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"elements": elements}))
    return path


def _spec(cache_dir: Path, **kwargs) -> ProcessSpec:
    kw = {"cache_dir": str(cache_dir), "tile_size_deg": 0.1, **kwargs}
    return ProcessSpec(name="osm-roads", kwargs=kw)


def test_is_offline_error_predicate():
    assert _is_offline_error(socket.gaierror(8, "no such host"))
    assert _is_offline_error(ConnectionRefusedError())
    assert _is_offline_error(
        urllib.error.URLError(socket.gaierror(8, "no such host"))
    )
    # 503 is transient, not offline.
    assert not _is_offline_error(
        urllib.error.HTTPError("u", 503, "boom", {}, None)
    )
    # Bare timeout is not "offline" — server may just be slow.
    assert not _is_offline_error(TimeoutError())


def test_dns_failure_latches_offline_and_serves_cache(tmp_path: Path):
    """Uncached tile + DNS error → latch offline, no retries, no sleep."""
    cache_dir = tmp_path / "cache"
    overpass_url = "https://overpass.example/api"

    # Pre-populate the FIRST tile (lon 0.0..0.07, lat 50.0..50.05 once
    # tile_size_deg=0.1 splits the 0.21° width into 3 equal columns of
    # 0.07°). The other two tiles will miss → trigger a network call.
    way_id = 1001
    _populate_tile(
        cache_dir, overpass_url,
        s=50.00, w=0.00, n=50.05, e=0.07,
        elements=[
            {"type": "node", "id": 1, "lon": 0.01, "lat": 50.01},
            {"type": "node", "id": 2, "lon": 0.05, "lat": 50.04},
            {
                "type": "way", "id": way_id,
                "nodes": [1, 2],
                "tags": {"highway": "residential"},
            },
        ],
    )

    spec = _spec(cache_dir, overpass_url=overpass_url, max_retries=3)
    proc = OsmRoadsProcess.from_spec(spec)

    def _raise_dns(*_a, **_kw):
        raise urllib.error.URLError(socket.gaierror(8, "no such host"))

    with patch("urllib.request.urlopen", side_effect=_raise_dns) as urlopen, \
         patch("time.sleep") as sleep:
        dets = proc.run(_raster_two_tiles())

    # Offline latched, sleep never called (no exponential backoff).
    assert proc._offline is True
    sleep.assert_not_called()
    # urlopen called at most once (the first cache miss); subsequent
    # tiles short-circuit via the latch.
    assert urlopen.call_count <= 1
    # The cached tile contributed a road.
    assert any(d.source_model == "osm-roads" for d in dets)


def test_transient_5xx_still_retries_and_does_not_latch(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    spec = _spec(cache_dir, max_retries=2)  # keep the test quick
    proc = OsmRoadsProcess.from_spec(spec)

    def _raise_503(*_a, **_kw):
        raise urllib.error.HTTPError(
            "u", 503, "service unavailable", {}, None,
        )

    with patch("urllib.request.urlopen", side_effect=_raise_503), \
         patch("time.sleep") as sleep, \
         pytest.raises(urllib.error.HTTPError):
        proc.run(_raster_two_tiles())

    # 5xx is transient → backoff sleep DID happen (max_retries-1 times
    # per failing tile; we only need to know it happened at least once).
    assert sleep.called
    # Did NOT latch into offline mode.
    assert proc._offline is False


def test_offline_latch_resets_between_runs(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    spec = _spec(cache_dir, max_retries=1)
    proc = OsmRoadsProcess.from_spec(spec)

    # First run: DNS error → latch sets.
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError(socket.gaierror(8, "x")),
    ), patch("time.sleep"):
        proc.run(_raster_two_tiles())
    assert proc._offline is True

    # Second run: latch must reset before the per-tile loop, otherwise
    # the network would never be retried even after connectivity comes
    # back. We check the reset directly by populating ALL tiles to
    # cache (no urlopen needed) and asserting the latch is False after.
    # Pre-populate every tile so the second run is fully cached.
    overpass_url = "https://overpass-api.de/api/interpreter"  # default
    for i in range(3):
        s, n = 50.00, 50.05
        w = i * (0.21 / 3)
        e = (i + 1) * (0.21 / 3)
        _populate_tile(
            cache_dir, overpass_url, s, w, n, e, elements=[],
        )
    # Use a fresh spec without cache_dir override so the URL matches default.
    spec2 = ProcessSpec(
        name="osm-roads",
        kwargs={"cache_dir": str(cache_dir), "tile_size_deg": 0.1},
    )
    proc2 = OsmRoadsProcess.from_spec(spec2)
    proc2._offline = True  # simulate a stale latch from a prior run
    with patch("urllib.request.urlopen") as urlopen:
        proc2.run(_raster_two_tiles())
    # Cache hits everywhere → urlopen should not be called.
    urlopen.assert_not_called()
    # And the latch reset at the start of _fetch_or_load.
    assert proc2._offline is False
