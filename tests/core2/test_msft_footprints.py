"""Tests for MsftFootprintProcess that don't hit the network.

Strategy: pre-populate the cache_dir with a tiny synthetic dataset-links.csv
and a gzipped JSON-Lines shard, then run `.run(raster)` and assert outputs.
"""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

from core.detection.types import Raster

from core2 import ProcessSpec, build
from core2.processes.msft_footprints import _qk_bounds, _shards_overlapping


def _raster_around(lon: float, lat: float) -> Raster:
    """1m-ish per-pixel raster centered around (lon, lat)."""
    deg = 0.001  # ~100m
    transform = Affine(deg / 100, 0, lon - deg / 2, 0, -deg / 100, lat + deg / 2)
    return Raster(
        data=np.zeros((100, 100, 3), dtype=np.uint8),
        transform=transform,
        crs="EPSG:4326",
    )


def test_qk_bounds_matches_inverse():
    qk = "1230"
    lon_min, lat_min, lon_max, lat_max = _qk_bounds(qk)
    assert lon_min < lon_max
    assert lat_min < lat_max
    assert -180 <= lon_min <= 180
    assert -math.degrees(math.atan(math.sinh(math.pi))) <= lat_min <= 90


def test_qk_bounds_rejects_invalid_chars():
    with pytest.raises(ValueError):
        _qk_bounds("1234")


def test_shards_overlapping_filters_by_country_and_roi(tmp_path: Path):
    # Two quadkeys near (51, 71) — one we'll use, one we won't.
    # quadkey for Astana area is roughly "1202..." at zoom 9.
    # We'll pick two synthetic qks of the same zoom level and filter on country.
    csv = tmp_path / "dataset-links.csv"
    csv.write_text(
        "Location,QuadKey,Url,Size\n"
        "Kazakhstan,120322232,https://example/used.csv.gz,1MB\n"
        "Kazakhstan,000000000,https://example/elsewhere.csv.gz,1MB\n"
        "Russia,120322232,https://example/wrong-country.csv.gz,1MB\n"
    )
    qk_bounds = _qk_bounds("120322232")
    from shapely.geometry import box as _box
    roi = _box(*qk_bounds).buffer(-0.001)  # safely inside the qk

    shards = _shards_overlapping(csv, "Kazakhstan", roi)
    assert "120322232" in shards
    assert "000000000" not in shards
    assert all("wrong-country" not in url for url in shards.values())


def test_msft_run_clips_to_roi_and_filters_small(tmp_path: Path):
    lon, lat = 71.43, 51.13
    raster = _raster_around(lon, lat)

    # Pick a real quadkey covering the AOI by walking the quadkey tree.
    def _qk_for(lon: float, lat: float, zoom: int = 9) -> str:
        x = int((lon + 180) / 360 * (2**zoom))
        sin_lat = math.sin(math.radians(lat))
        y = int(
            (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi))
            * (2**zoom)
        )
        qk = ""
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if x & mask:
                digit += 1
            if y & mask:
                digit += 2
            qk += str(digit)
        return qk

    qk = _qk_for(lon, lat, zoom=9)

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dataset-links.csv").write_text(
        "Location,QuadKey,Url,Size\n"
        f"Kazakhstan,{qk},https://example/{qk}.csv.gz,1MB\n"
    )

    # One ~50m² building inside ROI; one tiny 1m² building that should be
    # filtered out by min_area_m2; one building well outside ROI.
    inside_big = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - 0.0001, lat - 0.0001],
                [lon + 0.0001, lat - 0.0001],
                [lon + 0.0001, lat + 0.0001],
                [lon - 0.0001, lat + 0.0001],
                [lon - 0.0001, lat - 0.0001],
            ]],
        },
        "properties": {},
    }
    inside_tiny = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon, lat],
                [lon + 0.0000005, lat],
                [lon + 0.0000005, lat + 0.0000005],
                [lon, lat + 0.0000005],
                [lon, lat],
            ]],
        },
        "properties": {},
    }
    outside = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon + 1, lat + 1],
                [lon + 1.001, lat + 1],
                [lon + 1.001, lat + 1.001],
                [lon + 1, lat + 1.001],
                [lon + 1, lat + 1],
            ]],
        },
        "properties": {},
    }

    shard = cache / f"{qk}.csv.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as f:
        for feat in (inside_big, inside_tiny, outside):
            f.write(json.dumps(feat) + "\n")

    spec = ProcessSpec(
        name="msft-buildings",
        kwargs={
            "country": "Kazakhstan",
            "min_area_m2": 5.0,
            "cache_dir": str(cache),
            "index_url": "https://example/should-not-be-fetched.csv",
        },
    )
    proc = build(spec)
    out = proc.run(raster)

    assert all(d.class_name == "building" for d in out)
    assert all(d.source_model == "msft-buildings" for d in out)
    assert all(d.confidence == 1.0 for d in out)
    assert len(out) == 1  # outside dropped, tiny dropped by min_area_m2
