"""Tests for cli error paths exposed via the typer Runner.

Covers the error-handling branches added on top of the pure resolver:
- malformed JSON in --config and --processes
- missing AOI file
- invalid GeoJSON in AOI
- unknown process name
- list-processes side-effect (no other side effects).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def _write(path: Path, payload: object) -> Path:
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload))
    return path


def test_list_processes_includes_builtins():
    result = runner.invoke(app, ["--list-processes"])
    assert result.exit_code == 0
    out = result.stdout
    for name in ("msft-buildings", "osm-roads", "yolov8-satellite-vehicle"):
        assert name in out


def test_malformed_config_json_exits_cleanly(tmp_path: Path):
    cfg = _write(tmp_path / "bad.json", "{not valid json")
    result = runner.invoke(app, ["--config", str(cfg)])
    assert result.exit_code == 2
    assert "not valid JSON" in result.output


def test_malformed_processes_flag_exits_cleanly(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "--input", str(tmp_path / "in.tif"),
            "--processes", "{not valid",
        ],
    )
    assert result.exit_code == 2
    assert "not valid JSON" in result.output


def test_unknown_process_exits_cleanly(tmp_path: Path):
    # Real raster file — load_raster runs before the unknown-process error.
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    tif = tmp_path / "in.tif"
    with rasterio.open(
        tif, "w",
        driver="GTiff", height=8, width=8, count=3,
        dtype="uint8", crs="EPSG:4326",
        transform=from_bounds(10.0, 49.9, 10.1, 50.0, 8, 8),
    ) as dst:
        dst.write(np.zeros((3, 8, 8), dtype=np.uint8))

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "--input", str(tif),
            "--output", str(out),
            "--processes", json.dumps([{"name": "no-such-process"}]),
        ],
    )
    assert result.exit_code == 2
    assert "Unknown process" in result.output


def test_missing_aoi_file_exits_cleanly(tmp_path: Path):
    cfg = _write(
        tmp_path / "job.json",
        {
            "input": "raster.tif",
            "aoi": "no-such-aoi.geojson",
            "processes": [{"name": "msft-buildings"}],
        },
    )
    # Stamp an existing input so we get past the input check.
    (tmp_path / "raster.tif").write_bytes(b"")
    result = runner.invoke(app, ["--config", str(cfg)])
    assert result.exit_code == 2
    # The AOI lookup happens after resolution; it falls into _load_aoi which
    # checks .exists() up front and emits the AOI-specific error.
    assert "AOI file not found" in result.output


def test_invalid_aoi_geometry_exits_cleanly(tmp_path: Path):
    _write(tmp_path / "aoi.geojson", {"type": "Polygon"})  # missing coords
    cfg = _write(
        tmp_path / "job.json",
        {
            "input": "raster.tif",
            "aoi": "aoi.geojson",
            "processes": [{"name": "msft-buildings"}],
        },
    )
    (tmp_path / "raster.tif").write_bytes(b"")
    result = runner.invoke(app, ["--config", str(cfg)])
    assert result.exit_code == 2
    assert "AOI" in result.output
