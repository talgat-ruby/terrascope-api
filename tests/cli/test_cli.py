"""Tests for CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import box
from typer.testing import CliRunner

from cli.main import app
from core.detection.types import Detection, Raster

runner = CliRunner()


def _make_raster() -> Raster:
    return Raster(
        data=np.zeros((100, 100, 3), dtype=np.uint8),
        transform=Affine(0.0001, 0, 0, 0, -0.0001, 1),
        crs="EPSG:4326",
        aoi_geom=box(0, 0, 1, 1),
    )


def _make_detection() -> Detection:
    bnds = (0.1, 0.1, 0.2, 0.2)
    return Detection(
        id=0,
        class_name="car",
        confidence=0.85,
        bbox=bnds,
        pixel_bbox=(10, 10, 30, 30),
        centroid=box(*bnds).centroid,
    )


def test_process_local_runs_pipeline(tmp_path: Path):
    input_tif = tmp_path / "input.tif"
    input_tif.touch()
    aoi_path = tmp_path / "aoi.geojson"
    aoi_path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        )
    )
    output_dir = tmp_path / "output"

    raster = _make_raster()
    detection = _make_detection()

    stub_detector = MagicMock()
    stub_detector.name = "stub"
    stub_detector.detect.return_value = [detection]

    with (
        patch("core.services.imagery.ImageryLoaderService") as MockLoader,
        patch(
            "core.detection.build_detector", return_value=stub_detector
        ) as MockBuild,
        patch("core.detection.render_overlay") as MockRender,
        patch("core.services.exporter.GISExporterService") as MockExporter,
        patch("core.services.indicators.IndicatorCalculatorService") as MockCalc,
    ):
        MockLoader.return_value.load_clipped.return_value = raster
        MockExporter.return_value.export_geojson.return_value = (
            output_dir / "detections.geojson"
        )
        MockCalc.return_value.compute.return_value = []

        result = runner.invoke(
            app,
            [
                "process",
                "--input",
                str(input_tif),
                "--aoi",
                str(aoi_path),
                "--output",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Loading imagery" in result.output
    assert "Running detection" in result.output
    assert "Done." in result.output
    MockBuild.assert_called_once()
    stub_detector.detect.assert_called_once()
    MockRender.assert_called_once()
    MockExporter.return_value.export_geojson.assert_called_once()


def test_process_temporal_flag(tmp_path: Path):
    aoi_path = tmp_path / "aoi.geojson"
    aoi_path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        )
    )
    input_tif = tmp_path / "input.tif"
    input_tif.touch()

    with patch("cli.commands.process.asyncio") as mock_asyncio:
        mock_asyncio.run = MagicMock(return_value=None)

        result = runner.invoke(
            app,
            [
                "process",
                "--input",
                str(input_tif),
                "--aoi",
                str(aoi_path),
                "--use-temporal",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_asyncio.run.assert_called_once()


def test_stac_search():
    mock_item = MagicMock()
    mock_item.id = "S2A_MSIL2A_20240101"
    mock_item.datetime = MagicMock()
    mock_item.datetime.isoformat.return_value = "2024-01-01T00:00:00Z"
    mock_item.assets = {"visual": MagicMock(), "thumbnail": MagicMock()}

    with patch("cli.commands.stac.StacService") as MockStac:

        async def mock_search(*args, **kwargs):
            return [mock_item]

        MockStac.return_value.search = mock_search

        result = runner.invoke(
            app,
            [
                "stac",
                "search",
                "--bbox",
                "10.0,20.0,11.0,21.0",
                "--datetime",
                "2024-01-01/2024-06-01",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Found 1 items" in result.output
    assert "S2A_MSIL2A_20240101" in result.output


def test_worker_help():
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "Temporal worker" in result.output
