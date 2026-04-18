"""Tests for CLI commands."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
from shapely.geometry import box
from typer.testing import CliRunner

from cli.main import app
from core.services.detector import RawDetection
from core.services.indicators import ZoneIndicatorResult

runner = CliRunner()


# --- process command tests ---


def test_process_local_runs_pipeline(tmp_path):
    """Test local processing pipeline with mocked services."""
    input_tif = tmp_path / "input.tif"
    input_tif.touch()

    aoi_path = tmp_path / "aoi.geojson"
    aoi_geojson = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    aoi_path.write_text(json.dumps(aoi_geojson))

    output_dir = tmp_path / "output"

    mock_data = np.zeros((3, 100, 100), dtype=np.float32)
    mock_transform = MagicMock()
    mock_dataset = MagicMock()

    mock_detection = RawDetection(
        class_name="building",
        confidence=85,
        geometry=box(0.1, 0.1, 0.2, 0.2),
        source="test",
    )
    mock_indicator = ZoneIndicatorResult(
        zone_id="aoi",
        class_name="building",
        count=1,
        density_per_km2=10.0,
        total_area_m2=500.0,
    )

    with (
        patch("core.services.imagery.ImageryLoaderService") as MockLoader,
        patch("core.services.tiler.TilerService") as MockTiler,
        patch("core.services.detector.DetectorService") as MockDetector,
        patch("core.services.postprocessor.PostprocessorService") as MockPP,
        patch("core.services.exporter.GISExporterService") as MockExporter,
        patch("core.services.indicators.IndicatorCalculatorService") as MockCalc,
        patch("cli.commands.process.settings") as mock_settings,
    ):
        mock_settings.tile_size = 512
        mock_settings.tile_overlap = 64
        mock_settings.device = "cpu"
        mock_settings.nms_iou_threshold = 0.5
        mock_settings.confidence_threshold = 50
        mock_settings.min_area_m2 = 10.0
        mock_settings.max_area_m2 = 1_000_000.0
        mock_settings.simplify_tolerance_m = 1.0

        loader = MockLoader.return_value
        loader.load.return_value = mock_dataset
        loader.get_metadata.return_value = {
            "band_count": 3,
            "crs": "EPSG:4326",
            "bounds": {},
            "resolution": (10, 10),
        }
        loader.clip_to_aoi.return_value = (mock_data, mock_transform, "EPSG:4326")

        mock_tile = MagicMock()
        MockTiler.return_value.generate_tiles.return_value = [mock_tile]

        MockDetector.return_value.predict_tile.return_value = [mock_detection]

        MockPP.return_value.run.return_value = (
            [mock_detection],
            {"input_count": 1, "output_count": 1},
        )

        MockExporter.return_value.export_all.return_value = {
            "geojson": output_dir / "detections.geojson",
        }

        MockCalc.return_value.compute.return_value = [mock_indicator]

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
        assert "Tiling imagery" in result.output
        assert "Running detection" in result.output
        assert "Post-processing" in result.output
        assert "Exporting results" in result.output
        assert "Computing indicators" in result.output
        assert "Done!" in result.output

        loader.load.assert_called_once()
        MockDetector.return_value.load_models.assert_called_once()
        MockPP.return_value.run.assert_called_once()
        MockExporter.return_value.export_all.assert_called_once()
        MockCalc.return_value.compute.assert_called_once()


def test_process_temporal_flag(tmp_path):
    """Test that --use-temporal calls _run_temporal."""
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


# --- stac command tests ---


def test_stac_search():
    """Test STAC search command."""
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


def test_stac_search_invalid_bbox():
    """Test STAC search with invalid bbox."""
    result = runner.invoke(
        app,
        ["stac", "search", "--bbox", "10.0,20.0,11.0"],
    )

    assert result.exit_code == 1


def test_stac_download(tmp_path):
    """Test STAC download command."""
    mock_item = MagicMock()
    mock_item.id = "S2A_MSIL2A_20240101"

    downloaded = tmp_path / "downloaded.tif"

    with patch("cli.commands.stac.StacService") as MockStac:

        async def mock_search(*args, **kwargs):
            return [mock_item]

        async def mock_download(*args, **kwargs):
            return downloaded

        MockStac.return_value.search = mock_search
        MockStac.return_value.download = mock_download

        result = runner.invoke(
            app,
            [
                "stac",
                "download",
                "--item-id",
                "S2A_MSIL2A_20240101",
                "--bbox",
                "10.0,20.0,11.0,21.0",
                "--output",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Downloaded to:" in result.output


# --- evaluate command tests ---


def test_evaluate_stub():
    """Test evaluate command shows stub message."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--predictions",
            "/fake/preds.geojson",
            "--ground-truth",
            "/fake/gt.geojson",
        ],
    )

    assert result.exit_code == 0
    assert "Phase 9" in result.output


# --- worker command tests ---


def test_worker_help():
    """Test worker command help text."""
    result = runner.invoke(app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "Temporal worker" in result.output
