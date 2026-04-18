"""Tests for Temporal activity implementations."""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import box

from core.models.detection import Detection
from core.models.processing import JobStatus, ProcessingJob
from core.models.territory import Territory
from core.services.detector import RawDetection


def _make_job(
    job_id: str | None = None,
    status: JobStatus = JobStatus.PENDING,
    config: dict | None = None,
    checkpoint_data: dict | None = None,
) -> ProcessingJob:
    """Create a mock ProcessingJob."""
    return ProcessingJob(
        id=uuid.UUID(job_id) if job_id else uuid.uuid4(),
        status=status,
        input_path="/data/test.tif",
        config=config
        or {
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            "aoi_crs": "EPSG:4326",
        },
        checkpoint_data=checkpoint_data,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_raw_detection() -> RawDetection:
    return RawDetection(
        class_name="building",
        confidence=85,
        geometry=box(0.1, 0.1, 0.2, 0.2),
        source="test",
    )


@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    """Patch async_session_factory to return mock session."""
    with patch(
        "worker.activities._helpers.async_session_factory",
        return_value=mock_session,
    ) as factory:
        yield factory


def _setup_job_query(mock_session, job):
    """Configure the mock session to return a job on query."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=result_mock)


# --- load_imagery tests ---


@pytest.mark.asyncio
async def test_load_imagery_success(mock_session, tmp_path):
    job = _make_job()
    _setup_job_query(mock_session, job)

    mock_dataset = MagicMock()
    mock_dataset.crs = "EPSG:4326"
    mock_metadata = {
        "band_count": 3,
        "crs": "EPSG:4326",
        "bounds": {},
        "resolution": (10, 10),
    }
    mock_data = np.zeros((3, 100, 100), dtype=np.float32)
    mock_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    with (
        patch(
            "worker.activities.imagery.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.imagery.ImageryLoaderService") as MockLoader,
        patch("worker.activities.imagery.settings") as mock_settings,
        patch("worker.activities.imagery.np") as mock_np,
    ):
        loader_instance = MockLoader.return_value
        loader_instance.load.return_value = mock_dataset
        loader_instance.get_metadata.return_value = mock_metadata
        loader_instance.clip_to_aoi.return_value = (
            mock_data,
            mock_transform,
            "EPSG:4326",
        )

        mock_settings.output_dir = tmp_path
        mock_np.save = MagicMock()

        from worker.activities.imagery import load_imagery

        result = await load_imagery(str(job.id))

        assert result["status"] == "loaded"
        assert result["crs"] == "EPSG:4326"
        loader_instance.load.assert_called_once_with(job.input_path)
        loader_instance.clip_to_aoi.assert_called_once()
        mock_dataset.close.assert_called_once()


# --- tile_imagery tests ---


@pytest.mark.asyncio
async def test_tile_imagery_success(mock_session, tmp_path):
    job = _make_job(
        checkpoint_data={
            "load": {
                "clipped_path": str(tmp_path / "clipped.npy"),
                "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 100.0],
                "crs": "EPSG:4326",
                "shape": [3, 100, 100],
            }
        }
    )
    _setup_job_query(mock_session, job)

    mock_data = np.zeros((3, 100, 100), dtype=np.float32)

    mock_tile = MagicMock()
    mock_tile.index = (0, 0)
    mock_tile.pixel_window = (0, 0, 100, 100)
    mock_tile.transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    mock_tile.crs = "EPSG:4326"
    mock_tile.data = np.zeros((3, 512, 512), dtype=np.float32)
    mock_tile.valid_mask = np.ones((512, 512), dtype=bool)

    with (
        patch(
            "worker.activities.imagery.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.imagery.TilerService") as MockTiler,
        patch("worker.activities.imagery.settings") as mock_settings,
        patch("worker.activities.imagery.np") as mock_np,
    ):
        mock_np.load.return_value = mock_data
        mock_np.save = MagicMock()
        tiler_instance = MockTiler.return_value
        tiler_instance.generate_tiles.return_value = iter([mock_tile])

        mock_settings.output_dir = tmp_path
        mock_settings.tile_size = 512
        mock_settings.tile_overlap = 64

        from worker.activities.imagery import tile_imagery

        result = await tile_imagery(str(job.id))

        assert result["status"] == "tiled"
        assert result["tile_count"] == 1


# --- detect_objects tests ---


@pytest.mark.asyncio
async def test_detect_objects_success(mock_session, tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    # Write tile data files
    np.save(
        str(tiles_dir / "tile_0_0_data.npy"), np.zeros((3, 512, 512), dtype=np.float32)
    )
    np.save(str(tiles_dir / "tile_0_0_mask.npy"), np.ones((512, 512), dtype=bool))

    manifest = [
        {
            "index": [0, 0],
            "pixel_window": [0, 0, 512, 512],
            "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 100.0],
            "crs": "EPSG:4326",
            "name": "tile_0_0",
        }
    ]
    manifest_path = tiles_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    job = _make_job(
        checkpoint_data={
            "load": {"crs": "EPSG:4326"},
            "tile": {
                "tile_count": 1,
                "manifest_path": str(manifest_path),
                "tiles_dir": str(tiles_dir),
            },
        }
    )
    _setup_job_query(mock_session, job)

    raw_det = _make_raw_detection()

    with (
        patch(
            "worker.activities.detection.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.detection.DetectorService") as MockDetector,
        patch("worker.activities.detection.settings") as mock_settings,
    ):
        mock_settings.device = "cpu"
        detector_instance = MockDetector.return_value
        detector_instance.predict_tile.return_value = [raw_det]

        from worker.activities.detection import detect_objects

        result = await detect_objects(str(job.id))

        assert result["status"] == "detected"
        assert result["detection_count"] == 1
        detector_instance.load_models.assert_called_once()
        detector_instance.predict_tile.assert_called_once()
        mock_session.add_all.assert_called()


# --- postprocess tests ---


@pytest.mark.asyncio
async def test_postprocess_success(mock_session):
    job = _make_job(
        checkpoint_data={
            "load": {"crs": "EPSG:4326"},
            "detect": {"raw_detection_count": 5},
        }
    )
    _setup_job_query(mock_session, job)

    mock_db_detection = MagicMock(spec=Detection)
    mock_db_detection.class_name = "building"
    mock_db_detection.confidence = 85
    mock_db_detection.source = "test"
    mock_db_detection.geometry = MagicMock()

    raw_det = _make_raw_detection()
    stats = {"input_count": 1, "output_count": 1, "after_nms": 1}

    # Setup mock to return detections on second execute call
    result_job = MagicMock()
    result_job.scalar_one_or_none.return_value = job
    result_detections = MagicMock()
    result_detections.scalars.return_value.all.return_value = [mock_db_detection]

    mock_session.execute = AsyncMock(
        side_effect=[result_job, result_job, result_detections, MagicMock(), result_job]
    )

    with (
        patch(
            "worker.activities.postprocessing.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.postprocessing.PostprocessorService") as MockPP,
        patch("worker.activities.postprocessing.detections_to_raw") as mock_to_raw,
        patch("worker.activities.postprocessing.raw_to_detections") as mock_to_db,
        patch("worker.activities.postprocessing.settings") as mock_settings,
    ):
        mock_to_raw.return_value = [raw_det]
        mock_to_db.return_value = [mock_db_detection]

        mock_settings.nms_iou_threshold = 0.5
        mock_settings.confidence_threshold = 50
        mock_settings.min_area_m2 = 10.0
        mock_settings.max_area_m2 = 1_000_000.0
        mock_settings.simplify_tolerance_m = 1.0

        pp_instance = MockPP.return_value
        pp_instance.run.return_value = ([raw_det], stats)

        from worker.activities.postprocessing import postprocess

        result = await postprocess(str(job.id))

        assert result["status"] == "postprocessed"
        assert result["stats"] == stats
        pp_instance.run.assert_called_once()


# --- export_results tests ---


@pytest.mark.asyncio
async def test_export_results_success(mock_session, tmp_path):
    job = _make_job(
        checkpoint_data={
            "load": {"crs": "EPSG:4326"},
            "postprocess": {"output_count": 1},
        }
    )
    _setup_job_query(mock_session, job)

    mock_db_detection = MagicMock(spec=Detection)
    mock_db_detection.class_name = "building"
    mock_db_detection.confidence = 85
    mock_db_detection.source = "test"
    mock_db_detection.geometry = MagicMock()

    raw_det = _make_raw_detection()

    result_job = MagicMock()
    result_job.scalar_one_or_none.return_value = job
    result_detections = MagicMock()
    result_detections.scalars.return_value.all.return_value = [mock_db_detection]

    mock_session.execute = AsyncMock(
        side_effect=[result_job, result_job, result_detections, result_job]
    )

    export_paths = {
        "geojson": tmp_path / "detections.geojson",
        "gpkg": tmp_path / "detections.gpkg",
        "shp": tmp_path / "detections.shp",
    }

    with (
        patch(
            "worker.activities.export.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.export.GISExporterService") as MockExporter,
        patch("worker.activities.export.detections_to_raw") as mock_to_raw,
        patch("worker.activities.export.settings") as mock_settings,
    ):
        mock_to_raw.return_value = [raw_det]
        mock_settings.output_dir = tmp_path

        exporter_instance = MockExporter.return_value
        exporter_instance.export_all.return_value = export_paths

        from worker.activities.export import export_results

        result = await export_results(str(job.id))

        assert result["status"] == "exported"
        assert "geojson" in result["formats"]
        exporter_instance.export_all.assert_called_once()


# --- compute_indicators tests ---


@pytest.mark.asyncio
async def test_compute_indicators_success(mock_session):
    job = _make_job(
        checkpoint_data={
            "load": {"crs": "EPSG:4326"},
            "postprocess": {"output_count": 1},
        }
    )
    _setup_job_query(mock_session, job)

    mock_db_detection = MagicMock(spec=Detection)
    raw_det = _make_raw_detection()

    mock_territory = MagicMock(spec=Territory)
    mock_territory.id = uuid.uuid4()
    mock_territory.geometry = MagicMock()

    result_job = MagicMock()
    result_job.scalar_one_or_none.return_value = job
    result_detections = MagicMock()
    result_detections.scalars.return_value.all.return_value = [mock_db_detection]
    result_territories = MagicMock()
    result_territories.scalars.return_value.all.return_value = [mock_territory]

    mock_session.execute = AsyncMock(
        side_effect=[
            result_job,
            result_job,
            result_detections,
            result_territories,
            result_job,
        ]
    )

    from core.services.indicators import ZoneIndicatorResult

    mock_indicator = ZoneIndicatorResult(
        zone_id=str(mock_territory.id),
        class_name="building",
        count=1,
        density_per_km2=10.0,
        total_area_m2=500.0,
    )

    with (
        patch(
            "worker.activities.indicators.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.indicators.IndicatorCalculatorService") as MockCalc,
        patch("worker.activities.indicators.detections_to_raw") as mock_to_raw,
        patch("worker.activities.indicators.to_shape") as mock_to_shape,
    ):
        mock_to_raw.return_value = [raw_det]
        mock_to_shape.return_value = box(0, 0, 1, 1)

        calc_instance = MockCalc.return_value
        calc_instance.compute.return_value = [mock_indicator]

        from worker.activities.indicators import compute_indicators

        result = await compute_indicators(str(job.id))

        assert result["status"] == "computed"
        assert result["indicator_count"] == 1
        calc_instance.compute.assert_called_once()
        mock_session.add_all.assert_called()


# --- Helper tests ---


def test_detections_to_raw():
    from worker.activities._helpers import detections_to_raw

    mock_det = MagicMock(spec=Detection)
    mock_det.class_name = "building"
    mock_det.confidence = 90
    mock_det.source = "samgeo"

    geom = box(0, 0, 1, 1)
    with patch("worker.activities._helpers.to_shape", return_value=geom):
        result = detections_to_raw([mock_det])

    assert len(result) == 1
    assert result[0].class_name == "building"
    assert result[0].confidence == 90
    assert result[0].geometry == geom


def test_raw_to_detections():
    from worker.activities._helpers import raw_to_detections

    raw = _make_raw_detection()
    job_id = str(uuid.uuid4())

    from pyproj import Geod

    geod = Geod(ellps="WGS84")

    with patch("worker.activities._helpers.from_shape") as mock_from_shape:
        mock_from_shape.return_value = "WKBElement"
        result = raw_to_detections([raw], job_id, geod)

    assert len(result) == 1
    assert result[0].class_name == "building"
    assert result[0].confidence == 85
    assert result[0].job_id == uuid.UUID(job_id)
    mock_from_shape.assert_called_once()
