"""Tests for Temporal activity implementations."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import box

from core.detection.types import Detection as DomainDetection
from core.detection.types import Raster
from core.models.detection import Detection
from core.models.processing import JobStatus, ProcessingJob
from core.models.territory import Territory


def _make_job(
    job_id: str | None = None,
    status: JobStatus = JobStatus.PENDING,
    config: dict | None = None,
    checkpoint_data: dict | None = None,
) -> ProcessingJob:
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


def _make_detection() -> DomainDetection:
    bnds = (0.1, 0.1, 0.2, 0.2)
    return DomainDetection(
        id=0,
        class_name="car",
        confidence=0.85,
        bbox=bnds,
        pixel_bbox=(10, 10, 30, 30),
        centroid=box(*bnds).centroid,
    )


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.add = MagicMock()
    session.add_all = MagicMock()
    return session


def _setup_job_query(mock_session, job):
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    mock_session.execute = AsyncMock(return_value=result_mock)


@pytest.mark.asyncio
async def test_load_imagery_success(mock_session, tmp_path):
    job = _make_job()
    _setup_job_query(mock_session, job)

    raster = Raster(
        data=np.zeros((10, 10, 3), dtype=np.uint8),
        transform=Affine(1, 0, 0, 0, -1, 10),
        crs="EPSG:4326",
        aoi_geom=box(0, 0, 1, 1),
    )

    with (
        patch(
            "worker.activities.imagery.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.imagery.ImageryLoaderService") as MockLoader,
        patch("worker.activities.imagery.settings") as mock_settings,
    ):
        MockLoader.return_value.load_clipped.return_value = raster
        mock_settings.output_dir = tmp_path

        from worker.activities.imagery import load_imagery

        result = await load_imagery(str(job.id))

    assert result["status"] == "loaded"
    assert result["job_id"] == str(job.id)
    MockLoader.return_value.load_clipped.assert_called_once()


@pytest.mark.asyncio
async def test_detect_success(mock_session, tmp_path):
    transform = Affine(1, 0, 0, 0, -1, 10)
    clipped_path = tmp_path / "clipped.npy"
    np.save(clipped_path, np.zeros((10, 10, 3), dtype=np.uint8))

    job = _make_job(
        checkpoint_data={
            "load": {
                "crs": "EPSG:4326",
                "clipped_path": str(clipped_path),
                "transform": [
                    transform.a,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e,
                    transform.f,
                ],
                "shape": [10, 10, 3],
                "aoi_wkt": box(0, 0, 1, 1).wkt,
            }
        }
    )
    _setup_job_query(mock_session, job)

    stub_detector = MagicMock()
    stub_detector.name = "stub"
    stub_detector.detect.return_value = [_make_detection()]

    with (
        patch(
            "worker.activities.detection.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.detection.build_detector", return_value=stub_detector),
        patch("worker.activities.detection.render_overlay") as mock_render,
        patch("worker.activities.detection.settings") as mock_settings,
    ):
        mock_settings.output_dir = tmp_path

        from worker.activities.detection import detect

        result = await detect(str(job.id))

    assert result["status"] == "detected"
    assert result["detection_count"] == 1
    stub_detector.detect.assert_called_once()
    mock_render.assert_called_once()
    mock_session.add_all.assert_called()


@pytest.mark.asyncio
async def test_export_results_writes_geojson(mock_session, tmp_path):
    job = _make_job(
        checkpoint_data={
            "load": {"crs": "EPSG:4326"},
            "detect": {"overlay_path": str(tmp_path / "overlay.png")},
        }
    )

    db_det = MagicMock(spec=Detection)
    db_det.id = 0
    db_det.class_name = "car"
    db_det.confidence = 0.85
    db_det.geometry = MagicMock()

    job_result = MagicMock()
    job_result.scalar_one_or_none.return_value = job
    det_result = MagicMock()
    det_result.scalars.return_value.all.return_value = [db_det]
    mock_session.execute = AsyncMock(
        side_effect=[job_result, job_result, det_result, job_result]
    )

    with (
        patch(
            "worker.activities.export.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.export.GISExporterService") as MockExporter,
        patch("worker.activities.export.detections_to_domain") as mock_to_domain,
        patch("worker.activities.export.settings") as mock_settings,
    ):
        mock_to_domain.return_value = [_make_detection()]
        mock_settings.output_dir = tmp_path
        MockExporter.return_value.export_geojson.return_value = tmp_path / "x.geojson"

        from worker.activities.export import export_results

        result = await export_results(str(job.id))

    assert result["status"] == "exported"
    assert "geojson" in result["formats"]
    assert "png" in result["formats"]
    MockExporter.return_value.export_geojson.assert_called_once()


@pytest.mark.asyncio
async def test_compute_indicators_success(mock_session):
    job = _make_job(
        checkpoint_data={"load": {"crs": "EPSG:4326"}, "detect": {"detection_count": 1}}
    )
    job.aoi_id = None

    territory = MagicMock(spec=Territory)
    territory.id = uuid.uuid4()
    territory.geometry = box(0, 0, 1, 1)

    job_result = MagicMock()
    job_result.scalar_one_or_none.return_value = job
    det_result = MagicMock()
    det_result.scalars.return_value.all.return_value = [MagicMock(spec=Detection)]
    terr_result = MagicMock()
    terr_result.scalars.return_value.all.return_value = [territory]
    delete_result = MagicMock()
    mock_session.execute = AsyncMock(
        side_effect=[
            job_result,
            job_result,
            det_result,
            terr_result,
            delete_result,
            job_result,
        ]
    )

    with (
        patch(
            "worker.activities.indicators.async_session_factory",
            return_value=mock_session,
        ),
        patch("worker.activities.indicators.detections_to_domain") as mock_to_domain,
        patch("worker.activities.indicators.to_shape", return_value=box(0, 0, 1, 1)),
        patch("worker.activities.indicators.IndicatorCalculatorService") as MockCalc,
    ):
        mock_to_domain.return_value = [_make_detection()]
        mock_indicator = MagicMock()
        mock_indicator.zone_id = str(territory.id)
        mock_indicator.class_name = "car"
        mock_indicator.count = 1
        mock_indicator.density_per_km2 = 1.0
        mock_indicator.total_area_m2 = 100.0
        MockCalc.return_value.compute.return_value = [mock_indicator]

        from worker.activities.indicators import compute_indicators

        result = await compute_indicators(str(job.id))

    assert result["status"] == "computed"
    assert result["indicator_count"] == 1


# --- Helper tests ---


def test_detections_to_domain():
    from worker.activities._helpers import detections_to_domain

    db_det = MagicMock(spec=Detection)
    db_det.id = 7
    db_det.class_name = "car"
    db_det.confidence = 0.9
    db_det.geometry = MagicMock()

    bbox_poly = box(0, 0, 1, 1)

    with patch("worker.activities._helpers.to_shape", return_value=bbox_poly):
        result = detections_to_domain([db_det])

    assert len(result) == 1
    assert result[0].id == 7
    assert result[0].class_name == "car"
    assert result[0].confidence == 0.9
    assert result[0].bbox == (0.0, 0.0, 1.0, 1.0)


def test_domain_to_rows():
    from worker.activities._helpers import domain_to_rows

    job_id = str(uuid.uuid4())
    with patch("worker.activities._helpers.from_shape", return_value="WKBElement"):
        rows = domain_to_rows([_make_detection()], job_id)

    assert len(rows) == 1
    assert rows[0].class_name == "car"
    assert rows[0].confidence == 0.85
    assert rows[0].job_id == uuid.UUID(job_id)
    assert rows[0].id == 0
