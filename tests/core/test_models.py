import uuid

from core.models import (
    JobStatus,
    ProcessingJob,
    QualityMetrics,
    Territory,
    ZoneIndicator,
)


def test_job_status_enum():
    assert JobStatus.PENDING == "pending"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"


def test_processing_job_defaults():
    job = ProcessingJob(input_path="/data/test.tif")
    assert job.status == JobStatus.PENDING
    assert job.input_path == "/data/test.tif"
    assert job.id is not None
    assert job.config is None


def test_territory_defaults():
    t = Territory(name="Test Zone")
    assert t.name == "Test Zone"
    assert t.crs == "EPSG:4326"


def test_zone_indicator_fields():
    job_id = uuid.uuid4()
    zone_id = uuid.uuid4()
    ind = ZoneIndicator(
        job_id=job_id,
        zone_id=zone_id,
        class_name="building",
        count=42,
        density_per_km2=3.5,
        total_area_m2=12000.0,
    )
    assert ind.count == 42
    assert ind.class_name == "building"


def test_quality_metrics_fields():
    job_id = uuid.uuid4()
    qm = QualityMetrics(
        job_id=job_id,
        class_name="road",
        precision=0.85,
        recall=0.78,
        f1=0.81,
        iou=0.65,
        map=0.72,
    )
    assert qm.f1 == 0.81
