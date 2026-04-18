import uuid

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class QualityMetrics(SQLModel, table=True):
    __tablename__ = "quality_metrics"  # type: ignore[assignment]
    __table_args__ = (
        UniqueConstraint("job_id", "class_name", name="uq_quality_metrics_job_class"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="processing_jobs.id", index=True)
    class_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou: float = 0.0
    map: float = 0.0
