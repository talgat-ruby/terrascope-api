import uuid

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class ZoneIndicator(SQLModel, table=True):
    __tablename__ = "zone_indicators"  # type: ignore[assignment]
    __table_args__ = (
        UniqueConstraint(
            "job_id", "zone_id", "class_name",
            name="uq_zone_indicators_job_zone_class",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="processing_jobs.id", index=True)
    zone_id: uuid.UUID = Field(foreign_key="territories.id", index=True)
    class_name: str
    count: int = 0
    density_per_km2: float = 0.0
    total_area_m2: float = 0.0
