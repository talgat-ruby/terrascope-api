import uuid

from sqlmodel import Field, SQLModel


class ZoneIndicator(SQLModel, table=True):
    __tablename__ = "zone_indicators"  # type: ignore[assignment]

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="processing_jobs.id", index=True)
    zone_id: uuid.UUID = Field(foreign_key="territories.id", index=True)
    class_name: str
    count: int = 0
    density_per_km2: float = 0.0
    total_area_m2: float = 0.0
