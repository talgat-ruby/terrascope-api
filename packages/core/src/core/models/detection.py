import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Column
from sqlmodel import Field, SQLModel


class Detection(SQLModel, table=True):
    __tablename__ = "detections"  # type: ignore[assignment]

    id: int = Field(primary_key=True)
    job_id: uuid.UUID = Field(
        foreign_key="processing_jobs.id", primary_key=True, index=True
    )
    class_name: str = Field(index=True)
    confidence: float
    # Bbox polygon. Centroid is recomputed at export time.
    geometry: str | None = Field(
        default=None,
        sa_column=Column(Geometry(geometry_type="POLYGON", srid=4326)),
    )
