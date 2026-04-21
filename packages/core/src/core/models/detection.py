import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel

CLASS_REGISTRY: dict[str, dict] = {
    "building": {
        "description": "Buildings and man-made structures",
        "geometry_type": "Polygon",
        "justification": "Well-supported by pretrained SAM models; distinct spectral and spatial signature in satellite imagery",
    },
    "road": {
        "description": "Roads, paths, and paved surfaces",
        "geometry_type": "Polygon",
        "justification": "Linear features detectable via semantic segmentation; important for infrastructure monitoring",
    },
    "vegetation": {
        "description": "Forest, cropland, grassland, and other vegetated areas",
        "geometry_type": "Polygon",
        "justification": "Strong spectral signature (NDVI); large-area land cover class well-suited to semantic segmentation",
    },
    "water": {
        "description": "Rivers, lakes, ponds, and water bodies",
        "geometry_type": "Polygon",
        "justification": "Distinct spectral signature in near-infrared; important for environmental monitoring",
    },
}


class Detection(SQLModel, table=True):
    __tablename__ = "detections"  # type: ignore[assignment]

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="processing_jobs.id", index=True)
    class_name: str = Field(index=True)
    confidence: int = Field(ge=0, le=100)
    source: str
    geometry: str | None = Field(
        default=None,
        sa_column=Column(Geometry(geometry_type="GEOMETRY", srid=4326)),
    )
    area_m2: float | None = None
    length_m: float | None = None
    date: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    change_flag: bool | None = None
