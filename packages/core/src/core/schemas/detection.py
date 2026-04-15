import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DetectionCreate(BaseModel):
    class_name: str
    confidence: int = Field(ge=0, le=100)
    source: str
    geometry: dict  # GeoJSON geometry
    area_m2: float | None = None
    length_m: float | None = None
    date: datetime | None = None
    change_flag: bool | None = None


class DetectionResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    class_name: str
    confidence: int
    source: str
    geometry: dict  # GeoJSON geometry
    area_m2: float | None = None
    length_m: float | None = None
    date: datetime | None = None
    change_flag: bool | None = None
