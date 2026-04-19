import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator
from shapely.geometry import shape
from shapely.validation import explain_validity


class DetectionCreate(BaseModel):
    class_name: str
    confidence: int = Field(ge=0, le=100)
    source: str
    geometry: dict[str, Any]

    @field_validator("geometry")
    @classmethod
    def validate_geometry(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            geom = shape(v)
        except Exception as exc:
            raise ValueError(f"Invalid GeoJSON geometry: {exc}") from exc
        if not geom.is_valid:
            raise ValueError(f"Invalid geometry: {explain_validity(geom)}")
        return v
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
