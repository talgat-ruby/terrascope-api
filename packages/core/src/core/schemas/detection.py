import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator
from shapely.geometry import shape
from shapely.validation import explain_validity


class DetectionCreate(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    geometry: dict[str, Any]  # bbox polygon, GeoJSON

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


class DetectionResponse(BaseModel):
    id: int
    job_id: uuid.UUID
    class_name: str
    confidence: float
    geometry: dict  # bbox polygon
    centroid: dict | None = None
