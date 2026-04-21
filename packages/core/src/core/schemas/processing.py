import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator
from shapely.geometry import shape
from shapely.validation import explain_validity

from core.models.processing import JobStatus


def _validate_polygon(v: dict[str, Any]) -> dict[str, Any]:
    try:
        geom = shape(v)
    except Exception as exc:
        raise ValueError(f"Invalid GeoJSON geometry: {exc}") from exc
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"Expected Polygon or MultiPolygon, got {geom.geom_type}"
        )
    if not geom.is_valid:
        raise ValueError(f"Invalid geometry: {explain_validity(geom)}")
    return v


class ProcessingRequest(BaseModel):
    input_path: str
    aoi: dict[str, Any] | None = Field(
        default=None,
        description="AOI GeoJSON geometry. If omitted, the full extent of the input raster is used.",
    )
    aoi_crs: str = Field(default="EPSG:4326")
    config: dict | None = None

    @field_validator("aoi")
    @classmethod
    def validate_aoi(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        return _validate_polygon(v)


class ProcessingStatusResponse(BaseModel):
    id: uuid.UUID
    status: JobStatus
    current_step: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class ProcessingResultResponse(BaseModel):
    id: uuid.UUID
    status: JobStatus
    detection_count: int = 0
    export_formats: list[str] = []
    indicator_count: int = 0
