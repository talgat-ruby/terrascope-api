import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from core.models.processing import JobStatus


class ProcessingRequest(BaseModel):
    input_path: str
    aoi: dict  # GeoJSON polygon for area of interest
    aoi_crs: str = Field(default="EPSG:4326")
    config: dict | None = None  # Optional overrides for processing settings


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
