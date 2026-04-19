import uuid

from pydantic import BaseModel, Field


class IndicatorResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    zone_id: uuid.UUID
    class_name: str
    count: int = Field(ge=0)
    density_per_km2: float = Field(ge=0)
    total_area_m2: float = Field(ge=0)
