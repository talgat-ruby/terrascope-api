import uuid

from pydantic import BaseModel


class IndicatorResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    zone_id: uuid.UUID
    class_name: str
    count: int
    density_per_km2: float
    total_area_m2: float
