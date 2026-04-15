import uuid

from pydantic import BaseModel, Field


class TerritoryCreate(BaseModel):
    name: str
    crs: str = Field(default="EPSG:4326")
    geometry: dict  # GeoJSON polygon


class TerritoryResponse(BaseModel):
    id: uuid.UUID
    name: str
    crs: str
    geometry: dict  # GeoJSON polygon
