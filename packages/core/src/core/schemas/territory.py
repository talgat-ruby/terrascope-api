import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator
from shapely.geometry import shape
from shapely.validation import explain_validity


class TerritoryCreate(BaseModel):
    name: str
    crs: str = Field(default="EPSG:4326")
    geometry: dict[str, Any]

    @field_validator("geometry")
    @classmethod
    def validate_geometry(cls, v: dict[str, Any]) -> dict[str, Any]:
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


class TerritoryResponse(BaseModel):
    id: uuid.UUID
    name: str
    crs: str
    geometry: dict  # GeoJSON polygon
