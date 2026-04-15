from pydantic import BaseModel, Field


class ExportRequest(BaseModel):
    formats: list[str] = Field(default=["geojson", "gpkg"])
    target_crs: str = Field(default="EPSG:4326")
