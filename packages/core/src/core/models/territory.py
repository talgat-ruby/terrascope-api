import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Column
from sqlmodel import Field, SQLModel


class Territory(SQLModel, table=True):
    __tablename__ = "territories"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    crs: str = Field(default="EPSG:4326")
    geometry: str | None = Field(
        default=None,
        sa_column=Column(Geometry(geometry_type="POLYGON", srid=4326)),
    )
