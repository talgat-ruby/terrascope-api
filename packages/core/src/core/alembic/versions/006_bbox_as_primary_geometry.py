"""Make bbox the primary geometry; drop the dedicated bbox column

The previous schema had geometry=POINT (centroid) and bbox=POLYGON. This
migration flips it: geometry holds the bbox polygon (so QGIS / geojson.io
draw rectangles by default) and the bbox column is dropped. Centroid is
recomputed at export time as a WKT attribute.

Revision ID: 006
Revises: 005
Create Date: 2026-04-28
"""

from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Replace geometry (POINT) with the bbox polygon. Backfill from existing
    # bbox column where present.
    op.alter_column(
        "detections",
        "geometry",
        type_=geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326),
        existing_type=geoalchemy2.Geometry(geometry_type="POINT", srid=4326),
        existing_nullable=True,
        postgresql_using="bbox",
    )
    op.drop_column("detections", "bbox")


def downgrade() -> None:
    op.add_column(
        "detections",
        sa.Column(
            "bbox",
            geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326),
            nullable=True,
        ),
    )
    op.execute("UPDATE detections SET bbox = geometry")
    op.alter_column(
        "detections",
        "geometry",
        type_=geoalchemy2.Geometry(geometry_type="POINT", srid=4326),
        existing_type=geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326),
        existing_nullable=True,
        postgresql_using="ST_Centroid(geometry)",
    )
