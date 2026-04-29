"""Switch detections to per-job sequential int id, add subtype + bbox

Revision ID: 004
Revises: 003
Create Date: 2026-04-27
"""

from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_detections_class_name", table_name="detections")
    op.drop_index("ix_detections_job_id", table_name="detections")
    op.drop_table("detections")

    op.create_table(
        "detections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "job_id", sa.Uuid(), sa.ForeignKey("processing_jobs.id"), nullable=False
        ),
        sa.Column("class_name", sa.String(), nullable=False),
        sa.Column("subtype", sa.String(), nullable=True),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column(
            "geometry",
            geoalchemy2.Geometry(geometry_type="POINT", srid=4326),
            nullable=True,
        ),
        sa.Column(
            "bbox",
            geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326),
            nullable=True,
        ),
        sa.Column("area_m2", sa.Float(), nullable=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("change_flag", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("job_id", "id"),
    )
    op.create_index("ix_detections_job_id", "detections", ["job_id"])
    op.create_index("ix_detections_class_name", "detections", ["class_name"])


def downgrade() -> None:
    op.drop_index("ix_detections_class_name", table_name="detections")
    op.drop_index("ix_detections_job_id", table_name="detections")
    op.drop_table("detections")

    op.create_table(
        "detections",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "job_id", sa.Uuid(), sa.ForeignKey("processing_jobs.id"), nullable=False
        ),
        sa.Column("class_name", sa.String(), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column(
            "geometry",
            geoalchemy2.Geometry(geometry_type="GEOMETRY", srid=4326),
            nullable=True,
        ),
        sa.Column("area_m2", sa.Float(), nullable=True),
        sa.Column("length_m", sa.Float(), nullable=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("change_flag", sa.Boolean(), nullable=True),
    )
    op.create_index("ix_detections_job_id", "detections", ["job_id"])
    op.create_index("ix_detections_class_name", "detections", ["class_name"])
