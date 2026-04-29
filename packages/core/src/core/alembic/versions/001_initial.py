"""Initial schema with PostGIS

Revision ID: 001
Revises:
Create Date: 2026-04-15
"""

from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "territories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("crs", sa.String(), nullable=False),
        sa.Column(
            "geometry",
            geoalchemy2.Geometry(geometry_type="POLYGON", srid=4326),
            nullable=True,
        ),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("input_path", sa.String(), nullable=False),
        sa.Column("aoi_id", sa.Uuid(), sa.ForeignKey("territories.id"), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("current_step", sa.String(), nullable=True),
        sa.Column("checkpoint_data", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )

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
        sa.Column("date", sa.DateTime(), nullable=True),
        sa.Column("change_flag", sa.Boolean(), nullable=True),
    )
    op.create_index("ix_detections_job_id", "detections", ["job_id"])
    op.create_index("ix_detections_class_name", "detections", ["class_name"])

    op.create_table(
        "zone_indicators",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "job_id", sa.Uuid(), sa.ForeignKey("processing_jobs.id"), nullable=False
        ),
        sa.Column(
            "zone_id", sa.Uuid(), sa.ForeignKey("territories.id"), nullable=False
        ),
        sa.Column("class_name", sa.String(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("density_per_km2", sa.Float(), nullable=False),
        sa.Column("total_area_m2", sa.Float(), nullable=False),
    )
    op.create_index("ix_zone_indicators_job_id", "zone_indicators", ["job_id"])
    op.create_index("ix_zone_indicators_zone_id", "zone_indicators", ["zone_id"])

    op.create_table(
        "quality_metrics",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "job_id", sa.Uuid(), sa.ForeignKey("processing_jobs.id"), nullable=False
        ),
        sa.Column("class_name", sa.String(), nullable=False),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("f1", sa.Float(), nullable=False),
        sa.Column("iou", sa.Float(), nullable=False),
        sa.Column("map", sa.Float(), nullable=False),
    )
    op.create_index("ix_quality_metrics_job_id", "quality_metrics", ["job_id"])


def downgrade() -> None:
    op.drop_table("quality_metrics")
    op.drop_table("zone_indicators")
    op.drop_table("detections")
    op.drop_table("processing_jobs")
    op.drop_table("territories")
    op.execute("DROP EXTENSION IF EXISTS postgis")
