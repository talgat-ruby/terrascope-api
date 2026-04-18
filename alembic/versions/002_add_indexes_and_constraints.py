"""Add indexes and unique constraints

Revision ID: 002
Revises: 001
Create Date: 2026-04-18
"""

from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_processing_jobs_status", "processing_jobs", ["status"]
    )
    op.create_index(
        "ix_processing_jobs_created_at", "processing_jobs", ["created_at"]
    )
    op.create_unique_constraint(
        "uq_quality_metrics_job_class",
        "quality_metrics",
        ["job_id", "class_name"],
    )
    op.create_unique_constraint(
        "uq_zone_indicators_job_zone_class",
        "zone_indicators",
        ["job_id", "zone_id", "class_name"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_zone_indicators_job_zone_class", "zone_indicators", type_="unique"
    )
    op.drop_constraint(
        "uq_quality_metrics_job_class", "quality_metrics", type_="unique"
    )
    op.drop_index("ix_processing_jobs_created_at", "processing_jobs")
    op.drop_index("ix_processing_jobs_status", "processing_jobs")
