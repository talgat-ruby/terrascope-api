"""Simplify detections schema for the bbox-only detection pipeline

Drop subtype, source, area_m2, date, change_flag. Convert confidence from
int (0-100) to float (0.0-1.0). Keep id, job_id, class_name, geometry
(POINT), bbox (POLYGON), composite PK (job_id, id).

Revision ID: 005
Revises: 004
Create Date: 2026-04-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("detections", "subtype")
    op.drop_column("detections", "source")
    op.drop_column("detections", "area_m2")
    op.drop_column("detections", "date")
    op.drop_column("detections", "change_flag")

    # Convert confidence: int [0,100] -> float [0,1]. Backfill via USING.
    op.alter_column(
        "detections",
        "confidence",
        type_=sa.Float(),
        existing_type=sa.Integer(),
        existing_nullable=False,
        postgresql_using="confidence / 100.0",
    )


def downgrade() -> None:
    op.alter_column(
        "detections",
        "confidence",
        type_=sa.Integer(),
        existing_type=sa.Float(),
        existing_nullable=False,
        postgresql_using="(confidence * 100)::integer",
    )

    op.add_column("detections", sa.Column("change_flag", sa.Boolean(), nullable=True))
    op.add_column(
        "detections", sa.Column("date", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("detections", sa.Column("area_m2", sa.Float(), nullable=True))
    op.add_column(
        "detections",
        sa.Column("source", sa.String(), nullable=False, server_default=""),
    )
    op.add_column("detections", sa.Column("subtype", sa.String(), nullable=True))
