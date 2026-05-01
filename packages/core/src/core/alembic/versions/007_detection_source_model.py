"""Add source_model column to detections

Tracks which detector emitted each detection so multi-model jobs can be
debugged and analyzed by source. Column is NOT NULL with an index.

Revision ID: 007
Revises: 006
Create Date: 2026-05-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "detections",
        sa.Column("source_model", sa.String(), nullable=False),
    )
    op.create_index(
        "ix_detections_source_model",
        "detections",
        ["source_model"],
    )


def downgrade() -> None:
    op.drop_index("ix_detections_source_model", table_name="detections")
    op.drop_column("detections", "source_model")
