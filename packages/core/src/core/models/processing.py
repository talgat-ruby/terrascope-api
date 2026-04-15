import enum
import uuid
from datetime import UTC, datetime

from sqlmodel import JSON, Column, Field, SQLModel


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    LOADING = "loading"
    TILING = "tiling"
    DETECTING = "detecting"
    POSTPROCESSING = "postprocessing"
    EXPORTING = "exporting"
    COMPUTING_INDICATORS = "computing_indicators"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJob(SQLModel, table=True):
    __tablename__ = "processing_jobs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    status: JobStatus = Field(default=JobStatus.PENDING)
    input_path: str
    aoi_id: uuid.UUID | None = Field(default=None, foreign_key="territories.id")
    config: dict | None = Field(default=None, sa_column=Column(JSON))
    current_step: str | None = None
    checkpoint_data: dict | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
