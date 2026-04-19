import uuid

from pydantic import BaseModel, Field


class QualityMetricsResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    class_name: str
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    map: float = Field(ge=0, le=1)


class QualityReportResponse(BaseModel):
    job_id: uuid.UUID
    metrics: list[QualityMetricsResponse]
    error_examples: list[dict] = []
