import uuid

from pydantic import BaseModel


class QualityMetricsResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    class_name: str
    precision: float
    recall: float
    f1: float
    iou: float
    map: float


class QualityReportResponse(BaseModel):
    job_id: uuid.UUID
    metrics: list[QualityMetricsResponse]
    error_examples: list[dict] = []
