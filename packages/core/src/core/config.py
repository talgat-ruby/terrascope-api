from pathlib import Path

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Terrascope"
    app_env: str = "LOCAL"
    api_port: int = 30001
    debug: bool = False

    # Database (composed from individual env vars)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "terrascope"
    postgres_user: str = "terrascope"
    postgres_password: str = "terrascope"

    @computed_field
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @computed_field
    @property
    def database_url_sync(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # Temporal
    temporal_host: str = "127.0.0.1"
    temporal_port: int = 7233
    temporal_namespace: str = "default"
    temporal_task_queue: str = "terrascope-processing"

    @computed_field
    @property
    def temporal_address(self) -> str:
        return f"{self.temporal_host}:{self.temporal_port}"

    # File storage
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("output")
    model_cache_dir: Path = Path("model_cache")

    # Tiling
    tile_size: int = 512
    tile_overlap: int = 64

    # Detection
    confidence_threshold: int = 50
    min_area_m2: float = 10.0
    max_area_m2: float = 1_000_000.0
    min_length_m: float = 5.0
    nms_iou_threshold: float = 0.5
    simplify_tolerance_m: float = 1.0

    # ML model
    model_name: str = "torchgeo"
    device: str = "cpu"

    # STAC
    stac_api_url: str = "https://earth-search.aws.element84.com/v1"

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.min_area_m2 >= self.max_area_m2:
            raise ValueError("min_area_m2 must be less than max_area_m2")
        if not 0 <= self.nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be between 0 and 1")
        if not 0 <= self.confidence_threshold <= 100:
            raise ValueError("confidence_threshold must be between 0 and 100")
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be less than tile_size")
        return self


settings = Settings()
