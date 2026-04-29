from pathlib import Path

from pydantic import Field, computed_field, model_validator
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

    # Detection
    detector_name: str = "yolov8n-sahi"
    yolo_weights: str = "yolov8n.pt"
    min_confidence: float = 0.25
    landscape_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    landscape_max_dim: int = 1024  # downsample raster long-side cap for the segmenter
    landscape_min_pixels: int = 200  # drop connected components smaller than this
    device: str = Field(
        default_factory=lambda: (
            (
                __import__("torch").cuda.is_available()
                and "cuda"
                or (
                    hasattr(__import__("torch").backends, "mps")
                    and __import__("torch").backends.mps.is_available()
                    and "mps"
                )
                or "cpu"
            )
            if __import__("importlib.util").util.find_spec("torch")
            else "cpu"
        )
    )

    # STAC
    stac_api_url: str = "https://earth-search.aws.element84.com/v1"

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        return self


settings = Settings()
