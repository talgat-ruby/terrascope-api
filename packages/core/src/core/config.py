from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Terrascope"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://terrascope:terrascope@localhost:5432/terrascope"
    database_url_sync: str = "postgresql://terrascope:terrascope@localhost:5432/terrascope"

    # Temporal
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "terrascope-processing"

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


settings = Settings()
