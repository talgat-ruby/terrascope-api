from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Terrascope API"
    debug: bool = False


settings = Settings()
