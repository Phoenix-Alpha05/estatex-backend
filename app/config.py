from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Property Intelligence"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"
    ENV: str = "dev"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    @field_validator("API_V1_STR")
    @classmethod
    def api_prefix_must_start_with_slash(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("API_V1_STR must start with '/'")
        return v

    class Config:
        env_file = ".env"


settings = Settings()
