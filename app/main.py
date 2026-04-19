from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.router import api_router


def validate_settings() -> None:
    if not settings.API_V1_STR.startswith("/"):
        raise RuntimeError("API_V1_STR must start with '/'")
    if not settings.CORS_ORIGINS:
        raise RuntimeError("CORS_ORIGINS must not be empty")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _on_startup() -> None:
        validate_settings()

    app.include_router(api_router, prefix=settings.API_V1_STR)

    return app


app = create_app()
