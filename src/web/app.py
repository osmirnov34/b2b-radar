from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.web.routers import api_keys, dashboard, documents, sources

STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="B2B Radar")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(dashboard.router)
    app.include_router(sources.router)
    app.include_router(documents.router)
    app.include_router(api_keys.router)

    return app


app = create_app()
