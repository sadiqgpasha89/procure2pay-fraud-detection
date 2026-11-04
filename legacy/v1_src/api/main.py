"""FastAPI application entry point for Procure-to-Pay Fraud Detection & Prevention."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
from src.core.config import get_settings
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.ingestion import router as ingestion_router
from src.api.routes.cases import router as cases_router
from src.api.routes.monitoring import router as monitoring_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "app.startup",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.APP_ENV,
        stream_bus=settings.STREAM_BUS_TYPE,
        feature_store=settings.FEATURE_STORE_TYPE
    )
    yield
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Near real-time Procure-to-Pay (P2P) fraud detection, prevention, and forensic audit platform.",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routers
    app.include_router(evaluation_router)
    app.include_router(ingestion_router)
    app.include_router(cases_router)
    app.include_router(monitoring_router)

    @app.get("/")
    async def root():
        return {
            "message": "Procure-to-Pay Fraud Detection & Prevention Platform API",
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }

    return app


app = create_app()
