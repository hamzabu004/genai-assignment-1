import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.services.onnx_runtime_manager import load_all_models
from app.routers import (
    health,
    universal_restoration,
    hard_routing,
    soft_mixture,
    face_to_sketch,
)

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load and cache all available ONNX sessions
    logger.info("Initializing neural inference models...")
    loaded = load_all_models()
    logger.info(f"Loaded {len(loaded)} model(s): {loaded}")
    yield
    # Shutdown
    logger.info("Shutting down inference engine.")


app = FastAPI(
    title="GenAI Assignment API",
    description="Multi-Task Neural Image Restoration & Synthesis Suite",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Standard Error Format Handlers (Plan 2, Section 3)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail if isinstance(exc.detail, str) else "Request error",
            "detail": None if isinstance(exc.detail, str) else str(exc.detail),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Validation error") if errors else "Validation error"
    field_loc = " -> ".join([str(loc) for loc in errors[0].get("loc", [])]) if errors else ""
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": f"Validation failed: {first_msg} (field: {field_loc})",
            "detail": str(errors),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error during request processing: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal inference error",
            "detail": str(exc),
        },
    )


# Mount Endpoints
app.include_router(health.router)
app.include_router(universal_restoration.router)
app.include_router(hard_routing.router)
app.include_router(soft_mixture.router)
app.include_router(face_to_sketch.router)

