"""FastAPI application entrypoint for production-ready ML inference service."""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router as inference_router
from schemas.models import ErrorResponse
from services.ml_model import MLModelService, ModelInferenceError, ModelNotLoadedError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle: load once at startup and unload at shutdown.

    Time Complexity: O(1) for startup/shutdown hooks.
    Space Complexity: O(1) for service reference storage.
    """

    model_service = MLModelService()
    model_service.load_model()
    app.state.model_service = model_service
    try:
        yield
    finally:
        model_service.unload_model()


app = FastAPI(
    title="Enterprise ML Inference API",
    description="Scalable and production-ready FastAPI service for deep learning inference.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://example-frontend.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.include_router(inference_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a structured error response for schema validation failures.

    Time Complexity: O(k), where k is the number of validation errors.
    Space Complexity: O(k), for serialized error detail list.
    """

    error_body = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Invalid request payload.",
        details={"errors": exc.errors(), "path": str(request.url.path)},
    )
    return JSONResponse(status_code=422, content=error_body.dict())


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """Return a structured error when model resources are unavailable.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """

    error_body = ErrorResponse(
        error_code="MODEL_NOT_LOADED",
        message="Model is not available.",
        details={"reason": str(exc), "path": str(request.url.path)},
    )
    return JSONResponse(status_code=503, content=error_body.dict())


@app.exception_handler(ModelInferenceError)
async def model_inference_handler(request: Request, exc: ModelInferenceError) -> JSONResponse:
    """Return a structured error response for inference runtime failures.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """

    error_body = ErrorResponse(
        error_code="MODEL_INFERENCE_ERROR",
        message="Model inference failed.",
        details={"reason": str(exc), "path": str(request.url.path)},
    )
    return JSONResponse(status_code=500, content=error_body.dict())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a structured error response for HTTP-layer exceptions.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """

    detail: Dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {"reason": exc.detail}
    detail["path"] = str(request.url.path)
    error_body = ErrorResponse(
        error_code="HTTP_ERROR",
        message="Request failed.",
        details=detail,
    )
    return JSONResponse(status_code=exc.status_code, content=error_body.dict())
