"""API routes for model serving endpoints."""

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from schemas.models import InferenceRequest, InferenceResponse
from services.ml_model import MLModelService

router = APIRouter(prefix="/api/v1", tags=["Inference"])


def get_model_service(request: Request) -> MLModelService:
    """Resolve model service from application state.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """

    return request.app.state.model_service


@router.get("/health", summary="Service health check")
async def health_check() -> dict:
    """Return health status for monitoring and readiness checks.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """

    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=InferenceResponse,
    summary="Run model inference",
)
async def predict(
    payload: InferenceRequest,
    model_service: MLModelService = Depends(get_model_service),
) -> InferenceResponse:
    """Perform non-blocking model inference via threadpool execution.

    Time Complexity: O(n), where n is the number of input features.
    Space Complexity: O(1), excluding request/response payload sizes.
    """

    result = await run_in_threadpool(model_service.predict, payload.features)
    return InferenceResponse(
        prediction=result.prediction,
        confidence=result.confidence,
        model_version=result.model_version,
        request_id=payload.request_id,
    )
