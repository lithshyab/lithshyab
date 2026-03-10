"""Pydantic schemas for request and response payloads."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class InferenceRequest(BaseModel):
    """Input payload for model inference.

    Time Complexity: O(n), where n is the number of features.
    Space Complexity: O(n), for storing and validating the feature vector.
    """

    features: List[float] = Field(
        ...,
        min_items=1,
        max_items=1024,
        description="Numeric feature vector used for prediction.",
        example=[0.12, 1.8, -3.4, 6.7],
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied correlation ID.",
        example="req-2026-03-10-001",
    )

    @validator("features")
    def ensure_finite_numbers(cls, value: List[float]) -> List[float]:
        """Validate that all feature values are finite numbers.

        Time Complexity: O(n), where n is the number of features.
        Space Complexity: O(1), excluding the input list.
        """

        for number in value:
            if number != number or number in (float("inf"), float("-inf")):
                raise ValueError("All feature values must be finite numbers.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "features": [0.12, 1.8, -3.4, 6.7],
                "request_id": "req-2026-03-10-001",
            }
        }


class InferenceResponse(BaseModel):
    """Standard successful response payload for inference.

    Time Complexity: O(1), assuming prediction output is scalar/constant-size metadata.
    Space Complexity: O(1), for response construction.
    """

    prediction: float = Field(..., description="Model prediction score.", example=0.9375)
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score.", example=0.91
    )
    model_version: str = Field(..., description="Active model version.", example="v1.0.0")
    request_id: Optional[str] = Field(
        default=None,
        description="Echoed correlation ID from request.",
        example="req-2026-03-10-001",
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": 0.9375,
                "confidence": 0.91,
                "model_version": "v1.0.0",
                "request_id": "req-2026-03-10-001",
            }
        }


class ErrorResponse(BaseModel):
    """Structured error response returned by custom exception handlers.

    Time Complexity: O(1), excluding serialization cost of details.
    Space Complexity: O(1), excluding dynamic details payload.
    """

    error_code: str = Field(..., description="Machine-readable error code.", example="VALIDATION_ERROR")
    message: str = Field(..., description="Human-readable error message.", example="Invalid request payload.")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for troubleshooting.",
    )

    class Config:
        schema_extra = {
            "example": {
                "error_code": "MODEL_INFERENCE_ERROR",
                "message": "Model inference failed.",
                "details": {"reason": "Model not loaded."},
            }
        }
