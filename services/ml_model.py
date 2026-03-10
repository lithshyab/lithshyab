"""Service layer for model lifecycle and inference."""

import math
import time
from dataclasses import dataclass
from typing import List


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested before the model is loaded."""


class ModelInferenceError(RuntimeError):
    """Raised when inference fails due to model execution issues."""


@dataclass
class PredictionResult:
    """Internal prediction object.

    Time Complexity: O(1), as it stores fixed-size output fields.
    Space Complexity: O(1).
    """

    prediction: float
    confidence: float
    model_version: str


class MLModelService:
    """Stateful service responsible for model lifecycle and inference.

    Time Complexity: O(1) for lifecycle operations; inference complexity is O(n)
    where n equals the number of input features.
    Space Complexity: O(1) lifecycle overhead; inference uses O(1) extra space
    besides input features.
    """

    def __init__(self) -> None:
        """Initialize unloaded model service state.

        Time Complexity: O(1).
        Space Complexity: O(1).
        """

        self._is_loaded = False
        self._model_version = "v1.0.0"

    def load_model(self) -> None:
        """Load model artifacts into memory once at startup.

        Time Complexity: O(1), simulated fixed initialization.
        Space Complexity: O(1), simulated constant memory footprint metadata.
        """

        # Simulate startup initialization (e.g., loading weights/tokenizers).
        self._is_loaded = True

    def unload_model(self) -> None:
        """Release model resources during shutdown.

        Time Complexity: O(1).
        Space Complexity: O(1).
        """

        self._is_loaded = False

    def predict(self, features: List[float]) -> PredictionResult:
        """Run CPU-bound inference using a deterministic scoring function.

        Time Complexity: O(n), where n is the number of input features.
        Space Complexity: O(1), excluding input storage.
        """

        if not self._is_loaded:
            raise ModelNotLoadedError("Model is not loaded.")

        try:
            # Simulate expensive CPU-bound inference.
            time.sleep(0.25)
            weighted_sum = sum((idx + 1) * value for idx, value in enumerate(features))
            probability = 1 / (1 + math.exp(-weighted_sum / max(len(features), 1)))
            confidence = min(1.0, max(0.0, abs(probability - 0.5) * 2))
            return PredictionResult(
                prediction=round(probability, 6),
                confidence=round(confidence, 6),
                model_version=self._model_version,
            )
        except Exception as exc:  # pragma: no cover - defensive block for runtime failures
            raise ModelInferenceError("Model inference execution failed.") from exc
