"""Abstract base class for inference engines.

All LLM providers implement this interface, making it easy to swap
providers without changing evaluation code.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for LLM inference.

    Args:
        prompt: The rendered prompt to send to the model.
        request_id: Unique ID for tracking and caching.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        stop_sequences: Sequences that stop generation.
        metadata: Optional extra data passed through to response.
    """

    prompt: str
    request_id: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Response from LLM inference.

    Args:
        text: Generated text from the model.
        input_tokens: Tokens in the prompt (for cost tracking).
        output_tokens: Tokens in the response.
        latency_ms: Time taken for the request.
        cost_usd: Estimated cost of this request.
        request_id: Matches the request that produced this response.
        model: Model that generated this response.
        finish_reason: Why generation stopped.
        error: If not None, indicates the request failed.
        metadata: Passed through from request.
    """

    text: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    request_id: str = ""
    model: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class InferenceEngine(ABC):
    """Abstract base class for LLM inference engines.

    Implementations should handle:
    - Connection management and pooling
    - Rate limiting (via RateLimiter integration)
    - Retries with backoff
    - Cost calculation

    Lifecycle:
        1. Create instance with config
        2. Call initialize() before first use
        3. Call infer() or infer_batch() for inference
        4. Call shutdown() when done

    Example:
        engine = OpenAIInferenceEngine(config)
        engine.initialize()
        try:
            response = engine.infer(request)
        finally:
            engine.shutdown()
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the engine.

        Called once before any inference. Sets up connections,
        loads credentials, etc.

        Raises:
            ConfigurationError: If configuration is invalid.
            InferenceError: If connection cannot be established.
        """
        pass

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a single request.

        Args:
            request: The inference request.

        Returns:
            InferenceResponse with generated text or error.

        Note:
            Implementations should handle retries internally.
        """
        pass

    def infer_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Run inference on a batch of requests.

        Default implementation just loops over infer(). Subclasses
        can override for more efficient batch processing.

        Args:
            requests: List of inference requests.

        Returns:
            List of responses in the same order as requests.
        """
        # simple sequential implementation - override for parallelism
        responses = []
        for req in requests:
            try:
                resp = self.infer(req)
                responses.append(resp)
            except Exception as e:
                # don't let one failure kill the batch
                logger.warning(f"Request {req.request_id} failed: {e}")
                responses.append(
                    InferenceResponse(
                        request_id=req.request_id,
                        text=None,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0,
                        cost_usd=0,
                        error=str(e),
                        metadata=req.metadata,
                    )
                )
        return responses

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources.

        Called when the engine is no longer needed. Should close
        connections, flush caches, etc.
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Used for rate limiting when exact counts aren't available.
        Default implementation uses rough heuristic.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # rough estimate: ~4 chars per token for English
        # subclasses should override with tokenizer-specific counts
        return len(text) // 4 + 1

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass
