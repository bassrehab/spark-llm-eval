"""Inference engines for various LLM providers."""

from spark_llm_eval.inference.base import (
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
)
from spark_llm_eval.inference.rate_limiter import (
    NoOpRateLimiter,
    RateLimitConfig,
    TokenBucketRateLimiter,
)

__all__ = [
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "TokenBucketRateLimiter",
    "RateLimitConfig",
    "NoOpRateLimiter",
]

# lazy imports for optional dependencies
# these fail gracefully if deps not installed
_OpenAIInferenceEngine = None
_AnthropicEngine = None
_GeminiEngine = None

try:
    from spark_llm_eval.inference.openai_engine import OpenAIInferenceEngine

    _OpenAIInferenceEngine = OpenAIInferenceEngine
    __all__.append("OpenAIInferenceEngine")
except ImportError:
    pass

try:
    from spark_llm_eval.inference.anthropic_engine import AnthropicEngine

    _AnthropicEngine = AnthropicEngine
    __all__.append("AnthropicEngine")
except ImportError:
    pass

try:
    from spark_llm_eval.inference.gemini_engine import GeminiEngine

    _GeminiEngine = GeminiEngine
    __all__.append("GeminiEngine")
except ImportError:
    pass

try:
    from spark_llm_eval.inference.batch_udf import (
        INFERENCE_OUTPUT_SCHEMA,
        cleanup_engines,
        create_inference_udf,
    )

    __all__.extend(["create_inference_udf", "INFERENCE_OUTPUT_SCHEMA", "cleanup_engines"])
except ImportError:
    pass


def create_engine(config) -> InferenceEngine:
    """Factory function to create an inference engine.

    Args:
        config: ModelConfig with provider and model settings.

    Returns:
        InferenceEngine instance for the provider.

    Raises:
        ValueError: If provider not supported or deps not installed.
    """
    from spark_llm_eval.core.config import ModelProvider

    provider = config.provider

    if provider == ModelProvider.OPENAI:
        if _OpenAIInferenceEngine is None:
            raise ImportError("OpenAI not installed. Install with: pip install openai")
        return _OpenAIInferenceEngine(config)

    elif provider == ModelProvider.ANTHROPIC:
        if _AnthropicEngine is None:
            raise ImportError("Anthropic not installed. Install with: pip install anthropic")
        return _AnthropicEngine(config)

    elif provider == ModelProvider.GOOGLE:
        if _GeminiEngine is None:
            raise ImportError(
                "Google Generative AI not installed. Install with: pip install google-generativeai"
            )
        return _GeminiEngine(config)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__.append("create_engine")
