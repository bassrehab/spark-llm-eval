"""Inference engines for various LLM providers."""

from spark_llm_eval.inference.base import (
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
)
from spark_llm_eval.inference.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitConfig,
    NoOpRateLimiter,
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
try:
    from spark_llm_eval.inference.openai_engine import OpenAIInferenceEngine
    __all__.append("OpenAIInferenceEngine")
except ImportError:
    pass

try:
    from spark_llm_eval.inference.batch_udf import (
        create_inference_udf,
        INFERENCE_OUTPUT_SCHEMA,
        cleanup_engines,
    )
    __all__.extend(["create_inference_udf", "INFERENCE_OUTPUT_SCHEMA", "cleanup_engines"])
except ImportError:
    pass
