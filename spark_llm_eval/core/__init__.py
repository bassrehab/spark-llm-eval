"""Core module containing configuration, task definitions, and result types."""

from spark_llm_eval.core.config import (
    ModelConfig,
    ModelProvider,
    MetricConfig,
    InferenceConfig,
    StatisticsConfig,
    SamplingConfig,
)
from spark_llm_eval.core.task import EvalTask
from spark_llm_eval.core.result import EvalResult, MetricValue
from spark_llm_eval.core.exceptions import (
    SparkLLMEvalError,
    InferenceError,
    RateLimitError,
    MetricComputationError,
    ConfigurationError,
    DatasetError,
)

__all__ = [
    "EvalTask",
    "EvalResult",
    "MetricValue",
    "ModelConfig",
    "ModelProvider",
    "MetricConfig",
    "InferenceConfig",
    "StatisticsConfig",
    "SamplingConfig",
    "SparkLLMEvalError",
    "InferenceError",
    "RateLimitError",
    "MetricComputationError",
    "ConfigurationError",
    "DatasetError",
]
