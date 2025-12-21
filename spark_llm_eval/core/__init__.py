"""Core module containing configuration, task definitions, and result types."""

from spark_llm_eval.core.config import (
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    ModelProvider,
    SamplingConfig,
    StatisticsConfig,
)
from spark_llm_eval.core.exceptions import (
    ConfigurationError,
    DatasetError,
    InferenceError,
    MetricComputationError,
    RateLimitError,
    SparkLLMEvalError,
)
from spark_llm_eval.core.result import EvalResult, MetricValue
from spark_llm_eval.core.task import EvalTask

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
