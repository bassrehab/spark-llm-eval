"""
spark-llm-eval: Distributed LLM evaluation framework for Apache Spark.

This package provides tools for running large-scale LLM evaluations
with statistical rigor on Spark clusters.
"""

from spark_llm_eval.core.config import (
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    ModelProvider,
    SamplingConfig,
    StatisticsConfig,
)
from spark_llm_eval.core.result import EvalResult, MetricValue
from spark_llm_eval.core.task import EvalTask

__version__ = "0.1.3"

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
]
