"""
spark-llm-eval: Distributed LLM evaluation framework for Apache Spark.

This package provides tools for running large-scale LLM evaluations
with statistical rigor on Spark clusters.
"""

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

__version__ = "0.1.0"

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
