"""Evaluation metrics for LLM outputs."""

from spark_llm_eval.evaluation.base import (
    Metric,
    MetricResult,
    ReferenceFreeMetic,
    register_metric,
    get_metric,
    list_metrics,
)
from spark_llm_eval.evaluation.lexical import (
    ExactMatchMetric,
    F1Metric,
    ContainsMetric,
    BLEUMetric,
    ROUGELMetric,
    LengthRatioMetric,
    normalize_text,
    tokenize,
)
from spark_llm_eval.evaluation.aggregator import (
    MetricAggregator,
    AggregatedMetrics,
    compute_metrics,
)

__all__ = [
    # base
    "Metric",
    "MetricResult",
    "ReferenceFreeMetic",
    "register_metric",
    "get_metric",
    "list_metrics",
    # lexical
    "ExactMatchMetric",
    "F1Metric",
    "ContainsMetric",
    "BLEUMetric",
    "ROUGELMetric",
    "LengthRatioMetric",
    "normalize_text",
    "tokenize",
    # aggregator
    "MetricAggregator",
    "AggregatedMetrics",
    "compute_metrics",
]
