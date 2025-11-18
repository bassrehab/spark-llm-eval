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

# semantic metrics have optional dependencies
try:
    from spark_llm_eval.evaluation.semantic import (
        BERTScoreMetric,
        EmbeddingSimilarityMetric,
        SemanticSimilarityMetric,
    )
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False
    BERTScoreMetric = None
    EmbeddingSimilarityMetric = None
    SemanticSimilarityMetric = None

# LLM-as-judge metrics
from spark_llm_eval.evaluation.llm_judge import (
    JudgeConfig,
    LLMJudgeMetric,
    PairwiseJudgeMetric,
    GEvalMetric,
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
    # semantic (optional)
    "BERTScoreMetric",
    "EmbeddingSimilarityMetric",
    "SemanticSimilarityMetric",
    # llm-as-judge
    "JudgeConfig",
    "LLMJudgeMetric",
    "PairwiseJudgeMetric",
    "GEvalMetric",
]
