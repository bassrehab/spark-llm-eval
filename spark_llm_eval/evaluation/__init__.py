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

# RAG evaluation metrics
from spark_llm_eval.evaluation.rag import (
    RAGMetric,
    RAGJudgeConfig,
    ContextRelevanceMetric,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
)

# RAG embedding-based metrics (optional dependencies)
try:
    from spark_llm_eval.evaluation.rag import (
        ContextRelevanceEmbeddingMetric,
        AnswerRelevanceEmbeddingMetric,
        FaithfulnessNLIMetric,
    )
    _HAS_RAG_EMBEDDING = True
except ImportError:
    _HAS_RAG_EMBEDDING = False
    ContextRelevanceEmbeddingMetric = None
    AnswerRelevanceEmbeddingMetric = None
    FaithfulnessNLIMetric = None

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
    # RAG metrics
    "RAGMetric",
    "RAGJudgeConfig",
    "ContextRelevanceMetric",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    # RAG embedding metrics (optional)
    "ContextRelevanceEmbeddingMetric",
    "AnswerRelevanceEmbeddingMetric",
    "FaithfulnessNLIMetric",
]
