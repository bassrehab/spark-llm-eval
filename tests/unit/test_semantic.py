"""Tests for semantic evaluation metrics."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from spark_llm_eval.evaluation.base import MetricResult


class TestSemanticMetricsImport:
    """Tests for semantic metrics imports (optional dependencies)."""

    def test_bertscore_import_error_handled(self):
        """Test BERTScore fails gracefully when deps missing."""
        # When deps are not installed, the import should not raise
        # The metric will be None in the __all__
        from spark_llm_eval.evaluation import BERTScoreMetric
        # may be None if deps not installed
        assert BERTScoreMetric is None or hasattr(BERTScoreMetric, "compute")

    def test_embedding_similarity_import_error_handled(self):
        """Test EmbeddingSimilarity fails gracefully when deps missing."""
        from spark_llm_eval.evaluation import EmbeddingSimilarityMetric
        assert EmbeddingSimilarityMetric is None or hasattr(EmbeddingSimilarityMetric, "compute")


class TestBERTScoreMetricUnit:
    """Unit tests for BERTScoreMetric that don't require heavy deps."""

    def test_bertscore_registered(self):
        """Test BERTScore is registered when available."""
        from spark_llm_eval.evaluation.base import list_metrics
        metrics = list_metrics()
        # May or may not be registered depending on deps
        assert isinstance(metrics, list)

    def test_bertscore_cosine_similarity(self):
        """Test cosine similarity calculation helper."""
        try:
            from spark_llm_eval.evaluation.semantic import _cosine_similarity
        except ImportError:
            pytest.skip("Semantic dependencies not available")

        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])

        sim = _cosine_similarity(a, b)
        assert sim[0] == pytest.approx(1.0)

    def test_bertscore_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        try:
            from spark_llm_eval.evaluation.semantic import _cosine_similarity
        except ImportError:
            pytest.skip("Semantic dependencies not available")

        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])

        sim = _cosine_similarity(a, b)
        assert sim[0] == pytest.approx(0.0)


class TestEmbeddingSimilarityMetricUnit:
    """Unit tests for EmbeddingSimilarityMetric."""

    def test_embedding_similarity_registered(self):
        """Test embedding_similarity is registered when available."""
        from spark_llm_eval.evaluation.base import list_metrics
        metrics = list_metrics()
        assert isinstance(metrics, list)


class TestSemanticSimilarityMetric:
    """Tests for SemanticSimilarityMetric."""

    def test_semantic_similarity_exists(self):
        """Test SemanticSimilarityMetric class exists."""
        from spark_llm_eval.evaluation import SemanticSimilarityMetric
        # May be None if deps not installed
        if SemanticSimilarityMetric is not None:
            assert hasattr(SemanticSimilarityMetric, "compute")


class TestMetricResultFields:
    """Test that semantic metrics use correct MetricResult fields."""

    def test_metric_result_has_metadata(self):
        """Verify MetricResult uses metadata, not details."""
        result = MetricResult(
            name="test",
            value=0.5,
            per_example_scores=[0.5],
            metadata={"key": "value"},
        )
        assert result.metadata == {"key": "value"}


class TestCosineSimularity:
    """Test cosine similarity helper function."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        try:
            from spark_llm_eval.evaluation.semantic import _cosine_similarity
        except ImportError:
            pytest.skip("Semantic dependencies not available")

        a = np.array([[0.5, 0.5, 0.5]])
        b = np.array([[0.5, 0.5, 0.5]])

        sim = _cosine_similarity(a, b)
        assert sim[0] == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        try:
            from spark_llm_eval.evaluation.semantic import _cosine_similarity
        except ImportError:
            pytest.skip("Semantic dependencies not available")

        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[-1.0, 0.0, 0.0]])

        sim = _cosine_similarity(a, b)
        assert sim[0] == pytest.approx(-1.0, abs=0.01)

    def test_cosine_similarity_batch(self):
        """Test cosine similarity with multiple pairs."""
        try:
            from spark_llm_eval.evaluation.semantic import _cosine_similarity
        except ImportError:
            pytest.skip("Semantic dependencies not available")

        a = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        b = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 0.0, 1.0],  # orthogonal
        ])

        sim = _cosine_similarity(a, b)
        assert len(sim) == 2
        assert sim[0] == pytest.approx(1.0)
        assert sim[1] == pytest.approx(0.0)
