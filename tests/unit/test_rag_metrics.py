"""Tests for RAG evaluation metrics."""

import pytest
from unittest.mock import MagicMock, patch

from spark_llm_eval.evaluation.rag import (
    RAGMetric,
    RAGJudgeConfig,
    ContextRelevanceMetric,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
)
from spark_llm_eval.core.config import ModelConfig, ModelProvider


@pytest.fixture
def rag_judge_config():
    """Create a sample RAG judge config."""
    model_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
    )
    return RAGJudgeConfig(model_config=model_config)


@pytest.fixture
def mock_engine():
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.initialize = MagicMock()
    return engine


class TestRAGMetricBase:
    """Tests for RAGMetric base class."""

    def test_format_context_string(self):
        """Test formatting single string context."""
        result = RAGMetric.format_context("This is context.")
        assert result == "This is context."

    def test_format_context_list(self):
        """Test formatting list of context chunks."""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        result = RAGMetric.format_context(chunks)
        assert "[Chunk 1]:" in result
        assert "[Chunk 2]:" in result
        assert "[Chunk 3]:" in result
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "Chunk 3" in result

    def test_split_into_statements(self):
        """Test splitting text into statements."""
        text = "This is sentence one. This is sentence two! Is this three?"
        statements = RAGMetric.split_into_statements(text)
        assert len(statements) == 3
        assert "This is sentence one." in statements
        assert "This is sentence two!" in statements
        assert "Is this three?" in statements

    def test_split_into_statements_empty(self):
        """Test splitting empty text."""
        statements = RAGMetric.split_into_statements("")
        assert statements == []

    def test_split_into_statements_single(self):
        """Test splitting single statement."""
        text = "Just one sentence."
        statements = RAGMetric.split_into_statements(text)
        assert len(statements) == 1


class TestContextRelevanceMetric:
    """Tests for ContextRelevanceMetric."""

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_high_relevance(self, mock_create_engine, rag_judge_config):
        """Test high relevance score."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 5, "reasoning": "Highly relevant"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextRelevanceMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital of France."],
            queries=["What is the capital of France?"],
            contexts=["France is a country in Europe. Paris is its capital city."],
        )

        assert result.value == pytest.approx(1.0)  # (5-1)/4 = 1.0
        assert result.metadata["raw_scores"][0] == 5
        assert len(result.per_example_scores) == 1

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_low_relevance(self, mock_create_engine, rag_judge_config):
        """Test low relevance score."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 1, "reasoning": "Completely irrelevant"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextRelevanceMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Some answer"],
            queries=["What is the capital of France?"],
            contexts=["The moon orbits Earth every 27 days."],
        )

        assert result.value == pytest.approx(0.0)  # (1-1)/4 = 0.0
        assert result.metadata["raw_scores"][0] == 1

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_with_list_context(self, mock_create_engine, rag_judge_config):
        """Test with list of context chunks."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 4, "reasoning": "Mostly relevant"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextRelevanceMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital."],
            queries=["What is the capital of France?"],
            contexts=[["France is in Europe.", "Paris is the capital of France."]],
        )

        assert result.value == pytest.approx(0.75)  # (4-1)/4 = 0.75


class TestFaithfulnessMetric:
    """Tests for FaithfulnessMetric."""

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_faithful(self, mock_create_engine, rag_judge_config):
        """Test faithful answer detection."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 5, "reasoning": "All claims supported", "unsupported_claims": []}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = FaithfulnessMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital."],
            contexts=["Paris is the capital of France."],
        )

        assert result.value == pytest.approx(1.0)
        assert result.metadata["raw_scores"][0] == 5

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_hallucination(self, mock_create_engine, rag_judge_config):
        """Test hallucination detection."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 1, "reasoning": "Major hallucination", "unsupported_claims": ["The moon is made of cheese"]}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = FaithfulnessMetric(rag_judge_config)
        result = metric.compute(
            predictions=["The moon is made of cheese."],
            contexts=["The moon orbits Earth."],
        )

        assert result.value == pytest.approx(0.0)  # (1-1)/4 = 0
        assert len(result.metadata["details"][0].get("unsupported_claims", [])) > 0


class TestAnswerRelevanceMetric:
    """Tests for AnswerRelevanceMetric."""

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_relevant(self, mock_create_engine, rag_judge_config):
        """Test relevant answer."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 5, "reasoning": "Fully addresses the query"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = AnswerRelevanceMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital of France."],
            queries=["What is the capital of France?"],
        )

        assert result.value == pytest.approx(1.0)

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_irrelevant(self, mock_create_engine, rag_judge_config):
        """Test irrelevant answer."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 1, "reasoning": "Does not address the query"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = AnswerRelevanceMetric(rag_judge_config)
        result = metric.compute(
            predictions=["I like pizza."],
            queries=["What is the capital of France?"],
        )

        assert result.value == pytest.approx(0.0)


class TestContextPrecisionMetric:
    """Tests for ContextPrecisionMetric."""

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_high_precision(self, mock_create_engine, rag_judge_config):
        """Test high precision - all chunks relevant."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"relevant_chunks": [0, 1], "total_chunks": 2, "reasoning": "Both chunks relevant"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextPrecisionMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital."],
            references=["Paris"],
            queries=["What is the capital of France?"],
            contexts=[["Paris is in France.", "Paris is the capital."]],
        )

        assert result.value == pytest.approx(1.0)  # 2/2 = 1.0

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_low_precision(self, mock_create_engine, rag_judge_config):
        """Test low precision - only one chunk relevant."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"relevant_chunks": [1], "total_chunks": 3, "reasoning": "Only one relevant"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextPrecisionMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital."],
            references=["Paris"],
            queries=["What is the capital of France?"],
            contexts=[["Weather today.", "Paris is capital.", "Random text."]],
        )

        assert result.value == pytest.approx(1 / 3)


class TestContextRecallMetric:
    """Tests for ContextRecallMetric."""

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_high_recall(self, mock_create_engine, rag_judge_config):
        """Test high recall - all claims covered."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"supported_claims": ["Paris is capital"], "unsupported_claims": [], "recall_score": 1.0, "reasoning": "All covered"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextRecallMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris is the capital."],
            references=["Paris is the capital of France."],
            queries=["What is the capital of France?"],
            contexts=["France is a country. Paris is its capital."],
        )

        assert result.value == pytest.approx(1.0)

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_partial_recall(self, mock_create_engine, rag_judge_config):
        """Test partial recall - some claims missing."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"supported_claims": ["Paris"], "unsupported_claims": ["population"], "recall_score": 0.5, "reasoning": "Partial"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = ContextRecallMetric(rag_judge_config)
        result = metric.compute(
            predictions=["Paris has a population of 2 million."],
            references=["Paris is the capital with 2 million people."],
            queries=["Tell me about Paris."],
            contexts=["Paris is the capital of France."],
        )

        assert result.value == pytest.approx(0.5)


class TestRAGMetricValidation:
    """Tests for RAG metric input validation."""

    def test_missing_queries(self, rag_judge_config):
        """Test error when queries missing for metric that needs them."""
        metric = ContextRelevanceMetric(rag_judge_config)

        with pytest.raises(ValueError, match="requires 'queries'"):
            metric.validate_rag_inputs(
                predictions=["answer"],
                queries=None,
                contexts=["context"],
            )

    def test_missing_contexts(self, rag_judge_config):
        """Test error when contexts missing for metric that needs them."""
        metric = FaithfulnessMetric(rag_judge_config)

        with pytest.raises(ValueError, match="requires 'contexts'"):
            metric.validate_rag_inputs(
                predictions=["answer"],
                contexts=None,
            )

    def test_length_mismatch(self, rag_judge_config):
        """Test error when input lengths don't match."""
        metric = ContextRelevanceMetric(rag_judge_config)

        with pytest.raises(ValueError, match="length"):
            metric.validate_rag_inputs(
                predictions=["answer1", "answer2"],
                queries=["query1"],
                contexts=["context1", "context2"],
            )


class TestMetricRegistration:
    """Test that RAG metrics are registered."""

    def test_rag_metrics_registered(self):
        """Test all RAG metrics are in registry."""
        from spark_llm_eval.evaluation.base import list_metrics

        metrics = list_metrics()
        assert "context_relevance" in metrics
        assert "faithfulness" in metrics
        assert "answer_relevance" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics

    def test_rag_embedding_metrics_registered(self):
        """Test embedding RAG metrics are in registry."""
        from spark_llm_eval.evaluation.base import list_metrics

        metrics = list_metrics()
        assert "context_relevance_embedding" in metrics
        assert "answer_relevance_embedding" in metrics
        assert "faithfulness_nli" in metrics

    def test_get_metric_by_name(self, rag_judge_config):
        """Test getting RAG metric by name."""
        from spark_llm_eval.evaluation import get_metric

        metric = get_metric("context_relevance", judge_config=rag_judge_config)
        assert isinstance(metric, ContextRelevanceMetric)


class TestRAGJudgeConfig:
    """Tests for RAGJudgeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
        )
        config = RAGJudgeConfig(model_config=model_config)

        assert config.temperature == 0.0
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        model_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
        )
        config = RAGJudgeConfig(
            model_config=model_config,
            temperature=0.5,
            max_retries=5,
        )

        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.model_config.provider == ModelProvider.ANTHROPIC
