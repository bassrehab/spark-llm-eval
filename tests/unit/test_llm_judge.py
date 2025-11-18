"""Tests for LLM-as-judge evaluation metrics."""

import pytest
from unittest.mock import MagicMock, patch
import json

from spark_llm_eval.evaluation.llm_judge import (
    JudgeConfig,
    LLMJudgeMetric,
    PairwiseJudgeMetric,
    GEvalMetric,
    _parse_judge_response,
    _parse_pairwise_response,
    DEFAULT_JUDGE_PROMPT,
    DEFAULT_PAIRWISE_PROMPT,
)
from spark_llm_eval.evaluation.base import MetricResult
from spark_llm_eval.core.config import ModelConfig, ModelProvider


@pytest.fixture
def judge_config():
    """Create a sample judge config."""
    model_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
    )
    return JudgeConfig(
        model_config=model_config,
        rubric="Evaluate the response for accuracy and helpfulness.",
        scale=(1, 5),
    )


@pytest.fixture
def mock_engine():
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.initialize = MagicMock()
    return engine


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
        )
        config = JudgeConfig(
            model_config=model_config,
            rubric="Test rubric",
        )
        assert config.scale == (1, 5)
        assert config.criteria == []
        assert config.include_reasoning is True
        assert config.temperature == 0.0

    def test_custom_values(self):
        """Test custom values are set correctly."""
        model_config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-sonnet",
        )
        config = JudgeConfig(
            model_config=model_config,
            rubric="Custom rubric",
            scale=(1, 10),
            criteria=["accuracy", "fluency"],
            include_reasoning=False,
            temperature=0.5,
        )
        assert config.scale == (1, 10)
        assert config.criteria == ["accuracy", "fluency"]
        assert config.include_reasoning is False
        assert config.temperature == 0.5


class TestParseJudgeResponse:
    """Tests for _parse_judge_response helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"score": 4, "reasoning": "Good response"}'
        score, reasoning = _parse_judge_response(response, (1, 5))
        assert score == 4
        assert reasoning == "Good response"

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in text."""
        response = 'Here is my evaluation: {"score": 3, "reasoning": "Average"} end'
        score, reasoning = _parse_judge_response(response, (1, 5))
        assert score == 3
        assert reasoning == "Average"

    def test_clamp_score_above_max(self):
        """Test that scores above max are clamped."""
        response = '{"score": 10, "reasoning": "Excellent"}'
        score, reasoning = _parse_judge_response(response, (1, 5))
        assert score == 5

    def test_clamp_score_below_min(self):
        """Test that scores below min are clamped."""
        response = '{"score": 0, "reasoning": "Poor"}'
        score, reasoning = _parse_judge_response(response, (1, 5))
        assert score == 1

    def test_fallback_extract_number(self):
        """Test fallback to extracting number from text."""
        response = "I would give this a score of 4 out of 5."
        score, reasoning = _parse_judge_response(response, (1, 5))
        assert score == 4

    def test_fallback_to_middle(self):
        """Test fallback to middle of scale when no score found."""
        response = "This is a good response."
        score, reasoning = _parse_judge_response(response, (1, 5))
        assert score == 3  # (1+5)//2


class TestParsePairwiseResponse:
    """Tests for _parse_pairwise_response helper."""

    def test_parse_winner_a_json(self):
        """Test parsing winner A from JSON."""
        response = '{"winner": "A", "reasoning": "A is better"}'
        winner, reasoning = _parse_pairwise_response(response)
        assert winner == "A"

    def test_parse_winner_b_json(self):
        """Test parsing winner B from JSON."""
        response = '{"winner": "B", "reasoning": "B is more accurate"}'
        winner, reasoning = _parse_pairwise_response(response)
        assert winner == "B"

    def test_parse_tie_json(self):
        """Test parsing tie from JSON."""
        response = '{"winner": "tie", "reasoning": "Both are equal"}'
        winner, reasoning = _parse_pairwise_response(response)
        assert winner == "TIE"

    def test_fallback_response_a_better(self):
        """Test fallback parsing for A better."""
        response = "Response A is better because it is more detailed."
        winner, _ = _parse_pairwise_response(response)
        assert winner == "A"

    def test_fallback_response_b_better(self):
        """Test fallback parsing for B better."""
        response = "Response B is better than A."
        winner, _ = _parse_pairwise_response(response)
        assert winner == "B"

    def test_fallback_tie(self):
        """Test fallback to tie when unclear."""
        # Use a response without A or B to trigger tie
        response = "The two responses are equally good, neither is better."
        winner, _ = _parse_pairwise_response(response)
        assert winner == "TIE"


class TestLLMJudgeMetric:
    """Tests for LLMJudgeMetric."""

    def test_initialization(self, judge_config):
        """Test metric initialization."""
        metric = LLMJudgeMetric(judge_config)
        assert metric.judge_config == judge_config
        assert metric.prompt_template == DEFAULT_JUDGE_PROMPT

    def test_custom_prompt_template(self, judge_config):
        """Test custom prompt template."""
        custom_prompt = "Rate this: {prediction}"
        metric = LLMJudgeMetric(judge_config, prompt_template=custom_prompt)
        assert metric.prompt_template == custom_prompt

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_single_prediction(self, mock_create_engine, judge_config):
        """Test computing score for a single prediction."""
        # setup mock engine
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 4, "reasoning": "Good answer"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = LLMJudgeMetric(judge_config)
        result = metric.compute(
            predictions=["Paris is the capital of France."],
            references=["Paris"],
            inputs=["What is the capital of France?"],
        )

        assert isinstance(result, MetricResult)
        assert result.name == "llm_judge"
        # score 4 on 1-5 scale = (4-1)/(5-1) = 0.75
        assert result.value == pytest.approx(0.75)
        assert len(result.per_example_scores) == 1
        assert result.per_example_scores[0] == pytest.approx(0.75)
        assert "raw_scores" in result.metadata
        assert result.metadata["raw_scores"][0] == 4

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_without_references(self, mock_create_engine, judge_config):
        """Test computing without references."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 3, "reasoning": "Average"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = LLMJudgeMetric(judge_config)
        result = metric.compute(
            predictions=["Some response"],
            inputs=["Some question"],
        )

        assert isinstance(result, MetricResult)
        mock_engine.infer.assert_called_once()

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_handles_inference_error(self, mock_create_engine, judge_config):
        """Test handling of inference errors."""
        mock_engine = MagicMock()
        mock_engine.infer.side_effect = Exception("API Error")
        mock_create_engine.return_value = mock_engine

        metric = LLMJudgeMetric(judge_config)
        result = metric.compute(
            predictions=["Some response"],
            inputs=["Some question"],
        )

        # should fall back to middle score
        assert result.value == pytest.approx(0.5)  # (3-1)/(5-1)


class TestPairwiseJudgeMetric:
    """Tests for PairwiseJudgeMetric."""

    def test_initialization(self, judge_config):
        """Test metric initialization."""
        metric = PairwiseJudgeMetric(judge_config)
        assert metric.judge_config == judge_config
        assert metric.prompt_template == DEFAULT_PAIRWISE_PROMPT

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_a_wins(self, mock_create_engine, judge_config):
        """Test pairwise comparison where A wins."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"winner": "A", "reasoning": "A is better"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = PairwiseJudgeMetric(judge_config)
        result = metric.compute(
            predictions_a=["Response A"],
            predictions_b=["Response B"],
            inputs=["Question"],
        )

        assert result.value == 1.0  # A win rate
        assert result.metadata["a_wins"] == 1
        assert result.metadata["b_wins"] == 0

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_b_wins(self, mock_create_engine, judge_config):
        """Test pairwise comparison where B wins."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"winner": "B", "reasoning": "B is more accurate"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = PairwiseJudgeMetric(judge_config)
        result = metric.compute(
            predictions_a=["Response A"],
            predictions_b=["Response B"],
            inputs=["Question"],
        )

        assert result.value == 0.0  # A win rate
        assert result.metadata["b_wins"] == 1

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_tie(self, mock_create_engine, judge_config):
        """Test pairwise comparison with tie."""
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"winner": "tie", "reasoning": "Equal quality"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = PairwiseJudgeMetric(judge_config)
        result = metric.compute(
            predictions_a=["Response A"],
            predictions_b=["Response B"],
            inputs=["Question"],
        )

        assert result.value == 0.5
        assert result.metadata["ties"] == 1

    def test_mismatched_lengths(self, judge_config):
        """Test error on mismatched prediction list lengths."""
        metric = PairwiseJudgeMetric(judge_config)
        with pytest.raises(ValueError, match="same length"):
            metric.compute(
                predictions_a=["A1", "A2"],
                predictions_b=["B1"],
            )


class TestGEvalMetric:
    """Tests for GEvalMetric."""

    def test_initialization(self, judge_config):
        """Test G-Eval initialization."""
        judge_config.criteria = ["accuracy", "fluency", "relevance"]
        metric = GEvalMetric(judge_config)
        assert metric.judge_config == judge_config

    def test_with_weights(self, judge_config):
        """Test G-Eval with custom weights."""
        judge_config.criteria = ["accuracy", "fluency"]
        weights = {"accuracy": 2.0, "fluency": 1.0}
        metric = GEvalMetric(judge_config, criteria_weights=weights)
        assert metric.criteria_weights == weights

    @patch("spark_llm_eval.inference.create_engine")
    def test_compute_multiple_criteria(self, mock_create_engine, judge_config):
        """Test G-Eval with multiple criteria."""
        judge_config.criteria = ["accuracy", "fluency"]

        mock_engine = MagicMock()
        # return different scores for different criteria
        responses = [
            '{"score": 5, "reasoning": "Very accurate"}',
            '{"score": 3, "reasoning": "Average fluency"}',
        ]
        mock_response = MagicMock()
        mock_response.text = responses[0]

        call_count = [0]

        def side_effect(*args, **kwargs):
            result = MagicMock()
            result.text = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return result

        mock_engine.infer.side_effect = side_effect
        mock_create_engine.return_value = mock_engine

        metric = GEvalMetric(judge_config)
        result = metric.compute(
            predictions=["Test response"],
            inputs=["Test question"],
        )

        assert isinstance(result, MetricResult)
        assert result.name == "g_eval"
        assert "criteria_scores" in result.metadata
        assert "accuracy" in result.metadata["criteria_scores"]
        assert "fluency" in result.metadata["criteria_scores"]

    @patch("spark_llm_eval.inference.create_engine")
    def test_fallback_to_simple_judge(self, mock_create_engine, judge_config):
        """Test fallback when no criteria specified."""
        judge_config.criteria = []

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 4, "reasoning": "Good"}'
        mock_engine.infer.return_value = mock_response
        mock_create_engine.return_value = mock_engine

        metric = GEvalMetric(judge_config)
        result = metric.compute(
            predictions=["Test response"],
            inputs=["Test question"],
        )

        # should behave like LLMJudgeMetric
        assert isinstance(result, MetricResult)


class TestMetricRegistration:
    """Test that LLM judge metrics are registered."""

    def test_llm_judge_registered(self):
        """Test llm_judge metric is registered."""
        from spark_llm_eval.evaluation.base import get_metric, list_metrics

        metrics = list_metrics()
        assert "llm_judge" in metrics

    def test_pairwise_judge_registered(self):
        """Test pairwise_judge metric is registered."""
        from spark_llm_eval.evaluation.base import list_metrics

        metrics = list_metrics()
        assert "pairwise_judge" in metrics

    def test_g_eval_registered(self):
        """Test g_eval metric is registered."""
        from spark_llm_eval.evaluation.base import list_metrics

        metrics = list_metrics()
        assert "g_eval" in metrics
