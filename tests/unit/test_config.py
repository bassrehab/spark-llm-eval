"""Tests for core configuration classes."""

import pytest
from spark_llm_eval.core.config import (
    ModelProvider,
    ModelConfig,
    MetricConfig,
    SamplingConfig,
    StatisticsConfig,
    InferenceConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_create_basic_config(self):
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
        )
        assert config.provider == ModelProvider.OPENAI
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.0  # default
        assert config.max_tokens == 1024

    def test_create_with_all_params(self):
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=2048,
            api_key_secret="my-scope/anthropic-key",
            extra_params={"top_p": 0.9},
        )
        assert config.temperature == 0.7
        assert config.extra_params["top_p"] == 0.9

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                temperature=3.0,  # too high
            )

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                max_tokens=-1,
            )

    def test_config_is_frozen(self):
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            config.temperature = 0.5


class TestMetricConfig:
    """Tests for MetricConfig."""

    def test_create_lexical_metric(self):
        config = MetricConfig(
            name="exact_match",
            metric_type="lexical",
        )
        assert config.name == "exact_match"
        assert config.metric_type == "lexical"

    def test_create_llm_judge_metric(self):
        judge = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
        )
        config = MetricConfig(
            name="correctness",
            metric_type="llm_judge",
            params={"rubric": "Is the answer correct?"},
            judge_model=judge,
        )
        assert config.judge_model is not None
        assert config.params["rubric"] == "Is the answer correct?"

    def test_invalid_metric_type_raises(self):
        with pytest.raises(ValueError, match="metric_type"):
            MetricConfig(
                name="foo",
                metric_type="invalid_type",
            )

    def test_llm_judge_requires_judge_model(self):
        with pytest.raises(ValueError, match="judge_model"):
            MetricConfig(
                name="correctness",
                metric_type="llm_judge",
            )


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_create_with_sample_size(self):
        config = SamplingConfig(sample_size=1000)
        assert config.sample_size == 1000
        assert config.sample_fraction is None

    def test_create_with_fraction(self):
        config = SamplingConfig(sample_fraction=0.1)
        assert config.sample_fraction == 0.1

    def test_must_specify_size_or_fraction(self):
        with pytest.raises(ValueError):
            SamplingConfig()  # neither specified

    def test_cannot_specify_both(self):
        with pytest.raises(ValueError, match="pick one"):
            SamplingConfig(sample_size=100, sample_fraction=0.1)

    def test_stratified_requires_columns(self):
        with pytest.raises(ValueError, match="stratify_columns"):
            SamplingConfig(
                strategy="stratified",
                sample_size=100,
            )


class TestStatisticsConfig:
    """Tests for StatisticsConfig."""

    def test_default_values(self):
        config = StatisticsConfig()
        assert config.confidence_level == 0.95
        assert config.bootstrap_iterations == 1000
        assert config.significance_threshold == 0.05

    def test_invalid_confidence_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            StatisticsConfig(confidence_level=1.5)

    def test_too_few_bootstrap_iterations(self):
        with pytest.raises(ValueError, match="bootstrap"):
            StatisticsConfig(bootstrap_iterations=50)


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_default_values(self):
        config = InferenceConfig()
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.enable_caching is True

    def test_with_rate_limits(self):
        config = InferenceConfig(
            rate_limit_rpm=10000,
            rate_limit_tpm=1000000,
        )
        assert config.rate_limit_rpm == 10000
        assert config.rate_limit_tpm == 1000000

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            InferenceConfig(batch_size=0)
