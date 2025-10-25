"""Tests for orchestrator/runner module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from spark_llm_eval.core.config import (
    ModelConfig,
    ModelProvider,
    MetricConfig,
    StatisticsConfig,
    OutputConfig,
)
from spark_llm_eval.core.task import EvalTask
from spark_llm_eval.orchestrator.runner import (
    RunnerConfig,
    EvaluationRunner,
)


def create_test_model_config():
    """Helper to create a test model config."""
    return ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
    )


def create_test_task():
    """Helper to create a test task."""
    return EvalTask(
        task_id="test-task",
        name="Test Task",
        dataset_path="/test/dataset",
        model_config=create_test_model_config(),
        prompt_template="Answer: {{ input }}",
        metrics=[MetricConfig(name="exact_match")],
        reference_column="reference",
    )


class TestRunnerConfig:
    """Tests for RunnerConfig."""

    def test_default_values(self):
        model_config = create_test_model_config()
        metrics = [MetricConfig(name="exact_match")]

        config = RunnerConfig(
            model_config=model_config,
            metrics=metrics,
        )

        assert config.model_config == model_config
        assert len(config.metrics) == 1
        assert config.checkpoint_interval == 0
        assert config.cache_responses is True

    def test_custom_values(self):
        model_config = create_test_model_config()
        metrics = [
            MetricConfig(name="exact_match"),
            MetricConfig(name="f1"),
        ]
        stats_config = StatisticsConfig(confidence_level=0.99)
        output_config = OutputConfig(save_results=True, results_path="/output")

        config = RunnerConfig(
            model_config=model_config,
            metrics=metrics,
            statistics_config=stats_config,
            output_config=output_config,
            checkpoint_interval=100,
            cache_responses=False,
        )

        assert config.statistics_config.confidence_level == 0.99
        assert config.checkpoint_interval == 100
        assert config.cache_responses is False
        assert config.output_config.save_results is True


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    def test_initialization(self):
        mock_spark = MagicMock()
        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="exact_match")],
        )

        runner = EvaluationRunner(mock_spark, config)

        assert runner.spark == mock_spark
        assert runner.config == config
        assert runner.tracker is None

    def test_initialization_with_tracker(self):
        mock_spark = MagicMock()
        mock_tracker = MagicMock()
        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="exact_match")],
        )

        runner = EvaluationRunner(mock_spark, config, tracker=mock_tracker)

        assert runner.tracker == mock_tracker

    def test_validate_data_missing_columns(self):
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ["id", "input"]  # missing reference

        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="exact_match")],
        )

        runner = EvaluationRunner(mock_spark, config)
        task = create_test_task()

        with pytest.raises(ValueError, match="reference"):
            runner._validate_data(mock_df, task)

    def test_validate_data_valid(self):
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_df.columns = ["id", "input", "reference"]

        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="exact_match")],
        )

        runner = EvaluationRunner(mock_spark, config)
        task = create_test_task()

        # should not raise
        runner._validate_data(mock_df, task)


class TestComputeStatistics:
    """Tests for statistics computation in runner."""

    def test_compute_statistics_continuous(self):
        mock_spark = MagicMock()

        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="f1")],
            statistics_config=StatisticsConfig(
                confidence_level=0.95,
                bootstrap_iterations=100,  # small for test speed
            ),
        )

        runner = EvaluationRunner(mock_spark, config)

        metrics = {"f1": [0.7, 0.75, 0.8, 0.72, 0.78, 0.73, 0.77, 0.74]}
        result = runner._compute_statistics(metrics, 8)

        assert "f1" in result
        assert result["f1"].sample_size == 8
        ci = result["f1"].confidence_interval
        assert ci[0] < result["f1"].value < ci[1]

    def test_compute_statistics_binary(self):
        mock_spark = MagicMock()

        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="exact_match")],
            statistics_config=StatisticsConfig(
                confidence_level=0.95,
                ci_method="analytical",
            ),
        )

        runner = EvaluationRunner(mock_spark, config)

        # binary scores
        metrics = {"exact_match": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]}
        result = runner._compute_statistics(metrics, 10)

        assert "exact_match" in result
        assert result["exact_match"].value == 0.6  # 6 out of 10
        ci = result["exact_match"].confidence_interval
        assert ci[0] < 0.6
        assert ci[1] > 0.6


class TestBuildResult:
    """Tests for result building."""

    def test_build_result(self):
        mock_spark = MagicMock()

        model_config = create_test_model_config()
        config = RunnerConfig(
            model_config=model_config,
            metrics=[MetricConfig(name="exact_match")],
        )

        runner = EvaluationRunner(mock_spark, config)
        runner._start_time = 100.0

        from spark_llm_eval.core.result import MetricValue
        import time

        # mock time to control elapsed
        with patch("time.time", return_value=110.0):
            metrics = {
                "exact_match": MetricValue(
                    value=0.8,
                    confidence_interval=(0.75, 0.85),
                    confidence_level=0.95,
                    standard_error=0.02,
                    sample_size=100,
                )
            }
            task = create_test_task()

            result = runner._build_result(task, metrics, 100)

        assert result.task_id == "test-task"
        assert result.num_examples == 100
        assert "exact_match" in result.metrics
        assert result.latency.total_duration_s == 10.0
