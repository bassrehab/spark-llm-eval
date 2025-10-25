"""Tests for tracking module."""

import pytest
from unittest.mock import MagicMock, patch
from spark_llm_eval.tracking.mlflow_tracker import (
    TrackingConfig,
    MLflowTracker,
    create_tracker,
)


class TestTrackingConfig:
    """Tests for TrackingConfig."""

    def test_default_values(self):
        config = TrackingConfig(experiment_name="test-exp")
        assert config.experiment_name == "test-exp"
        assert config.tracking_uri is None
        assert config.run_name is None
        assert config.tags == {}
        assert config.log_artifacts is True

    def test_custom_values(self):
        config = TrackingConfig(
            experiment_name="eval-exp",
            tracking_uri="http://localhost:5000",
            run_name="gpt4-baseline",
            tags={"model": "gpt-4", "version": "1.0"},
            log_artifacts=False,
        )
        assert config.tracking_uri == "http://localhost:5000"
        assert config.run_name == "gpt4-baseline"
        assert config.tags["model"] == "gpt-4"


class TestMLflowTracker:
    """Tests for MLflowTracker."""

    def test_initialization(self):
        config = TrackingConfig(experiment_name="test")
        tracker = MLflowTracker(config)

        assert tracker.config == config
        assert tracker._run_id is None
        assert tracker._initialized is False

    def test_flatten_dict(self):
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        flat = MLflowTracker._flatten_dict(nested)

        assert flat["a"] == 1
        assert flat["b.c"] == 2
        assert flat["b.d.e"] == 3

    def test_flatten_dict_empty(self):
        flat = MLflowTracker._flatten_dict({})
        assert flat == {}

    @patch("spark_llm_eval.tracking.mlflow_tracker._get_mlflow")
    def test_log_params_truncates_long_values(self, mock_get_mlflow):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow

        config = TrackingConfig(experiment_name="test")
        tracker = MLflowTracker(config)
        tracker._mlflow = mock_mlflow
        tracker._initialized = True

        # param with very long value
        long_value = "x" * 600
        tracker.log_params({"long_param": long_value})

        call_args = mock_mlflow.log_params.call_args
        logged_params = call_args[0][0]

        assert len(logged_params["long_param"]) == 500
        assert logged_params["long_param"].endswith("...")

    @patch("spark_llm_eval.tracking.mlflow_tracker._get_mlflow")
    def test_log_metrics(self, mock_get_mlflow):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow

        config = TrackingConfig(experiment_name="test")
        tracker = MLflowTracker(config)
        tracker._mlflow = mock_mlflow
        tracker._initialized = True

        metrics = {"accuracy": 0.85, "f1": 0.82}
        tracker.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)

    @patch("spark_llm_eval.tracking.mlflow_tracker._get_mlflow")
    def test_log_metric_with_ci(self, mock_get_mlflow):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow

        config = TrackingConfig(experiment_name="test")
        tracker = MLflowTracker(config)
        tracker._mlflow = mock_mlflow
        tracker._initialized = True

        tracker.log_metric_with_ci("accuracy", 0.85, 0.82, 0.88)

        call_args = mock_mlflow.log_metrics.call_args
        logged = call_args[0][0]

        assert logged["accuracy"] == 0.85
        assert logged["accuracy_ci_lower"] == 0.82
        assert logged["accuracy_ci_upper"] == 0.88


class TestCreateTracker:
    """Tests for create_tracker function."""

    def test_creates_tracker_with_defaults(self):
        tracker = create_tracker("test-experiment")

        assert tracker.config.experiment_name == "test-experiment"
        assert tracker.config.tracking_uri is None

    def test_creates_tracker_with_options(self):
        tracker = create_tracker(
            "test-experiment",
            tracking_uri="http://localhost:5000",
            run_name="run-1",
            tags={"env": "test"},
        )

        assert tracker.config.tracking_uri == "http://localhost:5000"
        assert tracker.config.run_name == "run-1"
        assert tracker.config.tags == {"env": "test"}
