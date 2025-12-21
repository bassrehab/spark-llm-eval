"""MLflow integration for experiment tracking.

Provides experiment tracking, metric logging, and artifact management
for LLM evaluation runs.
"""

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# lazy import mlflow to avoid hard dependency
_mlflow = None


def _get_mlflow():
    """Lazy import MLflow."""
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow

            _mlflow = mlflow
        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")
    return _mlflow


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.

    Args:
        experiment_name: Name of MLflow experiment.
        tracking_uri: MLflow tracking server URI (None for local).
        run_name: Optional name for this run.
        tags: Additional tags to log.
        log_artifacts: Whether to log artifacts (results, configs).
        artifact_location: Custom artifact location.
    """

    experiment_name: str
    tracking_uri: str | None = None
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    log_artifacts: bool = True
    artifact_location: str | None = None


class MLflowTracker:
    """Handles MLflow experiment tracking for evaluations.

    Logs metrics, parameters, and artifacts to MLflow for
    experiment tracking and comparison.
    """

    def __init__(self, config: TrackingConfig):
        """Initialize tracker.

        Args:
            config: Tracking configuration.
        """
        self.config = config
        self._run_id: str | None = None
        self._mlflow = None
        self._initialized = False

    def _init_mlflow(self):
        """Initialize MLflow connection."""
        if self._initialized:
            return

        self._mlflow = _get_mlflow()

        if self.config.tracking_uri:
            self._mlflow.set_tracking_uri(self.config.tracking_uri)

        # create or get experiment
        experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = self._mlflow.create_experiment(
                self.config.experiment_name,
                artifact_location=self.config.artifact_location,
            )
            logger.info(f"Created experiment: {self.config.experiment_name}")
        else:
            experiment_id = experiment.experiment_id

        self._mlflow.set_experiment(experiment_id=experiment_id)
        self._initialized = True

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Context manager for MLflow run.

        Args:
            run_name: Optional run name override.

        Yields:
            Run ID.

        Example:
            tracker = MLflowTracker(config)
            with tracker.start_run("eval-gpt4"):
                tracker.log_params({"model": "gpt-4"})
                tracker.log_metrics({"accuracy": 0.85})
        """
        self._init_mlflow()

        name = run_name or self.config.run_name
        with self._mlflow.start_run(run_name=name) as run:
            self._run_id = run.info.run_id

            # log default tags
            if self.config.tags:
                self._mlflow.set_tags(self.config.tags)

            try:
                yield self._run_id
            finally:
                self._run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameters.
        """
        if self._mlflow is None:
            self._init_mlflow()

        # flatten nested dicts and convert to strings
        flat_params = self._flatten_dict(params)

        # MLflow has a limit on param value length
        truncated = {}
        for k, v in flat_params.items():
            str_v = str(v)
            if len(str_v) > 500:
                str_v = str_v[:497] + "..."
            truncated[k] = str_v

        self._mlflow.log_params(truncated)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step number.
        """
        if self._mlflow is None:
            self._init_mlflow()

        self._mlflow.log_metrics(metrics, step=step)

    def log_metric_with_ci(
        self,
        name: str,
        value: float,
        ci_lower: float,
        ci_upper: float,
        step: int | None = None,
    ) -> None:
        """Log metric with confidence interval.

        Args:
            name: Metric name.
            value: Point estimate.
            ci_lower: CI lower bound.
            ci_upper: CI upper bound.
            step: Optional step number.
        """
        metrics = {
            name: value,
            f"{name}_ci_lower": ci_lower,
            f"{name}_ci_upper": ci_upper,
        }
        self.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        data: Any,
        filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Log artifact to MLflow.

        Args:
            data: Data to log (dict will be saved as JSON).
            filename: Name of the artifact file.
            artifact_path: Subdirectory in artifacts.
        """
        if not self.config.log_artifacts:
            return

        if self._mlflow is None:
            self._init_mlflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)

            if isinstance(data, dict):
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            elif isinstance(data, str):
                with open(filepath, "w") as f:
                    f.write(data)
            else:
                # try to pickle
                import pickle

                with open(filepath, "wb") as f:
                    pickle.dump(data, f)

            self._mlflow.log_artifact(filepath, artifact_path)

    def log_eval_config(self, config: dict[str, Any]) -> None:
        """Log evaluation configuration.

        Args:
            config: Configuration dictionary.
        """
        self.log_params(config)
        self.log_artifact(config, "eval_config.json", "config")

    def log_results_summary(
        self,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        """Log evaluation results summary.

        Args:
            metrics: Dictionary of metric name to result dict.
                Each result should have 'value', 'ci_lower', 'ci_upper'.
        """
        flat_metrics = {}
        for name, result in metrics.items():
            if isinstance(result, dict):
                flat_metrics[name] = result.get("value", result.get("mean", 0))
                if "ci_lower" in result:
                    flat_metrics[f"{name}_ci_lower"] = result["ci_lower"]
                if "ci_upper" in result:
                    flat_metrics[f"{name}_ci_upper"] = result["ci_upper"]
            else:
                flat_metrics[name] = float(result)

        self.log_metrics(flat_metrics)
        self.log_artifact(metrics, "results_summary.json", "results")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key.
            value: Tag value.
        """
        if self._mlflow is None:
            self._init_mlflow()
        self._mlflow.set_tag(key, value)

    def get_run_url(self) -> str | None:
        """Get URL to current run in MLflow UI.

        Returns:
            URL string or None if not available.
        """
        if self._run_id is None:
            return None

        try:
            tracking_uri = self._mlflow.get_tracking_uri()
            # this is a simplification - actual URL depends on server setup
            return f"{tracking_uri}/#/experiments/{self._mlflow.active_run().info.experiment_id}/runs/{self._run_id}"
        except Exception:
            return None

    @staticmethod
    def _flatten_dict(
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def create_tracker(
    experiment_name: str,
    tracking_uri: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> MLflowTracker:
    """Create an MLflow tracker with convenient defaults.

    Args:
        experiment_name: Name of experiment.
        tracking_uri: MLflow tracking URI.
        run_name: Optional run name.
        tags: Optional tags.

    Returns:
        Configured MLflowTracker.

    Example:
        tracker = create_tracker("llm-eval", run_name="gpt4-baseline")
        with tracker.start_run():
            # run evaluation
            tracker.log_metrics({"accuracy": 0.85})
    """
    config = TrackingConfig(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags or {},
    )
    return MLflowTracker(config)
