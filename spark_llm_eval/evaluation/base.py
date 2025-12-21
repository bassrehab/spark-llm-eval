"""Base classes for evaluation metrics.

All metrics inherit from the Metric base class, which defines the
interface for computing scores on predictions vs references.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of computing a metric on a batch.

    Args:
        name: Metric identifier.
        value: Aggregate score for the batch.
        per_example_scores: Individual scores for each example.
        metadata: Additional metric-specific data.
    """

    name: str
    value: float
    per_example_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Subclasses must implement compute() to calculate scores.

    Example:
        class MyMetric(Metric):
            name = "my_metric"

            def compute(self, predictions, references):
                scores = [self._score(p, r) for p, r in zip(predictions, references)]
                return MetricResult(
                    name=self.name,
                    value=sum(scores) / len(scores),
                    per_example_scores=scores,
                )
    """

    # subclasses should override this
    name: str = "base_metric"

    def __init__(self, **kwargs):
        """Initialize metric with optional parameters.

        Args:
            **kwargs: Metric-specific configuration.
        """
        self.config = kwargs

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute metric scores.

        Args:
            predictions: Model outputs.
            references: Ground truth answers.

        Returns:
            MetricResult with aggregate and per-example scores.

        Raises:
            MetricComputationError: If computation fails.
        """
        pass

    def compute_single(self, prediction: str, reference: str) -> float:
        """Compute score for a single example.

        Default implementation calls compute() with single-item lists.
        Override for efficiency if needed.

        Args:
            prediction: Single model output.
            reference: Single ground truth.

        Returns:
            Score for this example.
        """
        result = self.compute([prediction], [reference])
        return result.per_example_scores[0] if result.per_example_scores else result.value

    def validate_inputs(
        self,
        predictions: list[str],
        references: list[str],
    ) -> None:
        """Validate inputs before computation.

        Args:
            predictions: Model outputs.
            references: Ground truth answers.

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must have same length, "
                f"got {len(predictions)} vs {len(references)}"
            )
        if not predictions:
            raise ValueError("predictions cannot be empty")


class ReferenceFreeMetic(Metric):
    """Base class for metrics that don't need references.

    Examples: fluency, coherence, toxicity detection.
    """

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
    ) -> MetricResult:
        """Compute metric scores.

        Args:
            predictions: Model outputs.
            references: Ignored for reference-free metrics.

        Returns:
            MetricResult with scores.
        """
        pass

    def validate_inputs(
        self,
        predictions: list[str],
        references: list[str] | None = None,
    ) -> None:
        """Validate inputs - only checks predictions."""
        if not predictions:
            raise ValueError("predictions cannot be empty")


# registry for metric lookup by name
_METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(cls_or_name=None):
    """Decorator to register a metric class.

    Usage:
        @register_metric
        class MyMetric(Metric):
            name = "my_metric"
            ...

        # or with explicit name
        @register_metric("custom_name")
        class MyMetric(Metric):
            ...
    """

    def decorator(cls: type[Metric]) -> type[Metric]:
        name = cls_or_name if isinstance(cls_or_name, str) else cls.name
        _METRIC_REGISTRY[name] = cls
        return cls

    # handle both @register_metric and @register_metric("name")
    if cls_or_name is None or isinstance(cls_or_name, str):
        return decorator
    else:
        # called as @register_metric without parentheses
        return decorator(cls_or_name)


def get_metric(name: str, **kwargs) -> Metric:
    """Get a metric instance by name.

    Args:
        name: Metric name (e.g., "exact_match", "bleu").
        **kwargs: Passed to metric constructor.

    Returns:
        Metric instance.

    Raises:
        KeyError: If metric not found.
    """
    if name not in _METRIC_REGISTRY:
        available = list(_METRIC_REGISTRY.keys())
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return _METRIC_REGISTRY[name](**kwargs)


def list_metrics() -> list[str]:
    """List all registered metric names."""
    return list(_METRIC_REGISTRY.keys())
