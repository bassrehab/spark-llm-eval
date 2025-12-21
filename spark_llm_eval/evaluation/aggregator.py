"""Metric aggregation with statistical analysis.

Combines per-example scores into aggregate metrics with
confidence intervals and optional stratification.
"""

import logging
from dataclasses import dataclass

import numpy as np

from spark_llm_eval.core.config import StatisticsConfig
from spark_llm_eval.core.result import MetricValue
from spark_llm_eval.evaluation.base import MetricResult, get_metric

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Container for aggregated metrics with statistics.

    Args:
        metrics: Metric name -> MetricValue with CIs.
        stratified: Stratum -> metric name -> MetricValue.
        raw_results: Original MetricResult objects.
    """

    metrics: dict[str, MetricValue]
    stratified: dict[str, dict[str, MetricValue]]
    raw_results: list[MetricResult]


class MetricAggregator:
    """Aggregates metric scores with statistical analysis.

    Takes per-example scores and computes:
    - Point estimates (mean)
    - Confidence intervals (bootstrap)
    - Stratified breakdowns

    Example:
        aggregator = MetricAggregator(stats_config)
        results = [metric.compute(preds, refs) for metric in metrics]
        aggregated = aggregator.aggregate(results, strata=df["category"].tolist())
    """

    def __init__(self, stats_config: StatisticsConfig | None = None):
        self.stats_config = stats_config or StatisticsConfig()

    def aggregate(
        self,
        results: list[MetricResult],
        strata: list[str] | None = None,
    ) -> AggregatedMetrics:
        """Aggregate metric results with statistics.

        Args:
            results: MetricResult objects from metric.compute().
            strata: Optional stratum labels for each example.

        Returns:
            AggregatedMetrics with CIs and stratification.
        """
        metrics = {}
        for result in results:
            scores = np.array(result.per_example_scores)
            metrics[result.name] = self._compute_metric_value(scores)

        # stratified analysis
        stratified = {}
        if strata:
            stratified = self._compute_stratified(results, strata)

        return AggregatedMetrics(
            metrics=metrics,
            stratified=stratified,
            raw_results=results,
        )

    def _compute_metric_value(self, scores: np.ndarray) -> MetricValue:
        """Compute MetricValue with confidence interval."""
        if len(scores) == 0:
            return MetricValue(
                value=0.0,
                confidence_interval=(0.0, 0.0),
                confidence_level=self.stats_config.confidence_level,
                standard_error=0.0,
                sample_size=0,
            )

        point_estimate = float(np.mean(scores))
        ci = self._bootstrap_ci(scores)
        se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))

        return MetricValue(
            value=point_estimate,
            confidence_interval=ci,
            confidence_level=self.stats_config.confidence_level,
            standard_error=se,
            sample_size=len(scores),
        )

    def _bootstrap_ci(self, scores: np.ndarray) -> tuple[float, float]:
        """Compute bootstrap confidence interval."""
        n = len(scores)
        if n < 2:
            mean = float(scores[0]) if n == 1 else 0.0
            return (mean, mean)

        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        n_iterations = self.stats_config.bootstrap_iterations

        boot_means = []
        for _ in range(n_iterations):
            sample = rng.choice(scores, size=n, replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)

        alpha = 1 - self.stats_config.confidence_level
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        return (lower, upper)

    def _compute_stratified(
        self,
        results: list[MetricResult],
        strata: list[str],
    ) -> dict[str, dict[str, MetricValue]]:
        """Compute metrics stratified by group."""
        # organize scores by stratum
        stratum_scores: dict[str, dict[str, list[float]]] = {}

        unique_strata = set(strata)
        for stratum in unique_strata:
            stratum_scores[stratum] = {r.name: [] for r in results}

        for i, stratum in enumerate(strata):
            for result in results:
                if i < len(result.per_example_scores):
                    stratum_scores[stratum][result.name].append(result.per_example_scores[i])

        # compute metrics for each stratum
        stratified = {}
        for stratum, metric_scores in stratum_scores.items():
            stratified[stratum] = {}
            for metric_name, scores in metric_scores.items():
                if scores:
                    stratified[stratum][metric_name] = self._compute_metric_value(np.array(scores))

        return stratified


def compute_metrics(
    predictions: list[str],
    references: list[str],
    metric_names: list[str],
    stats_config: StatisticsConfig | None = None,
    strata: list[str] | None = None,
    metric_kwargs: dict[str, dict] | None = None,
) -> AggregatedMetrics:
    """Convenience function to compute multiple metrics.

    Args:
        predictions: Model outputs.
        references: Ground truth.
        metric_names: List of metric names to compute.
        stats_config: Statistics configuration.
        strata: Optional stratum labels for stratified analysis.
        metric_kwargs: Per-metric keyword arguments.

    Returns:
        AggregatedMetrics with all results.

    Example:
        results = compute_metrics(
            predictions=["hello world", "foo bar"],
            references=["hello world", "baz qux"],
            metric_names=["exact_match", "f1", "bleu"],
        )
        print(results.metrics["exact_match"])
    """
    metric_kwargs = metric_kwargs or {}

    results = []
    for name in metric_names:
        kwargs = metric_kwargs.get(name, {})
        metric = get_metric(name, **kwargs)
        result = metric.compute(predictions, references)
        results.append(result)

    aggregator = MetricAggregator(stats_config)
    return aggregator.aggregate(results, strata)
