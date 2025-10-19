"""Tests for metric aggregation."""

import pytest
import numpy as np
from spark_llm_eval.core.config import StatisticsConfig
from spark_llm_eval.evaluation.base import MetricResult
from spark_llm_eval.evaluation.aggregator import (
    MetricAggregator,
    AggregatedMetrics,
    compute_metrics,
)


class TestMetricAggregator:
    """Tests for MetricAggregator."""

    @pytest.fixture
    def sample_results(self):
        return [
            MetricResult(
                name="exact_match",
                value=0.75,
                per_example_scores=[1.0, 1.0, 1.0, 0.0],
            ),
            MetricResult(
                name="f1",
                value=0.85,
                per_example_scores=[0.9, 0.8, 0.9, 0.8],
            ),
        ]

    def test_aggregate_basic(self, sample_results):
        aggregator = MetricAggregator()
        aggregated = aggregator.aggregate(sample_results)

        assert "exact_match" in aggregated.metrics
        assert "f1" in aggregated.metrics
        assert aggregated.metrics["exact_match"].value == pytest.approx(0.75)
        assert aggregated.metrics["f1"].value == pytest.approx(0.85)

    def test_confidence_intervals(self, sample_results):
        aggregator = MetricAggregator(StatisticsConfig(
            confidence_level=0.95,
            bootstrap_iterations=1000,
        ))
        aggregated = aggregator.aggregate(sample_results)

        em = aggregated.metrics["exact_match"]
        assert em.confidence_interval[0] <= em.value <= em.confidence_interval[1]
        assert em.confidence_level == 0.95

    def test_sample_size(self, sample_results):
        aggregator = MetricAggregator()
        aggregated = aggregator.aggregate(sample_results)

        assert aggregated.metrics["exact_match"].sample_size == 4
        assert aggregated.metrics["f1"].sample_size == 4

    def test_standard_error(self, sample_results):
        aggregator = MetricAggregator()
        aggregated = aggregator.aggregate(sample_results)

        # exact_match has scores [1, 1, 1, 0], se should be reasonable
        em = aggregated.metrics["exact_match"]
        assert 0 < em.standard_error < 0.5

    def test_stratified_aggregation(self, sample_results):
        aggregator = MetricAggregator()
        strata = ["easy", "easy", "hard", "hard"]
        aggregated = aggregator.aggregate(sample_results, strata=strata)

        assert "easy" in aggregated.stratified
        assert "hard" in aggregated.stratified
        assert "exact_match" in aggregated.stratified["easy"]
        assert "exact_match" in aggregated.stratified["hard"]

        # easy examples: [1.0, 1.0] -> mean 1.0
        # hard examples: [1.0, 0.0] -> mean 0.5
        assert aggregated.stratified["easy"]["exact_match"].value == pytest.approx(1.0)
        assert aggregated.stratified["hard"]["exact_match"].value == pytest.approx(0.5)

    def test_empty_results(self):
        aggregator = MetricAggregator()
        result = MetricResult(name="test", value=0.0, per_example_scores=[])
        aggregated = aggregator.aggregate([result])

        assert aggregated.metrics["test"].value == 0.0
        assert aggregated.metrics["test"].sample_size == 0


class TestComputeMetrics:
    """Tests for compute_metrics convenience function."""

    def test_compute_multiple_metrics(self):
        predictions = ["hello world", "foo bar", "test"]
        references = ["hello world", "bar foo", "test"]

        result = compute_metrics(
            predictions=predictions,
            references=references,
            metric_names=["exact_match", "f1"],
        )

        assert isinstance(result, AggregatedMetrics)
        assert "exact_match" in result.metrics
        assert "f1" in result.metrics

    def test_compute_with_stratification(self):
        predictions = ["a", "b", "c", "d"]
        references = ["a", "x", "c", "y"]
        strata = ["cat1", "cat1", "cat2", "cat2"]

        result = compute_metrics(
            predictions=predictions,
            references=references,
            metric_names=["exact_match"],
            strata=strata,
        )

        assert "cat1" in result.stratified
        assert "cat2" in result.stratified

    def test_compute_with_metric_kwargs(self):
        predictions = ["Hello World"]
        references = ["hello world"]

        # without normalization, should not match
        result = compute_metrics(
            predictions=predictions,
            references=references,
            metric_names=["exact_match"],
            metric_kwargs={"exact_match": {"normalize": False, "case_sensitive": True}},
        )
        assert result.metrics["exact_match"].value == 0.0

        # with normalization, should match
        result = compute_metrics(
            predictions=predictions,
            references=references,
            metric_names=["exact_match"],
            metric_kwargs={"exact_match": {"normalize": True}},
        )
        assert result.metrics["exact_match"].value == 1.0

    def test_compute_with_stats_config(self):
        predictions = ["a", "b", "a", "b"]
        references = ["a", "b", "a", "x"]

        result = compute_metrics(
            predictions=predictions,
            references=references,
            metric_names=["exact_match"],
            stats_config=StatisticsConfig(
                confidence_level=0.99,
                bootstrap_iterations=500,
            ),
        )

        assert result.metrics["exact_match"].confidence_level == 0.99


class TestBootstrapCI:
    """Tests for bootstrap confidence interval calculation."""

    def test_ci_contains_point_estimate(self):
        aggregator = MetricAggregator(StatisticsConfig(
            confidence_level=0.95,
            bootstrap_iterations=2000,
        ))

        # create result with known distribution
        np.random.seed(42)
        scores = np.random.normal(0.7, 0.1, 100).tolist()
        result = MetricResult(name="test", value=np.mean(scores), per_example_scores=scores)

        aggregated = aggregator.aggregate([result])
        mv = aggregated.metrics["test"]

        assert mv.confidence_interval[0] <= mv.value <= mv.confidence_interval[1]

    def test_wider_ci_for_higher_confidence(self):
        scores = [0.6, 0.7, 0.8, 0.65, 0.75, 0.85, 0.7, 0.8]
        result = MetricResult(name="test", value=np.mean(scores), per_example_scores=scores)

        agg_95 = MetricAggregator(StatisticsConfig(confidence_level=0.95))
        agg_99 = MetricAggregator(StatisticsConfig(confidence_level=0.99))

        ci_95 = agg_95.aggregate([result]).metrics["test"].confidence_interval
        ci_99 = agg_99.aggregate([result]).metrics["test"].confidence_interval

        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_99 > width_95

    def test_single_value(self):
        aggregator = MetricAggregator()
        result = MetricResult(name="test", value=0.5, per_example_scores=[0.5])
        aggregated = aggregator.aggregate([result])

        # single value should have point interval
        mv = aggregated.metrics["test"]
        assert mv.confidence_interval[0] == pytest.approx(0.5)
        assert mv.confidence_interval[1] == pytest.approx(0.5)
