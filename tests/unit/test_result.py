"""Tests for result types."""

import pytest
from datetime import datetime
from spark_llm_eval.core.result import (
    MetricValue,
    ComparisonResult,
    CostBreakdown,
    LatencyStats,
    EvalResult,
)


class TestMetricValue:
    """Tests for MetricValue."""

    def test_create_metric_value(self):
        mv = MetricValue(
            value=0.85,
            confidence_interval=(0.82, 0.88),
            confidence_level=0.95,
            standard_error=0.015,
            sample_size=1000,
        )
        assert mv.value == 0.85
        assert mv.confidence_interval == (0.82, 0.88)
        assert mv.sample_size == 1000

    def test_str_format(self):
        mv = MetricValue(
            value=0.8523,
            confidence_interval=(0.8234, 0.8812),
            confidence_level=0.95,
            standard_error=0.015,
            sample_size=500,
        )
        s = str(mv)
        assert "0.8523" in s
        assert "0.8234" in s
        assert "0.8812" in s

    def test_ci_width(self):
        mv = MetricValue(
            value=0.5,
            confidence_interval=(0.4, 0.6),
            confidence_level=0.95,
            standard_error=0.05,
            sample_size=100,
        )
        assert mv.ci_width == pytest.approx(0.2)

    def test_overlaps_true(self):
        mv1 = MetricValue(0.5, (0.4, 0.6), 0.95, 0.05, 100)
        mv2 = MetricValue(0.55, (0.45, 0.65), 0.95, 0.05, 100)
        assert mv1.overlaps(mv2)

    def test_overlaps_false(self):
        mv1 = MetricValue(0.5, (0.4, 0.55), 0.95, 0.05, 100)
        mv2 = MetricValue(0.7, (0.65, 0.75), 0.95, 0.05, 100)
        assert not mv1.overlaps(mv2)


class TestCostBreakdown:
    """Tests for CostBreakdown."""

    def test_cost_per_example(self):
        cost = CostBreakdown(
            total_cost_usd=10.0,
            input_tokens=50000,
            output_tokens=25000,
            num_requests=1000,
        )
        assert cost.cost_per_example == 0.01

    def test_cost_per_example_zero_requests(self):
        cost = CostBreakdown(
            total_cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            num_requests=0,
        )
        assert cost.cost_per_example == 0.0

    def test_cache_hit_rate(self):
        cost = CostBreakdown(
            total_cost_usd=5.0,
            input_tokens=25000,
            output_tokens=12500,
            num_requests=500,
            cached_requests=500,  # 50% cache hit
        )
        assert cost.cache_hit_rate == 0.5


class TestLatencyStats:
    """Tests for LatencyStats."""

    def test_str_format(self):
        stats = LatencyStats(
            mean_ms=150.5,
            median_ms=140.0,
            p95_ms=280.0,
            p99_ms=450.0,
            min_ms=50.0,
            max_ms=800.0,
            total_duration_s=300.0,
        )
        s = str(stats)
        assert "150.5" in s
        assert "280.0" in s


class TestEvalResult:
    """Tests for EvalResult."""

    @pytest.fixture
    def sample_result(self):
        return EvalResult(
            task_id="test-eval",
            run_id="mlflow-123",
            timestamp=datetime(2025, 10, 18, 14, 30, 0),
            metrics={
                "exact_match": MetricValue(0.75, (0.72, 0.78), 0.95, 0.015, 1000),
                "f1": MetricValue(0.82, (0.80, 0.84), 0.95, 0.01, 1000),
            },
            stratified_metrics={
                "easy": {
                    "exact_match": MetricValue(0.90, (0.87, 0.93), 0.95, 0.015, 500),
                },
                "hard": {
                    "exact_match": MetricValue(0.60, (0.55, 0.65), 0.95, 0.025, 500),
                },
            },
            cost=CostBreakdown(10.0, 50000, 25000, 1000),
            latency=LatencyStats(150.0, 140.0, 280.0, 450.0, 50.0, 800.0, 300.0),
            predictions_table="/delta/predictions/test-eval",
            config_snapshot={"task_id": "test-eval"},
            num_examples=1000,
            num_failures=5,
        )

    def test_failure_rate(self, sample_result):
        assert sample_result.failure_rate == 0.005

    def test_get_metric(self, sample_result):
        em = sample_result.get_metric("exact_match")
        assert em is not None
        assert em.value == 0.75

    def test_get_metric_missing(self, sample_result):
        assert sample_result.get_metric("nonexistent") is None

    def test_get_stratified_metric(self, sample_result):
        em_easy = sample_result.get_stratified_metric("easy", "exact_match")
        assert em_easy is not None
        assert em_easy.value == 0.90

    def test_get_stratified_metric_missing_stratum(self, sample_result):
        assert sample_result.get_stratified_metric("medium", "exact_match") is None

    def test_summary(self, sample_result):
        summary = sample_result.summary()
        assert "test-eval" in summary
        assert "exact_match" in summary
        assert "0.75" in summary
        assert "easy" in summary
