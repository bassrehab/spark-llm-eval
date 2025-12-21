"""Result types for evaluation outputs."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MetricValue:
    """A metric value with CI. The main output type for all metrics."""

    value: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    standard_error: float
    sample_size: int

    def __str__(self):
        lo, hi = self.confidence_interval
        return f"{self.value:.4f} [{lo:.4f}, {hi:.4f}]"

    def __repr__(self):
        return (
            f"MetricValue(value={self.value:.4f}, "
            f"ci=[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}], "
            f"n={self.sample_size})"
        )

    @property
    def ci_width(self):
        return self.confidence_interval[1] - self.confidence_interval[0]

    def overlaps(self, other: "MetricValue") -> bool:
        """Quick check if CIs overlap (not a proper significance test though)."""
        lo1, hi1 = self.confidence_interval
        lo2, hi2 = other.confidence_interval
        return not (hi1 < lo2 or hi2 < lo1)


@dataclass
class ComparisonResult:
    """Result of comparing two runs (is model A better than B?)"""

    metric_name: str
    baseline_value: MetricValue
    comparison_value: MetricValue
    difference: float
    relative_change: float
    p_value: float
    effect_size: float  # cohen's d
    is_significant: bool

    def __str__(self):
        sig = "significant" if self.is_significant else "not significant"
        direction = "better" if self.difference > 0 else "worse"
        return f"{self.metric_name}: {self.difference:+.4f} ({direction}), p={self.p_value:.4f} ({sig})"


@dataclass
class CostBreakdown:
    """Cost tracking for eval run.

    Includes both actual API costs and estimated savings from cache hits.
    """

    total_cost_usd: float
    input_tokens: int
    output_tokens: int
    num_requests: int
    cached_requests: int = 0
    estimated_savings_usd: float = 0.0  # Cost avoided via cache hits

    @property
    def cost_per_example(self) -> float:
        """Average cost per API request."""
        if self.num_requests == 0:
            return 0.0
        return self.total_cost_usd / self.num_requests

    @property
    def effective_cost_per_example(self) -> float:
        """Average cost per example including cached requests (amortized)."""
        total_examples = self.num_requests + self.cached_requests
        if total_examples == 0:
            return 0.0
        return self.total_cost_usd / total_examples

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a fraction (0-1)."""
        total = self.num_requests + self.cached_requests
        return self.cached_requests / total if total > 0 else 0.0

    @property
    def cache_savings_pct(self) -> float:
        """Percentage of cost saved by caching (0-100)."""
        total_potential = self.total_cost_usd + self.estimated_savings_usd
        if total_potential == 0:
            return 0.0
        return (self.estimated_savings_usd / total_potential) * 100

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class LatencyStats:
    """Latency stats in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_duration_s: float

    def __str__(self):
        return f"mean={self.mean_ms:.1f}ms, p95={self.p95_ms:.1f}ms"


@dataclass
class EvalResult:
    """What you get back from runner.run(). Has metrics, costs, latencies etc."""

    task_id: str
    run_id: str | None
    timestamp: datetime
    metrics: dict[str, MetricValue]
    stratified_metrics: dict[str, dict[str, MetricValue]]
    cost: CostBreakdown
    latency: LatencyStats
    predictions_table: str | None
    config_snapshot: dict[str, Any]
    num_examples: int
    num_failures: int = 0

    @property
    def failure_rate(self):
        if self.num_examples == 0:
            return 0.0
        return self.num_failures / self.num_examples

    def get_metric(self, name: str):
        return self.metrics.get(name)

    def get_stratified_metric(self, stratum: str, metric_name: str):
        if stratum not in self.stratified_metrics:
            return None
        return self.stratified_metrics[stratum].get(metric_name)

    def summary(self):
        lines = [
            f"Evaluation: {self.task_id}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Examples: {self.num_examples} ({self.num_failures} failures)",
            "",
            "Metrics:",
        ]
        for name, val in self.metrics.items():
            lines.append(f"  {name}: {val}")

        if self.stratified_metrics:
            lines.append("")
            lines.append("Stratified:")
            for stratum, metrics in self.stratified_metrics.items():
                lines.append(f"  {stratum}:")
                for name, val in metrics.items():
                    lines.append(f"    {name}: {val}")

        lines.extend(
            [
                "",
                f"Cost: ${self.cost.total_cost_usd:.4f} ({self.cost.num_requests} requests)",
                f"Latency: {self.latency}",
            ]
        )
        return "\n".join(lines)
