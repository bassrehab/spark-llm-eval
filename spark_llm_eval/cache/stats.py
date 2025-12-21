"""Cache statistics tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CacheStatistics:
    """Tracks cache performance statistics for an evaluation session.

    Attributes:
        lookup_count: Total number of cache lookups performed
        hit_count: Number of cache hits (responses found in cache)
        miss_count: Number of cache misses (responses not in cache)
        store_count: Number of responses stored in cache
        estimated_cost_saved_usd: Estimated cost saved by using cached responses
        actual_api_cost_usd: Actual cost incurred from API calls
        session_start: When this statistics session started
    """

    lookup_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    store_count: int = 0
    estimated_cost_saved_usd: float = 0.0
    actual_api_cost_usd: float = 0.0
    session_start: datetime = field(default_factory=datetime.utcnow)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage (0-100)."""
        if self.lookup_count == 0:
            return 0.0
        return (self.hit_count / self.lookup_count) * 100

    @property
    def miss_rate(self) -> float:
        """Cache miss rate as percentage (0-100)."""
        if self.lookup_count == 0:
            return 0.0
        return (self.miss_count / self.lookup_count) * 100

    @property
    def total_requests(self) -> int:
        """Total logical requests (hits + misses)."""
        return self.hit_count + self.miss_count

    @property
    def cost_savings_pct(self) -> float:
        """Percentage of cost saved by caching."""
        total_potential = self.actual_api_cost_usd + self.estimated_cost_saved_usd
        if total_potential == 0:
            return 0.0
        return (self.estimated_cost_saved_usd / total_potential) * 100

    def record_lookups(self, hits: int, misses: int) -> None:
        """Record lookup results.

        Args:
            hits: Number of cache hits
            misses: Number of cache misses
        """
        self.hit_count += hits
        self.miss_count += misses
        self.lookup_count += hits + misses

    def record_stores(self, count: int) -> None:
        """Record cache stores.

        Args:
            count: Number of responses stored
        """
        self.store_count += count

    def record_cost_savings(self, saved_usd: float) -> None:
        """Record estimated cost savings from cache hits.

        Args:
            saved_usd: Estimated cost saved in USD
        """
        self.estimated_cost_saved_usd += saved_usd

    def record_api_cost(self, cost_usd: float) -> None:
        """Record actual API cost incurred.

        Args:
            cost_usd: Actual cost in USD
        """
        self.actual_api_cost_usd += cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics to dictionary for logging/tracking.

        Returns:
            Dictionary with all statistics
        """
        return {
            "cache_lookup_count": self.lookup_count,
            "cache_hit_count": self.hit_count,
            "cache_miss_count": self.miss_count,
            "cache_hit_rate_pct": round(self.hit_rate, 2),
            "cache_store_count": self.store_count,
            "estimated_cost_saved_usd": round(self.estimated_cost_saved_usd, 4),
            "actual_api_cost_usd": round(self.actual_api_cost_usd, 4),
            "cost_savings_pct": round(self.cost_savings_pct, 2),
            "session_duration_s": (datetime.utcnow() - self.session_start).total_seconds(),
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"CacheStats(hits={self.hit_count}, misses={self.miss_count}, "
            f"hit_rate={self.hit_rate:.1f}%, saved=${self.estimated_cost_saved_usd:.4f})"
        )
