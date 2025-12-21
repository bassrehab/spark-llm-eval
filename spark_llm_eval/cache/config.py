"""Cache configuration for response caching."""

from dataclasses import dataclass
from enum import Enum


class CachePolicy(Enum):
    """Cache lookup and storage policies."""

    ENABLED = "enabled"  # Normal caching: lookup and store
    DISABLED = "disabled"  # No caching
    READ_ONLY = "read_only"  # Lookup only, don't store new responses
    WRITE_ONLY = "write_only"  # Store only, don't lookup (cache warm-up)
    REPLAY = "replay"  # Require cache hit, error on miss (metrics iteration)


@dataclass
class CacheConfig:
    """Configuration for Delta-backed response caching.

    Args:
        policy: Cache behavior policy (enabled, disabled, read_only, write_only, replay)
        table_path: Path to Delta cache table (required unless policy is DISABLED)
        ttl_hours: Time-to-live in hours (None = no expiry)
        cache_version: Version string for invalidation (bump to invalidate old entries)
        track_statistics: Whether to collect hit/miss statistics

    Example:
        >>> config = CacheConfig(
        ...     policy=CachePolicy.ENABLED,
        ...     table_path="dbfs:/mnt/cache/llm_responses",
        ...     ttl_hours=24,
        ... )
    """

    policy: CachePolicy = CachePolicy.ENABLED
    table_path: str | None = None
    ttl_hours: int | None = None
    cache_version: str = "1.0"
    track_statistics: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.policy != CachePolicy.DISABLED and not self.table_path:
            raise ValueError("table_path is required when cache policy is not DISABLED")
        if self.ttl_hours is not None and self.ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
