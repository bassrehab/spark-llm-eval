"""Delta Lake-backed response caching for LLM evaluations.

This module provides content-addressable caching of LLM responses using Delta Lake.
Caching avoids redundant API calls, reducing cost and latency for repeated evaluations.

Key components:
- CacheConfig: Configuration for cache behavior
- CachePolicy: Cache modes (enabled, disabled, read_only, write_only, replay)
- DeltaCacheManager: Main cache manager for lookup and storage
- CacheStatistics: Hit/miss tracking and cost savings

Example:
    >>> from spark_llm_eval.cache import CacheConfig, CachePolicy

    >>> # Standard caching with 24-hour TTL
    >>> config = CacheConfig(
    ...     policy=CachePolicy.ENABLED,
    ...     table_path="dbfs:/mnt/cache/llm_responses",
    ...     ttl_hours=24,
    ... )

    >>> # Replay mode for metrics iteration without API calls
    >>> replay_config = CacheConfig(
    ...     policy=CachePolicy.REPLAY,
    ...     table_path="dbfs:/mnt/cache/llm_responses",
    ... )
"""

from spark_llm_eval.cache.config import CacheConfig, CachePolicy
from spark_llm_eval.cache.delta_cache import (
    CACHE_TABLE_SCHEMA,
    CacheError,
    DeltaCacheManager,
    create_cache_table,
)
from spark_llm_eval.cache.key_generator import generate_cache_key, generate_prompt_hash
from spark_llm_eval.cache.stats import CacheStatistics

__all__ = [
    # Config
    "CacheConfig",
    "CachePolicy",
    # Cache manager
    "DeltaCacheManager",
    "CacheError",
    "CACHE_TABLE_SCHEMA",
    "create_cache_table",
    # Key generation
    "generate_cache_key",
    "generate_prompt_hash",
    # Statistics
    "CacheStatistics",
]
