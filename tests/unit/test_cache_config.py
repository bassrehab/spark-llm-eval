"""Tests for cache configuration."""

import pytest

from spark_llm_eval.cache.config import CacheConfig, CachePolicy
from spark_llm_eval.cache.stats import CacheStatistics
from spark_llm_eval.core.config import InferenceConfig


class TestCachePolicy:
    """Tests for CachePolicy enum."""

    def test_policy_values(self):
        """Test that all expected policies exist."""
        assert CachePolicy.ENABLED.value == "enabled"
        assert CachePolicy.DISABLED.value == "disabled"
        assert CachePolicy.READ_ONLY.value == "read_only"
        assert CachePolicy.WRITE_ONLY.value == "write_only"
        assert CachePolicy.REPLAY.value == "replay"

    def test_policy_count(self):
        """Test that we have exactly 5 policies."""
        assert len(CachePolicy) == 5


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig(table_path="/tmp/cache")
        assert config.policy == CachePolicy.ENABLED
        assert config.table_path == "/tmp/cache"
        assert config.ttl_hours is None
        assert config.cache_version == "1.0"
        assert config.track_statistics is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            policy=CachePolicy.READ_ONLY,
            table_path="/mnt/cache",
            ttl_hours=24,
            cache_version="2.0",
            track_statistics=False,
        )
        assert config.policy == CachePolicy.READ_ONLY
        assert config.table_path == "/mnt/cache"
        assert config.ttl_hours == 24
        assert config.cache_version == "2.0"
        assert config.track_statistics is False

    def test_disabled_policy_no_path_required(self):
        """Disabled policy should not require table_path."""
        config = CacheConfig(policy=CachePolicy.DISABLED, table_path=None)
        assert config.policy == CachePolicy.DISABLED
        assert config.table_path is None

    def test_enabled_policy_requires_path(self):
        """Enabled policy should require table_path."""
        with pytest.raises(ValueError, match="table_path is required"):
            CacheConfig(policy=CachePolicy.ENABLED, table_path=None)

    def test_read_only_requires_path(self):
        """Read-only policy should require table_path."""
        with pytest.raises(ValueError, match="table_path is required"):
            CacheConfig(policy=CachePolicy.READ_ONLY, table_path=None)

    def test_write_only_requires_path(self):
        """Write-only policy should require table_path."""
        with pytest.raises(ValueError, match="table_path is required"):
            CacheConfig(policy=CachePolicy.WRITE_ONLY, table_path=None)

    def test_replay_requires_path(self):
        """Replay policy should require table_path."""
        with pytest.raises(ValueError, match="table_path is required"):
            CacheConfig(policy=CachePolicy.REPLAY, table_path=None)

    def test_invalid_ttl_zero(self):
        """TTL of 0 should raise error."""
        with pytest.raises(ValueError, match="ttl_hours must be positive"):
            CacheConfig(table_path="/tmp/cache", ttl_hours=0)

    def test_invalid_ttl_negative(self):
        """Negative TTL should raise error."""
        with pytest.raises(ValueError, match="ttl_hours must be positive"):
            CacheConfig(table_path="/tmp/cache", ttl_hours=-1)

    def test_valid_ttl(self):
        """Positive TTL should be accepted."""
        config = CacheConfig(table_path="/tmp/cache", ttl_hours=24)
        assert config.ttl_hours == 24


class TestCacheStatistics:
    """Tests for CacheStatistics dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = CacheStatistics()
        assert stats.lookup_count == 0
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.store_count == 0
        assert stats.estimated_cost_saved_usd == 0.0
        assert stats.actual_api_cost_usd == 0.0

    def test_hit_rate_zero_lookups(self):
        """Hit rate should be 0 with no lookups."""
        stats = CacheStatistics()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStatistics()
        stats.record_lookups(hits=3, misses=7)
        assert stats.hit_rate == 30.0

    def test_miss_rate_calculation(self):
        """Test miss rate calculation."""
        stats = CacheStatistics()
        stats.record_lookups(hits=3, misses=7)
        assert stats.miss_rate == 70.0

    def test_total_requests(self):
        """Test total requests calculation."""
        stats = CacheStatistics()
        stats.record_lookups(hits=5, misses=15)
        assert stats.total_requests == 20

    def test_cost_savings_pct_no_cost(self):
        """Cost savings should be 0 with no costs."""
        stats = CacheStatistics()
        assert stats.cost_savings_pct == 0.0

    def test_cost_savings_pct_calculation(self):
        """Test cost savings percentage calculation."""
        stats = CacheStatistics()
        stats.record_cost_savings(0.5)  # $0.50 saved
        stats.record_api_cost(0.5)  # $0.50 spent
        assert stats.cost_savings_pct == 50.0

    def test_record_lookups(self):
        """Test recording lookups."""
        stats = CacheStatistics()
        stats.record_lookups(hits=10, misses=5)
        assert stats.hit_count == 10
        assert stats.miss_count == 5
        assert stats.lookup_count == 15

    def test_record_lookups_cumulative(self):
        """Test that lookups are cumulative."""
        stats = CacheStatistics()
        stats.record_lookups(hits=5, misses=5)
        stats.record_lookups(hits=3, misses=2)
        assert stats.hit_count == 8
        assert stats.miss_count == 7
        assert stats.lookup_count == 15

    def test_record_stores(self):
        """Test recording stores."""
        stats = CacheStatistics()
        stats.record_stores(10)
        assert stats.store_count == 10

    def test_record_stores_cumulative(self):
        """Test that stores are cumulative."""
        stats = CacheStatistics()
        stats.record_stores(5)
        stats.record_stores(3)
        assert stats.store_count == 8

    def test_to_dict(self):
        """Test serialization to dict."""
        stats = CacheStatistics()
        stats.record_lookups(hits=7, misses=3)
        stats.record_stores(7)
        stats.record_cost_savings(0.35)
        stats.record_api_cost(0.15)

        d = stats.to_dict()
        assert d["cache_lookup_count"] == 10
        assert d["cache_hit_count"] == 7
        assert d["cache_miss_count"] == 3
        assert d["cache_hit_rate_pct"] == 70.0
        assert d["cache_store_count"] == 7
        assert d["estimated_cost_saved_usd"] == 0.35
        assert d["actual_api_cost_usd"] == 0.15
        assert d["cost_savings_pct"] == 70.0
        assert "session_duration_s" in d

    def test_str_representation(self):
        """Test string representation."""
        stats = CacheStatistics()
        stats.record_lookups(hits=7, misses=3)
        stats.record_cost_savings(0.35)
        s = str(stats)
        assert "hits=7" in s
        assert "misses=3" in s
        assert "70.0%" in s
        assert "$0.35" in s


class TestInferenceConfigCaching:
    """Tests for InferenceConfig cache integration."""

    def test_get_effective_cache_config_new_style(self):
        """Test new-style cache_config takes precedence."""
        cache_config = CacheConfig(
            policy=CachePolicy.ENABLED,
            table_path="/mnt/cache",
            ttl_hours=24,
        )
        inference_config = InferenceConfig(cache_config=cache_config)

        effective = inference_config.get_effective_cache_config()
        assert effective.policy == CachePolicy.ENABLED
        assert effective.table_path == "/mnt/cache"
        assert effective.ttl_hours == 24

    def test_get_effective_cache_config_legacy_disabled(self):
        """Test legacy enable_caching=False."""
        inference_config = InferenceConfig(enable_caching=False)

        effective = inference_config.get_effective_cache_config()
        assert effective.policy == CachePolicy.DISABLED

    def test_get_effective_cache_config_legacy_with_table(self):
        """Test legacy cache_table field."""
        inference_config = InferenceConfig(
            enable_caching=True,
            cache_table="/legacy/cache",
        )

        effective = inference_config.get_effective_cache_config()
        assert effective.policy == CachePolicy.ENABLED
        assert effective.table_path == "/legacy/cache"

    def test_get_effective_cache_config_no_cache(self):
        """Test default - caching enabled but no table."""
        inference_config = InferenceConfig()

        effective = inference_config.get_effective_cache_config()
        assert effective.policy == CachePolicy.DISABLED

    def test_new_style_overrides_legacy(self):
        """Test that new cache_config overrides legacy fields."""
        cache_config = CacheConfig(
            policy=CachePolicy.REPLAY,
            table_path="/new/cache",
        )
        inference_config = InferenceConfig(
            cache_config=cache_config,
            enable_caching=False,  # Should be ignored
            cache_table="/legacy/cache",  # Should be ignored
        )

        effective = inference_config.get_effective_cache_config()
        assert effective.policy == CachePolicy.REPLAY
        assert effective.table_path == "/new/cache"
