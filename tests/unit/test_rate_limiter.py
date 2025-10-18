"""Tests for rate limiter."""

import time
import threading
import pytest
from spark_llm_eval.inference.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitConfig,
    NoOpRateLimiter,
)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_no_limit_returns_zero(self):
        """When no limits set, acquire returns 0 immediately."""
        limiter = TokenBucketRateLimiter(RateLimitConfig())
        wait = limiter.acquire(1000)
        assert wait == 0.0

    def test_rpm_limit_basic(self):
        """Basic RPM limiting works."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=60,  # 1 per second
        ))

        # first request should go through
        wait = limiter.acquire()
        assert wait == 0.0

        # exhaust the bucket
        for _ in range(59):
            limiter.acquire()

        # next request should require wait
        wait = limiter.acquire()
        assert wait > 0

    def test_tpm_limit_basic(self):
        """Basic TPM limiting works."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            tokens_per_minute=1000,
        ))

        # request within limit
        wait = limiter.acquire(100)
        assert wait == 0.0

        # request that exceeds remaining
        wait = limiter.acquire(1000)
        assert wait > 0

    def test_bucket_refills(self):
        """Bucket refills over time."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=600,  # 10 per second
        ))

        # exhaust most of the bucket
        for _ in range(600):
            limiter.acquire()

        # wait a bit for refill
        time.sleep(0.1)

        # should have some tokens back
        wait = limiter.acquire()
        assert wait == 0.0

    def test_wait_and_acquire(self):
        """wait_and_acquire blocks until ready."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=60,
        ))

        # exhaust bucket
        for _ in range(60):
            limiter.acquire()

        start = time.monotonic()
        limiter.wait_and_acquire()
        elapsed = time.monotonic() - start

        # should have waited some time
        assert elapsed > 0.5  # at least half a second

    def test_report_actual_tokens(self):
        """report_actual_tokens adjusts the bucket."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            tokens_per_minute=1000,
        ))

        # acquire with estimate
        limiter.acquire(100)

        # report actual was higher
        limiter.report_actual_tokens(actual_tokens=150, estimated_tokens=100)

        # bucket should have 50 fewer tokens
        stats = limiter.stats
        # initial 1000, minus 100 from acquire, minus 50 from adjustment = 850
        assert stats["current_token_tokens"] == pytest.approx(850, abs=10)

    def test_stats(self):
        """Stats are tracked correctly."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=100,
        ))

        for _ in range(10):
            limiter.acquire()

        stats = limiter.stats
        assert stats["total_requests"] == 10
        assert stats["current_request_tokens"] == pytest.approx(90, abs=1)

    def test_reset(self):
        """Reset restores initial state."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=1000,
        ))

        # use some tokens
        for _ in range(50):
            limiter.acquire(10)

        # reset
        limiter.reset()

        stats = limiter.stats
        assert stats["total_requests"] == 0
        assert stats["current_request_tokens"] == 100
        assert stats["current_token_tokens"] == 1000

    def test_thread_safety(self):
        """Rate limiter is thread-safe."""
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=1000,
        ))

        results = []

        def worker():
            for _ in range(100):
                wait = limiter.acquire()
                results.append(wait)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # all requests should have been processed
        assert len(results) == 1000

        # stats should be consistent
        stats = limiter.stats
        assert stats["total_requests"] == 1000


class TestNoOpRateLimiter:
    """Tests for NoOpRateLimiter."""

    def test_always_returns_zero(self):
        limiter = NoOpRateLimiter()
        assert limiter.acquire(9999) == 0.0

    def test_wait_and_acquire_returns_immediately(self):
        limiter = NoOpRateLimiter()
        start = time.monotonic()
        limiter.wait_and_acquire(9999)
        elapsed = time.monotonic() - start
        assert elapsed < 0.01

    def test_stats(self):
        limiter = NoOpRateLimiter()
        assert limiter.stats["enabled"] is False
