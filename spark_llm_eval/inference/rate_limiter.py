"""Token bucket rate limiter for API calls.

Supports both requests-per-minute (RPM) and tokens-per-minute (TPM)
limits, which is what most LLM APIs use.
"""

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Args:
        requests_per_minute: Max requests per minute (None = unlimited).
        tokens_per_minute: Max tokens per minute (None = unlimited).
    """

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter.

    Uses the token bucket algorithm to smooth out request rates.
    Buckets refill continuously, so you can burst up to the limit
    and then the rate naturally throttles.

    Example:
        limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=100000,
        ))

        # before each request
        wait_time = limiter.acquire(estimated_tokens=500)
        if wait_time > 0:
            time.sleep(wait_time)
        # ... make request ...
    """

    def __init__(self, config: RateLimitConfig):
        self.rpm = config.requests_per_minute
        self.tpm = config.tokens_per_minute

        # request bucket
        if self.rpm:
            self._request_tokens = float(self.rpm)
            self._request_rate = self.rpm / 60.0  # tokens per second
        else:
            self._request_tokens = float("inf")
            self._request_rate = float("inf")

        # token bucket
        if self.tpm:
            self._token_tokens = float(self.tpm)
            self._token_rate = self.tpm / 60.0
        else:
            self._token_tokens = float("inf")
            self._token_rate = float("inf")

        self._last_update = time.monotonic()
        self._lock = threading.Lock()

        # stats for debugging
        self._total_wait_time = 0.0
        self._total_requests = 0

    def acquire(self, estimated_tokens: int = 1) -> float:
        """Try to acquire permission to make a request.

        Args:
            estimated_tokens: Estimated tokens for this request.

        Returns:
            Wait time in seconds (0 if no wait needed).
            Caller should sleep for this duration before proceeding.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # refill buckets based on time elapsed
            if self.rpm:
                self._request_tokens = min(
                    float(self.rpm), self._request_tokens + elapsed * self._request_rate
                )
            if self.tpm:
                self._token_tokens = min(
                    float(self.tpm), self._token_tokens + elapsed * self._token_rate
                )

            # calculate wait time if we don't have enough tokens
            wait_time = 0.0

            if self._request_tokens < 1:
                wait_for_request = (1 - self._request_tokens) / self._request_rate
                wait_time = max(wait_time, wait_for_request)

            if self._token_tokens < estimated_tokens:
                wait_for_tokens = (estimated_tokens - self._token_tokens) / self._token_rate
                wait_time = max(wait_time, wait_for_tokens)

            if wait_time > 0:
                self._total_wait_time += wait_time
                return wait_time

            # consume tokens
            self._request_tokens -= 1
            self._token_tokens -= estimated_tokens
            self._total_requests += 1

            return 0.0

    def wait_and_acquire(self, estimated_tokens: int = 1) -> None:
        """Block until request can be made.

        Convenience method that handles the sleep internally.

        Args:
            estimated_tokens: Estimated tokens for this request.
        """
        while True:
            wait_time = self.acquire(estimated_tokens)
            if wait_time <= 0:
                return
            logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
            time.sleep(wait_time)

    def report_actual_tokens(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Adjust token bucket after learning actual token count.

        If we over- or under-estimated, this corrects the bucket.
        Call this after getting the response with actual token counts.

        Args:
            actual_tokens: Actual tokens used.
            estimated_tokens: What we estimated when calling acquire().
        """
        diff = actual_tokens - estimated_tokens
        if diff == 0:
            return

        with self._lock:
            # if we used more than estimated, consume the difference
            # if we used less, give back the difference
            self._token_tokens -= diff

    @property
    def stats(self) -> dict:
        """Return statistics about rate limiting."""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_wait_time_s": self._total_wait_time,
                "avg_wait_time_s": (
                    self._total_wait_time / self._total_requests if self._total_requests > 0 else 0
                ),
                "current_request_tokens": self._request_tokens,
                "current_token_tokens": self._token_tokens,
            }

    def reset(self) -> None:
        """Reset the limiter to initial state.

        Useful for testing or starting a new batch.
        """
        with self._lock:
            if self.rpm:
                self._request_tokens = float(self.rpm)
            if self.tpm:
                self._token_tokens = float(self.tpm)
            self._last_update = time.monotonic()
            self._total_wait_time = 0.0
            self._total_requests = 0


class NoOpRateLimiter:
    """Rate limiter that does nothing.

    Use when rate limiting is disabled.
    """

    def acquire(self, estimated_tokens: int = 1) -> float:
        return 0.0

    def wait_and_acquire(self, estimated_tokens: int = 1) -> None:
        pass

    def report_actual_tokens(self, actual_tokens: int, estimated_tokens: int) -> None:
        pass

    @property
    def stats(self) -> dict:
        return {"enabled": False}

    def reset(self) -> None:
        pass
