"""Tests for custom exceptions."""

import pytest
from spark_llm_eval.core.exceptions import (
    SparkLLMEvalError,
    ConfigurationError,
    InferenceError,
    RateLimitError,
    MetricComputationError,
    DatasetError,
    CacheError,
)


class TestSparkLLMEvalError:
    """Tests for base exception."""

    def test_basic_message(self):
        err = SparkLLMEvalError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"

    def test_with_details(self):
        err = SparkLLMEvalError(
            "API call failed",
            details={"status_code": 500, "endpoint": "/v1/chat"}
        )
        s = str(err)
        assert "API call failed" in s
        assert "status_code" in s
        assert "500" in s

    def test_empty_details(self):
        err = SparkLLMEvalError("Error")
        assert err.details == {}


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_with_retry_after(self):
        err = RateLimitError(
            "Rate limit exceeded",
            retry_after=30.0,
            details={"limit": "10000 RPM"}
        )
        assert err.retry_after == 30.0
        assert "10000 RPM" in str(err)

    def test_without_retry_after(self):
        err = RateLimitError("Rate limited")
        assert err.retry_after is None


class TestExceptionHierarchy:
    """Test exception inheritance."""

    def test_inference_error_is_base(self):
        err = InferenceError("Inference failed")
        assert isinstance(err, SparkLLMEvalError)

    def test_rate_limit_is_inference(self):
        err = RateLimitError("Rate limited")
        assert isinstance(err, InferenceError)
        assert isinstance(err, SparkLLMEvalError)

    def test_config_error_is_base(self):
        err = ConfigurationError("Bad config")
        assert isinstance(err, SparkLLMEvalError)

    def test_metric_error_is_base(self):
        err = MetricComputationError("Metric failed")
        assert isinstance(err, SparkLLMEvalError)

    def test_dataset_error_is_base(self):
        err = DatasetError("Dataset not found")
        assert isinstance(err, SparkLLMEvalError)

    def test_cache_error_is_base(self):
        err = CacheError("Cache miss")
        assert isinstance(err, SparkLLMEvalError)


class TestCatchingExceptions:
    """Test that exceptions can be caught at appropriate levels."""

    def test_catch_all_with_base(self):
        """Can catch any spark-llm-eval error with base class."""
        errors = [
            ConfigurationError("bad config"),
            InferenceError("inference failed"),
            RateLimitError("rate limited"),
            MetricComputationError("metric failed"),
        ]
        for err in errors:
            try:
                raise err
            except SparkLLMEvalError as e:
                assert e is not None  # successfully caught

    def test_catch_rate_limit_specifically(self):
        """Can catch rate limit specifically while letting others through."""
        try:
            raise RateLimitError("rate limited", retry_after=5.0)
        except RateLimitError as e:
            assert e.retry_after == 5.0
