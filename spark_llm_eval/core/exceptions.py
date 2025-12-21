"""Custom exceptions."""


class SparkLLMEvalError(Exception):
    """Base exception - catch this to get all spark-llm-eval errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(SparkLLMEvalError):
    """Bad config - missing api key, invalid model name, etc."""

    pass


class InferenceError(SparkLLMEvalError):
    """LLM API call failed. See RateLimitError for rate limits specifically."""

    pass


class RateLimitError(InferenceError):
    """Hit the rate limit. Check retry_after for when to retry."""

    def __init__(self, message: str, retry_after: float | None = None, details: dict | None = None):
        super().__init__(message, details)
        self.retry_after = retry_after


class MetricComputationError(SparkLLMEvalError):
    """Metric computation blew up."""

    pass


class DatasetError(SparkLLMEvalError):
    """Something wrong with the dataset - missing table, bad schema, etc."""

    pass


class CacheError(SparkLLMEvalError):
    """Cache operation failed. Usually non-fatal, eval can continue without it."""

    pass


# TODO: TrackingError for mlflow stuff
