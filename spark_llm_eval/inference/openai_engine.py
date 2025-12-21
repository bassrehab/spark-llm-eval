"""OpenAI inference engine implementation."""

import logging
import time
from typing import Any

from openai import APIError, OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from spark_llm_eval.core.config import InferenceConfig, ModelConfig
from spark_llm_eval.core.exceptions import ConfigurationError, InferenceError, RateLimitError
from spark_llm_eval.inference.base import InferenceEngine, InferenceRequest, InferenceResponse
from spark_llm_eval.inference.rate_limiter import (
    NoOpRateLimiter,
    RateLimitConfig,
    TokenBucketRateLimiter,
)

logger = logging.getLogger(__name__)

# pricing per 1M tokens (as of late 2024, will need updates)
# TODO: make this configurable or fetch from API
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


class OpenAIInferenceEngine(InferenceEngine):
    """Inference engine for OpenAI API.

    Handles connection management, rate limiting, retries, and cost tracking.

    Example:
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
            api_key_secret="openai/api_key",
        )
        engine = OpenAIInferenceEngine(config, inference_config)
        engine.initialize()
        response = engine.infer(request)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig | None = None,
        api_key: str | None = None,  # for testing, normally use secrets
    ):
        self.model_config = model_config
        self.inference_config = inference_config or InferenceConfig()
        self._api_key = api_key
        self._client: OpenAI | None = None
        self._rate_limiter: TokenBucketRateLimiter | NoOpRateLimiter | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the OpenAI client and rate limiter."""
        if self._initialized:
            return

        # get API key
        api_key = self._api_key
        if not api_key and self.model_config.api_key_secret:
            api_key = self._get_secret(self.model_config.api_key_secret)

        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not provided. Set api_key_secret in ModelConfig "
                "or pass api_key directly."
            )

        # create client
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.model_config.endpoint:
            client_kwargs["base_url"] = self.model_config.endpoint

        self._client = OpenAI(**client_kwargs)

        # setup rate limiter
        if self.inference_config.rate_limit_rpm or self.inference_config.rate_limit_tpm:
            self._rate_limiter = TokenBucketRateLimiter(
                RateLimitConfig(
                    requests_per_minute=self.inference_config.rate_limit_rpm,
                    tokens_per_minute=self.inference_config.rate_limit_tpm,
                )
            )
        else:
            self._rate_limiter = NoOpRateLimiter()

        self._initialized = True
        logger.info(f"Initialized OpenAI engine with model {self.model_config.model_name}")

    def _get_secret(self, secret_path: str) -> str:
        """Get secret from Databricks or environment.

        In Databricks, use dbutils.secrets.get(). For local dev,
        fall back to environment variables.
        """
        # try Databricks secrets first
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark:
                # secret_path format: "scope/key"
                parts = secret_path.split("/", 1)
                if len(parts) == 2:
                    scope, key = parts
                    # this only works in Databricks runtime
                    dbutils = spark._jvm.com.databricks.service.DBUtils(spark._jsc)
                    return dbutils.secrets().get(scope, key)
        except Exception:
            pass  # not in Databricks or no access

        # fall back to environment variable
        import os

        env_key = secret_path.replace("/", "_").upper()
        value = os.environ.get(env_key)
        if value:
            return value

        # also try OPENAI_API_KEY directly
        return os.environ.get("OPENAI_API_KEY", "")

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a single request."""
        if not self._initialized:
            raise InferenceError("Engine not initialized. Call initialize() first.")

        estimated_tokens = self.estimate_tokens(request.prompt)
        self._rate_limiter.wait_and_acquire(estimated_tokens)

        start_time = time.perf_counter()
        try:
            response = self._call_api(request)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # update rate limiter with actual tokens
            actual_tokens = response.input_tokens + response.output_tokens
            self._rate_limiter.report_actual_tokens(actual_tokens, estimated_tokens)

            response.latency_ms = latency_ms
            return response

        except OpenAIRateLimitError as e:
            # extract retry-after if available
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    retry_after = float(retry_after)
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {e}",
                retry_after=retry_after,
            ) from e

        except APIError as e:
            raise InferenceError(
                f"OpenAI API error: {e}",
                details={"status_code": getattr(e, "status_code", None)},
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIError,)),
        reraise=True,
    )
    def _call_api(self, request: InferenceRequest) -> InferenceResponse:
        """Make the actual API call with retries."""
        messages = [{"role": "user", "content": request.prompt}]

        response = self._client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            **self.model_config.extra_params,
        )

        text = response.choices[0].message.content
        usage = response.usage

        cost = self._calculate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return InferenceResponse(
            request_id=request.request_id,
            text=text,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=0,  # filled in by caller
            cost_usd=cost,
            model=response.model,
            metadata=request.metadata,
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        model = self.model_config.model_name

        # find pricing - try exact match then prefix match
        pricing = OPENAI_PRICING.get(model)
        if not pricing:
            # try prefix match for versioned models like gpt-4o-2024-05-13
            for model_prefix, prices in OPENAI_PRICING.items():
                if model.startswith(model_prefix):
                    pricing = prices
                    break

        if not pricing:
            logger.warning(f"No pricing info for model {model}, using gpt-4o pricing")
            pricing = OPENAI_PRICING["gpt-4o"]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using tiktoken if available."""
        try:
            import tiktoken

            # gpt-4o uses cl100k_base
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # fallback to rough estimate
            return len(text) // 4 + 1

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._rate_limiter:
            stats = self._rate_limiter.stats
            logger.info(f"Rate limiter stats: {stats}")
        self._client = None
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return "openai"
