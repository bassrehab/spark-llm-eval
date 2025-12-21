"""Anthropic/Claude inference."""

import logging
import os
import time

from spark_llm_eval.core.config import ModelConfig
from spark_llm_eval.core.exceptions import InferenceError, RateLimitError
from spark_llm_eval.inference.base import InferenceEngine, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

# lazy load the sdk
_anthropic_module = None


def _load_anthropic():
    global _anthropic_module
    if _anthropic_module is None:
        try:
            import anthropic

            _anthropic_module = anthropic
        except ImportError:
            raise ImportError("need anthropic: pip install anthropic")
    return _anthropic_module


class AnthropicEngine(InferenceEngine):
    """Claude models via Anthropic API."""

    # per 1M tokens, these change often so may be outdated
    PRICING = {
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-2.1": {"input": 8.00, "output": 24.00},
        "claude-2.0": {"input": 8.00, "output": 24.00},
        "claude-instant": {"input": 0.80, "output": 2.40},
    }

    def __init__(self, config: ModelConfig, api_key: str | None = None):
        self.config = config
        self._api_key = api_key  # for testing, normally use api_key_secret
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def initialize(self):
        anthropic = _load_anthropic()
        api_key = self._get_api_key()
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic engine with model: {self.config.model_name}")

    def _get_api_key(self):
        # explicit key takes priority (mostly for tests)
        if self._api_key:
            return self._api_key

        if self.config.api_key_secret:
            # could be env var name or databricks secret path
            key = os.environ.get(self.config.api_key_secret)
            if key:
                return key

            # try databricks secrets - this is a bit hacky
            try:
                from pyspark.sql import SparkSession

                spark = SparkSession.getActiveSession()
                if spark and "/" in self.config.api_key_secret:
                    scope, key_name = self.config.api_key_secret.split("/", 1)
                    return spark._jvm.com.databricks.service.DBUtils.secrets().get(scope, key_name)
            except:
                pass  # not on databricks, no big deal

        # last resort
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key

        raise ValueError(
            "No API key found. Set ANTHROPIC_API_KEY environment variable "
            "or provide api_key in config."
        )

    def _get_pricing(self) -> tuple[float, float]:
        """Get pricing for current model (per 1M tokens)."""
        model = self.config.model_name.lower()

        for key, prices in self.PRICING.items():
            if key in model:
                return prices["input"], prices["output"]

        # default to sonnet pricing
        return 3.00, 15.00

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference for a single request.

        Args:
            request: The inference request.

        Returns:
            InferenceResponse with generated text.

        Raises:
            InferenceError: If inference fails.
            RateLimitError: If rate limited.
        """
        if self._client is None:
            raise InferenceError("Engine not initialized. Call initialize() first.")

        anthropic = _load_anthropic()

        start_time = time.time()

        try:
            message = self._client.messages.create(
                model=self.config.model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}],
                stop_sequences=request.stop_sequences or [],
            )

            # extract text from response
            text = ""
            for block in message.content:
                if hasattr(block, "text"):
                    text += block.text

            latency_ms = (time.time() - start_time) * 1000

            # track tokens
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # calculate cost
            input_price, output_price = self._get_pricing()
            cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

            return InferenceResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                model=self.config.model_name,
                finish_reason=message.stop_reason,
            )

        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}") from e
        except anthropic.APIError as e:
            raise InferenceError(f"Anthropic API error: {e}") from e
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

    def infer_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResponse]:
        """Run inference for a batch of requests.

        Anthropic doesn't have a batch API, so we process sequentially.

        Args:
            requests: List of inference requests.

        Returns:
            List of responses.
        """
        responses = []
        for request in requests:
            try:
                response = self.infer(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch inference failed for request: {e}")
                # return error response
                responses.append(
                    InferenceResponse(
                        text="",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0,
                        cost_usd=0,
                        model=self.config.model_name,
                        finish_reason="error",
                        error=str(e),
                    )
                )
        return responses

    def shutdown(self) -> None:
        """Clean up resources."""
        self._client = None

    @property
    def total_tokens(self) -> tuple[int, int]:
        """Get total tokens used (input, output)."""
        return self._total_input_tokens, self._total_output_tokens

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        input_price, output_price = self._get_pricing()
        return (
            self._total_input_tokens * input_price + self._total_output_tokens * output_price
        ) / 1_000_000

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"
