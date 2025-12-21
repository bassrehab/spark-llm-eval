"""Gemini inference via Google's generativeai SDK."""

import logging
import os
import time

from spark_llm_eval.core.config import ModelConfig
from spark_llm_eval.core.exceptions import InferenceError, RateLimitError
from spark_llm_eval.inference.base import InferenceEngine, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


def _import_genai():
    """Import google.generativeai on first use."""
    try:
        import google.generativeai as genai

        return genai
    except ImportError:
        raise ImportError("missing google-generativeai, run: pip install google-generativeai")


class GeminiEngine(InferenceEngine):
    """Google Gemini models."""

    # rough pricing, check google's site for current rates
    PRICING = {
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, config: ModelConfig, api_key=None):
        self.config = config
        self._api_key = api_key
        self._model = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def initialize(self) -> None:
        """Initialize the Gemini client."""
        genai = _import_genai()

        api_key = self._get_api_key()
        genai.configure(api_key=api_key)

        # create model instance
        self._model = genai.GenerativeModel(self.config.model_name)

        logger.info(f"Initialized Gemini engine with model: {self.config.model_name}")

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""

        # check explicit api_key first (for testing)
        if self._api_key:
            return self._api_key

        if self.config.api_key_secret:
            # try environment variable
            key = os.environ.get(self.config.api_key_secret)
            if key:
                return key

            # try Databricks secrets if available
            try:
                from pyspark.sql import SparkSession

                spark = SparkSession.getActiveSession()
                if spark:
                    parts = self.config.api_key_secret.split("/")
                    if len(parts) == 2:
                        scope, key_name = parts
                        return spark._jvm.com.databricks.service.DBUtils.secrets().get(
                            scope, key_name
                        )
            except Exception:
                pass

        # fallback to environment
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            return key

        raise ValueError(
            "No API key found. Set GOOGLE_API_KEY environment variable "
            "or provide api_key in config."
        )

    def _get_pricing(self) -> tuple[float, float]:
        """Get pricing for current model (per 1M tokens)."""
        model = self.config.model_name.lower()

        for key, prices in self.PRICING.items():
            if key in model:
                return prices["input"], prices["output"]

        # default to gemini-1.5-pro pricing
        return 3.50, 10.50

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
        if self._model is None:
            raise InferenceError("Engine not initialized. Call initialize() first.")

        genai = _import_genai()

        start_time = time.time()

        try:
            # build generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_sequences=request.stop_sequences or [],
            )

            # generate response
            response = self._model.generate_content(
                request.prompt,
                generation_config=generation_config,
            )

            # extract text from response
            text = ""
            if response.parts:
                text = response.text

            latency_ms = (time.time() - start_time) * 1000

            # get token counts from usage metadata
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            # track tokens
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # calculate cost
            input_price, output_price = self._get_pricing()
            cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

            # determine finish reason
            finish_reason = "stop"
            if response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason.name).lower()

            return InferenceResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                model=self.config.model_name,
                finish_reason=finish_reason,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg and "limit" in error_msg:
                raise RateLimitError(f"Gemini rate limit exceeded: {e}") from e
            elif "quota" in error_msg:
                raise RateLimitError(f"Gemini quota exceeded: {e}") from e
            elif "api" in error_msg or "invalid" in error_msg:
                raise InferenceError(f"Gemini API error: {e}") from e
            else:
                raise InferenceError(f"Inference failed: {e}") from e

    def infer_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResponse]:
        """Run inference for a batch of requests.

        Gemini doesn't have a batch API, so we process sequentially.

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
        self._model = None

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
        return "google"
