"""Spark Pandas UDF wrappers for distributed inference.

This module provides the bridge between Spark DataFrames and inference engines.
Inference happens inside Pandas UDFs, which allows efficient batch processing
with Arrow serialization.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict

import pandas as pd
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from spark_llm_eval.core.config import InferenceConfig, ModelConfig, ModelProvider
from spark_llm_eval.inference.base import InferenceEngine, InferenceRequest

logger = logging.getLogger(__name__)

# executor-local cache for inference engines
# key = config hash, value = engine instance
_ENGINE_CACHE: dict[str, InferenceEngine] = {}


def _get_engine_cache_key(model_config: dict, inference_config: dict) -> str:
    """Generate cache key from config dicts."""
    # sort keys for deterministic hashing
    combined = {
        "model": model_config,
        "inference": inference_config,
    }
    return json.dumps(combined, sort_keys=True)


def _create_engine(model_config: dict, inference_config: dict) -> InferenceEngine:
    """Factory function to create inference engine from config dicts."""
    provider = ModelProvider(model_config["provider"])

    # reconstruct config objects
    mc = ModelConfig(
        provider=provider,
        model_name=model_config["model_name"],
        temperature=model_config.get("temperature", 0.0),
        max_tokens=model_config.get("max_tokens", 1024),
        api_key_secret=model_config.get("api_key_secret"),
        endpoint=model_config.get("endpoint"),
        extra_params=model_config.get("extra_params", {}),
    )

    ic = InferenceConfig(
        batch_size=inference_config.get("batch_size", 32),
        max_retries=inference_config.get("max_retries", 3),
        retry_delay=inference_config.get("retry_delay", 1.0),
        timeout=inference_config.get("timeout", 60.0),
        rate_limit_rpm=inference_config.get("rate_limit_rpm"),
        rate_limit_tpm=inference_config.get("rate_limit_tpm"),
        enable_caching=inference_config.get("enable_caching", True),
        cache_table=inference_config.get("cache_table"),
    )

    # import the right engine based on provider
    if provider == ModelProvider.OPENAI:
        from spark_llm_eval.inference.openai_engine import OpenAIInferenceEngine

        engine = OpenAIInferenceEngine(mc, ic)
    elif provider == ModelProvider.ANTHROPIC:
        from spark_llm_eval.inference.anthropic_engine import AnthropicEngine

        engine = AnthropicEngine(mc, ic)
    elif provider == ModelProvider.GOOGLE:
        from spark_llm_eval.inference.gemini_engine import GeminiEngine

        engine = GeminiEngine(mc, ic)
    elif provider == ModelProvider.DATABRICKS:
        # TODO: implement DatabricksInferenceEngine
        raise NotImplementedError(f"Provider {provider} not yet implemented")
    else:
        raise NotImplementedError(f"Provider {provider} not yet implemented")

    engine.initialize()
    return engine


def _get_or_create_engine(model_config: dict, inference_config: dict) -> InferenceEngine:
    """Get cached engine or create new one.

    Engines are cached per executor to avoid repeated initialization.
    This is safe because each executor processes partitions sequentially.
    """
    cache_key = _get_engine_cache_key(model_config, inference_config)

    if cache_key not in _ENGINE_CACHE:
        logger.info("Creating new inference engine for this executor")
        _ENGINE_CACHE[cache_key] = _create_engine(model_config, inference_config)

    return _ENGINE_CACHE[cache_key]


# output schema for inference results
INFERENCE_OUTPUT_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("response_text", StringType(), True),
        StructField("input_tokens", IntegerType(), True),
        StructField("output_tokens", IntegerType(), True),
        StructField("latency_ms", FloatType(), True),
        StructField("cost_usd", FloatType(), True),
        StructField("error", StringType(), True),
    ]
)


def create_inference_udf(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
):
    """Create a Pandas UDF for batch inference.

    The returned UDF expects a DataFrame with columns:
        - request_id: unique identifier
        - prompt: the rendered prompt text

    And returns:
        - request_id: matches input
        - response_text: generated text (null on error)
        - input_tokens: prompt token count
        - output_tokens: response token count
        - latency_ms: request latency
        - cost_usd: estimated cost
        - error: error message if failed (null on success)

    Args:
        model_config: Model configuration.
        inference_config: Inference configuration.

    Returns:
        A Pandas UDF function that can be used with DataFrame.mapInPandas().
    """
    # serialize configs for broadcast to executors
    model_config_dict = _serialize_model_config(model_config)
    inference_config_dict = asdict(inference_config)

    # convert to JSON strings for closure
    model_json = json.dumps(model_config_dict)
    inference_json = json.dumps(inference_config_dict)

    def inference_batch(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Process batches of prompts through the inference engine."""
        # deserialize configs
        mc = json.loads(model_json)
        ic = json.loads(inference_json)

        # get or create engine (cached per executor)
        engine = _get_or_create_engine(mc, ic)

        for pdf in iterator:
            # build requests
            requests = [
                InferenceRequest(
                    prompt=row["prompt"],
                    request_id=row["request_id"],
                )
                for _, row in pdf.iterrows()
            ]

            # run inference
            responses = engine.infer_batch(requests)

            # build output dataframe
            results = []
            for resp in responses:
                results.append(
                    {
                        "request_id": resp.request_id,
                        "response_text": resp.text,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "latency_ms": resp.latency_ms,
                        "cost_usd": resp.cost_usd,
                        "error": resp.error,
                    }
                )

            yield pd.DataFrame(results)

    return inference_batch


def _serialize_model_config(config: ModelConfig) -> dict:
    """Serialize ModelConfig to dict, handling the enum."""
    d = asdict(config)
    # enum needs to be converted to string value
    d["provider"] = config.provider.value
    return d


def create_inference_map_udf(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
):
    """Alternative UDF using mapInPandas for simpler usage.

    This version is easier to use but may be slightly less efficient
    than the grouped map version for very large batches.

    Usage:
        udf = create_inference_map_udf(model_config, inference_config)
        result_df = df.mapInPandas(udf, schema=INFERENCE_OUTPUT_SCHEMA)
    """
    return create_inference_udf(model_config, inference_config)


# for backwards compatibility
def cleanup_engines():
    """Clean up cached engines.

    Call this at the end of your job to properly shut down connections.
    In practice, Spark will kill executors anyway, but this is cleaner.
    """
    global _ENGINE_CACHE
    for engine in _ENGINE_CACHE.values():
        try:
            engine.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down engine: {e}")
    _ENGINE_CACHE.clear()
