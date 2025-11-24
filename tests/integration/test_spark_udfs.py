"""Integration tests for Spark UDFs with distributed inference.

These tests verify that inference works correctly in a distributed Spark environment.
Run with: pytest tests/integration/test_spark_udfs.py -v
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

from spark_llm_eval.core.config import ModelConfig, ModelProvider, InferenceConfig

# Output schema for inference results (matches batch_udf.py)
INFERENCE_OUTPUT_SCHEMA = StructType([
    StructField("request_id", StringType(), False),
    StructField("response_text", StringType(), True),
    StructField("input_tokens", IntegerType(), True),
    StructField("output_tokens", IntegerType(), True),
    StructField("latency_ms", FloatType(), True),
    StructField("cost_usd", FloatType(), True),
    StructField("error", StringType(), True),
])


class TestSparkDistributedInference:
    """Test inference running distributed across Spark workers."""

    @pytest.mark.openai
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_distributed_openai_inference(self, spark_session, sample_qa_df, openai_config):
        """Test OpenAI inference distributed across Spark partitions."""
        from spark_llm_eval.inference import create_inference_udf

        # Repartition to ensure work is distributed
        df = sample_qa_df.repartition(2)

        # Override max_tokens for this test
        model_config = ModelConfig(
            provider=openai_config.provider,
            model_name=openai_config.model_name,
            api_key_secret=openai_config.api_key_secret,
            max_tokens=50,
            temperature=0.0,
        )
        inference_config = InferenceConfig(batch_size=5)

        # Create UDF
        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        # Prepare input - UDF expects request_id and prompt columns
        df_input = df.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", F.col("question")
        ).select("request_id", "prompt")

        # Apply inference using mapInPandas
        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        # Collect results
        results = result_df.collect()

        assert len(results) == 10
        for row in results:
            assert row.response_text is not None
            assert len(row.response_text) > 0

    @pytest.mark.google
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_distributed_gemini_inference(self, spark_session, sample_qa_df, google_config):
        """Test Gemini inference distributed across Spark partitions."""
        from spark_llm_eval.inference import create_inference_udf

        # Repartition to ensure work is distributed
        df = sample_qa_df.repartition(2)

        # Override max_tokens for this test
        model_config = ModelConfig(
            provider=google_config.provider,
            model_name=google_config.model_name,
            api_key_secret=google_config.api_key_secret,
            max_tokens=50,
            temperature=0.0,
        )
        inference_config = InferenceConfig(batch_size=5)

        # Create UDF
        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        # Prepare input - UDF expects request_id and prompt columns
        df_input = df.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", F.col("question")
        ).select("request_id", "prompt")

        # Apply inference using mapInPandas
        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        # Collect results
        results = result_df.collect()

        assert len(results) == 10
        for row in results:
            assert row.response_text is not None
            assert len(row.response_text) > 0


class TestSparkPartitioning:
    """Test behavior with different partitioning strategies."""

    @pytest.mark.openai
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_single_partition(self, spark_session, sample_qa_df, openai_config):
        """Test inference with a single partition."""
        from spark_llm_eval.inference import create_inference_udf

        df = sample_qa_df.coalesce(1)
        assert df.rdd.getNumPartitions() == 1

        model_config = ModelConfig(
            provider=openai_config.provider,
            model_name=openai_config.model_name,
            api_key_secret=openai_config.api_key_secret,
            max_tokens=20,
            temperature=0.0,
        )
        inference_config = InferenceConfig(batch_size=10)

        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        # Prepare input
        df_input = df.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", F.col("question")
        ).select("request_id", "prompt")

        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        results = result_df.collect()
        assert len(results) == 10

    @pytest.mark.openai
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_many_partitions(self, spark_session, sample_qa_df, openai_config):
        """Test inference with many partitions (one per row)."""
        from spark_llm_eval.inference import create_inference_udf

        df = sample_qa_df.repartition(10)
        assert df.rdd.getNumPartitions() == 10

        model_config = ModelConfig(
            provider=openai_config.provider,
            model_name=openai_config.model_name,
            api_key_secret=openai_config.api_key_secret,
            max_tokens=20,
            temperature=0.0,
        )
        inference_config = InferenceConfig(batch_size=1)

        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        # Prepare input
        df_input = df.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", F.col("question")
        ).select("request_id", "prompt")

        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        results = result_df.collect()
        assert len(results) == 10


class TestErrorHandling:
    """Test error handling in distributed inference."""

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_handles_empty_input(self, spark_session, openai_config):
        """Test that empty input is handled gracefully."""
        from spark_llm_eval.inference import create_inference_udf

        df = spark_session.createDataFrame([
            {"question": ""},
            {"question": "What is 2+2?"},
        ])

        model_config = ModelConfig(
            provider=openai_config.provider,
            model_name=openai_config.model_name,
            api_key_secret=openai_config.api_key_secret,
            max_tokens=20,
            temperature=0.0,
        )
        inference_config = InferenceConfig(batch_size=2)

        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        # Prepare input
        df_input = df.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", F.col("question")
        ).select("request_id", "prompt")

        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        results = result_df.collect()
        assert len(results) == 2


class TestPromptTemplates:
    """Test prompt template rendering in distributed setting."""

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_prompt_template(self, spark_session, openai_config):
        """Test that prompt templates are rendered correctly via pre-processing."""
        from spark_llm_eval.inference import create_inference_udf
        from pyspark.sql.functions import concat, lit

        df = spark_session.createDataFrame([
            {"question": "France", "context": "capital cities"},
            {"question": "Germany", "context": "capital cities"},
        ])

        model_config = ModelConfig(
            provider=openai_config.provider,
            model_name=openai_config.model_name,
            api_key_secret=openai_config.api_key_secret,
            max_tokens=20,
            temperature=0.0,
        )
        inference_config = InferenceConfig(batch_size=2)

        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        # Build prompt using Spark SQL functions (template rendering via DataFrame)
        df_input = df.withColumn(
            "prompt",
            concat(
                lit("Answer about "),
                F.col("context"),
                lit(": What is the capital of "),
                F.col("question"),
                lit("? Answer in one word.")
            )
        ).withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).select("request_id", "prompt")

        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        results = result_df.collect()
        assert len(results) == 2
        # Should contain city names
        responses = [r.response_text.lower() for r in results if r.response_text]
        assert any("paris" in r for r in responses)
        assert any("berlin" in r for r in responses)


class TestRateLimiting:
    """Test rate limiting behavior in distributed setting."""

    @pytest.mark.openai
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_rate_limiting_respected(self, spark_session, sample_qa_df, openai_config):
        """Test that rate limiting prevents API errors."""
        from spark_llm_eval.inference import create_inference_udf
        import time

        df = sample_qa_df.repartition(2)

        model_config = ModelConfig(
            provider=openai_config.provider,
            model_name=openai_config.model_name,
            api_key_secret=openai_config.api_key_secret,
            max_tokens=20,
            temperature=0.0,
        )
        # Create config with strict rate limiting
        inference_config = InferenceConfig(
            batch_size=5,
            rate_limit_rpm=30,  # Low rate limit
        )

        inference_udf = create_inference_udf(
            model_config=model_config,
            inference_config=inference_config,
        )

        start_time = time.time()

        # Prepare input
        df_input = df.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", F.col("question")
        ).select("request_id", "prompt")

        result_df = df_input.mapInPandas(inference_udf, schema=INFERENCE_OUTPUT_SCHEMA)

        results = result_df.collect()
        elapsed = time.time() - start_time

        # All requests should complete without rate limit errors
        assert len(results) == 10
        for row in results:
            assert row.response_text is not None

        # With 10 requests at 30 RPM, should take at least some time
        # (though this depends on actual timing)
        print(f"Elapsed time: {elapsed:.2f}s")
