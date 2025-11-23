"""Integration tests for Spark UDFs with distributed inference.

These tests verify that inference works correctly in a distributed Spark environment.
Run with: pytest tests/integration/test_spark_udfs.py -v
"""

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


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

        # Create UDF
        inference_udf = create_inference_udf(
            model_config=openai_config,
            max_tokens=50,
            temperature=0.0,
        )

        # Apply inference
        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

        # Collect results
        results = result_df.collect()

        assert len(results) == 10
        for row in results:
            assert row.response is not None
            assert len(row.response) > 0

    @pytest.mark.google
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_distributed_gemini_inference(self, spark_session, sample_qa_df, google_config):
        """Test Gemini inference distributed across Spark partitions."""
        from spark_llm_eval.inference import create_inference_udf

        # Repartition to ensure work is distributed
        df = sample_qa_df.repartition(2)

        # Create UDF
        inference_udf = create_inference_udf(
            model_config=google_config,
            max_tokens=50,
            temperature=0.0,
        )

        # Apply inference
        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

        # Collect results
        results = result_df.collect()

        assert len(results) == 10
        for row in results:
            assert row.response is not None
            assert len(row.response) > 0


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

        inference_udf = create_inference_udf(
            model_config=openai_config,
            max_tokens=20,
            temperature=0.0,
        )

        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

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

        inference_udf = create_inference_udf(
            model_config=openai_config,
            max_tokens=20,
            temperature=0.0,
        )

        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

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

        inference_udf = create_inference_udf(
            model_config=openai_config,
            max_tokens=20,
            temperature=0.0,
        )

        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

        results = result_df.collect()
        assert len(results) == 2


class TestPromptTemplates:
    """Test prompt template rendering in distributed setting."""

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_prompt_template(self, spark_session, openai_config):
        """Test that prompt templates are rendered correctly."""
        from spark_llm_eval.inference import create_inference_udf

        df = spark_session.createDataFrame([
            {"question": "France", "context": "capital cities"},
            {"question": "Germany", "context": "capital cities"},
        ])

        inference_udf = create_inference_udf(
            model_config=openai_config,
            max_tokens=20,
            temperature=0.0,
            prompt_template="Answer about {{ context }}: What is the capital of {{ input }}? Answer in one word.",
        )

        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

        results = result_df.collect()
        assert len(results) == 2
        # Should contain city names
        assert any("Paris" in r.response for r in results)
        assert any("Berlin" in r.response for r in results)


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

        # Create UDF with strict rate limiting
        inference_udf = create_inference_udf(
            model_config=openai_config,
            max_tokens=20,
            temperature=0.0,
            rate_limit_rpm=30,  # Low rate limit
        )

        start_time = time.time()

        result_df = df.withColumn(
            "response",
            inference_udf(F.col("question"))
        )

        results = result_df.collect()
        elapsed = time.time() - start_time

        # All requests should complete without rate limit errors
        assert len(results) == 10
        for row in results:
            assert row.response is not None

        # With 10 requests at 30 RPM, should take at least some time
        # (though this depends on actual timing)
        print(f"Elapsed time: {elapsed:.2f}s")
