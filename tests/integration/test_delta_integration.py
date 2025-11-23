"""Integration tests for Delta Lake operations.

These tests verify Delta Lake read/write functionality.
Run with: pytest tests/integration/test_delta_integration.py -v
"""

import pytest
import os


class TestDeltaWrite:
    """Test writing evaluation results to Delta Lake."""

    @pytest.mark.delta
    def test_write_results_to_delta(self, spark_session, sample_qa_df, temp_delta_path):
        """Test writing DataFrame to Delta table."""
        delta_path = os.path.join(temp_delta_path, "test_results")

        # Add some evaluation columns
        df = sample_qa_df.withColumn("prediction", sample_qa_df.answer)
        df = df.withColumn("score", df.answer.isNotNull().cast("double"))

        # Write to Delta
        df.write.format("delta").mode("overwrite").save(delta_path)

        # Verify we can read it back
        result_df = spark_session.read.format("delta").load(delta_path)
        assert result_df.count() == 10

    @pytest.mark.delta
    def test_append_to_delta(self, spark_session, sample_qa_df, temp_delta_path):
        """Test appending to existing Delta table."""
        delta_path = os.path.join(temp_delta_path, "append_test")

        # Initial write
        sample_qa_df.write.format("delta").mode("overwrite").save(delta_path)
        initial_count = spark_session.read.format("delta").load(delta_path).count()

        # Append same data
        sample_qa_df.write.format("delta").mode("append").save(delta_path)
        final_count = spark_session.read.format("delta").load(delta_path).count()

        assert final_count == initial_count * 2

    @pytest.mark.delta
    def test_delta_schema_evolution(self, spark_session, sample_qa_df, temp_delta_path):
        """Test adding new columns to Delta table."""
        delta_path = os.path.join(temp_delta_path, "schema_evolution")

        # Initial write
        sample_qa_df.write.format("delta").mode("overwrite").save(delta_path)

        # Add new column and append
        from pyspark.sql import functions as F
        df_with_new_col = sample_qa_df.withColumn("new_metric", F.lit(0.5))

        df_with_new_col.write.format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .save(delta_path)

        # Verify schema includes new column
        result_df = spark_session.read.format("delta").load(delta_path)
        assert "new_metric" in result_df.columns


class TestDeltaRead:
    """Test reading evaluation data from Delta Lake."""

    @pytest.mark.delta
    def test_read_delta_table(self, spark_session, sample_qa_df, temp_delta_path):
        """Test reading from Delta table."""
        delta_path = os.path.join(temp_delta_path, "read_test")

        # Write test data
        sample_qa_df.write.format("delta").mode("overwrite").save(delta_path)

        # Read back
        result_df = spark_session.read.format("delta").load(delta_path)

        assert result_df.count() == 10
        assert set(result_df.columns) == {"question", "answer"}

    @pytest.mark.delta
    def test_delta_time_travel(self, spark_session, sample_qa_df, temp_delta_path):
        """Test Delta time travel functionality."""
        delta_path = os.path.join(temp_delta_path, "time_travel")

        # Version 0: initial write
        sample_qa_df.write.format("delta").mode("overwrite").save(delta_path)

        # Version 1: add more data
        sample_qa_df.write.format("delta").mode("append").save(delta_path)

        # Read current version
        current_df = spark_session.read.format("delta").load(delta_path)
        assert current_df.count() == 20

        # Read version 0
        version0_df = spark_session.read.format("delta") \
            .option("versionAsOf", 0) \
            .load(delta_path)
        assert version0_df.count() == 10

    @pytest.mark.delta
    def test_delta_partition_filtering(self, spark_session, temp_delta_path):
        """Test reading with partition filtering."""
        delta_path = os.path.join(temp_delta_path, "partitioned")

        # Create partitioned data
        data = [
            {"model": "gpt-4", "question": "Q1", "score": 0.9},
            {"model": "gpt-4", "question": "Q2", "score": 0.8},
            {"model": "claude", "question": "Q1", "score": 0.85},
            {"model": "claude", "question": "Q2", "score": 0.95},
        ]
        df = spark_session.createDataFrame(data)

        # Write partitioned by model
        df.write.format("delta") \
            .partitionBy("model") \
            .mode("overwrite") \
            .save(delta_path)

        # Read only gpt-4 partition
        gpt4_df = spark_session.read.format("delta") \
            .load(delta_path) \
            .filter("model = 'gpt-4'")

        assert gpt4_df.count() == 2


class TestDatasetLoader:
    """Test the dataset loader utility."""

    @pytest.mark.delta
    def test_load_dataset_from_delta(self, spark_session, sample_qa_df, temp_delta_path):
        """Test loading dataset using the datasets module."""
        from spark_llm_eval.datasets import load_dataset

        delta_path = os.path.join(temp_delta_path, "loader_test")

        # Write test data
        sample_qa_df.write.format("delta").mode("overwrite").save(delta_path)

        # Load using our utility
        dataset = load_dataset(
            spark_session,
            table_path=delta_path,
            input_column="question",
            reference_column="answer",
        )

        assert dataset.count() == 10
        assert "question" in dataset.columns
        assert "answer" in dataset.columns

    @pytest.mark.delta
    def test_load_dataset_with_limit(self, spark_session, sample_qa_df, temp_delta_path):
        """Test loading dataset with limit."""
        from spark_llm_eval.datasets import load_dataset

        delta_path = os.path.join(temp_delta_path, "limit_test")

        # Write test data
        sample_qa_df.write.format("delta").mode("overwrite").save(delta_path)

        # Load dataset and apply limit manually
        dataset = load_dataset(
            spark_session,
            table_path=delta_path,
            input_column="question",
            reference_column="answer",
        )

        # Verify full dataset loads, then test sampling manually
        assert dataset.count() == 10
        sampled = dataset.sample(fraction=0.5, seed=42)
        assert sampled.count() < 10


class TestResultsPersistence:
    """Test persisting evaluation results."""

    @pytest.mark.delta
    def test_persist_metric_results(self, spark_session, temp_delta_path):
        """Test persisting metric results to Delta."""
        from datetime import datetime

        delta_path = os.path.join(temp_delta_path, "metrics")

        # Create sample metric results
        results = [
            {
                "run_id": "run-001",
                "metric_name": "exact_match",
                "value": 0.85,
                "ci_lower": 0.80,
                "ci_upper": 0.90,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "run_id": "run-001",
                "metric_name": "f1",
                "value": 0.88,
                "ci_lower": 0.83,
                "ci_upper": 0.93,
                "timestamp": datetime.now().isoformat(),
            },
        ]
        df = spark_session.createDataFrame(results)

        # Write to Delta
        df.write.format("delta").mode("overwrite").save(delta_path)

        # Read back and verify
        result_df = spark_session.read.format("delta").load(delta_path)
        assert result_df.count() == 2

        # Query specific metric
        exact_match = result_df.filter("metric_name = 'exact_match'").collect()[0]
        assert exact_match.value == 0.85
