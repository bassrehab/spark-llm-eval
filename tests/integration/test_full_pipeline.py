"""End-to-end integration tests for the full evaluation pipeline.

These tests run the complete evaluation workflow from data loading to results.
Run with: pytest tests/integration/test_full_pipeline.py -v
"""

import pytest
import os


class TestFullEvaluationPipeline:
    """End-to-end tests for the evaluation pipeline."""

    @pytest.mark.openai
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_full_qa_evaluation_openai(self, spark_session, sample_qa_df, openai_config, temp_delta_path):
        """Test complete QA evaluation with OpenAI."""
        from spark_llm_eval.core.config import MetricConfig, StatisticsConfig, OutputConfig
        from spark_llm_eval.core.task import EvalTask
        from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig

        # Setup paths
        input_path = os.path.join(temp_delta_path, "qa_input")
        output_path = os.path.join(temp_delta_path, "qa_output")

        # Save input data
        sample_qa_df.write.format("delta").mode("overwrite").save(input_path)

        # Define task
        task = EvalTask(
            task_id="integration-test-001",
            name="QA Integration Test",
            dataset_path=input_path,
            model_config=openai_config,
            prompt_template="""Answer the following question concisely in a few words.

Question: {{ question }}

Answer:""",
            input_column="question",
            reference_column="answer",
            metrics=[
                MetricConfig(name="exact_match"),
                MetricConfig(name="f1"),
                MetricConfig(name="contains"),
            ],
        )

        # Configure runner
        runner_config = RunnerConfig(
            model_config=openai_config,
            metrics=task.metrics,
            statistics_config=StatisticsConfig(
                confidence_level=0.95,
                bootstrap_iterations=100,
            ),
            output_config=OutputConfig(results_path=output_path, save_results=True),
        )

        # Run evaluation
        runner = EvaluationRunner(spark_session, runner_config)
        result = runner.run(
            spark_session.read.format("delta").load(input_path),
            task,
        )

        # Verify results
        assert result is not None
        assert "exact_match" in result.metrics
        assert "f1" in result.metrics
        assert "contains" in result.metrics

        # Check confidence intervals exist
        for metric_name, metric_result in result.metrics.items():
            assert metric_result.value >= 0
            assert metric_result.value <= 1
            if metric_result.confidence_interval:
                ci_lower, ci_upper = metric_result.confidence_interval
                assert ci_lower <= metric_result.value <= ci_upper

    @pytest.mark.anthropic
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_full_qa_evaluation_anthropic(self, spark_session, sample_qa_df, anthropic_config, temp_delta_path):
        """Test complete QA evaluation with Anthropic Claude."""
        from spark_llm_eval.core.config import MetricConfig, StatisticsConfig, OutputConfig
        from spark_llm_eval.core.task import EvalTask
        from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig

        # Setup paths
        input_path = os.path.join(temp_delta_path, "qa_input_anthropic")
        output_path = os.path.join(temp_delta_path, "qa_output_anthropic")

        # Save input data (use subset for speed)
        sample_qa_df.limit(5).write.format("delta").mode("overwrite").save(input_path)

        # Define task
        task = EvalTask(
            task_id="integration-test-anthropic",
            name="QA Integration Test Anthropic",
            dataset_path=input_path,
            model_config=anthropic_config,
            prompt_template="""Answer the following question concisely in a few words.

Question: {{ question }}

Answer:""",
            input_column="question",
            reference_column="answer",
            metrics=[
                MetricConfig(name="exact_match"),
                MetricConfig(name="f1"),
            ],
        )

        # Configure runner
        runner_config = RunnerConfig(
            model_config=anthropic_config,
            metrics=task.metrics,
            statistics_config=StatisticsConfig(
                confidence_level=0.95,
                bootstrap_iterations=100,
            ),
            output_config=OutputConfig(results_path=output_path, save_results=True),
        )

        # Run evaluation
        runner = EvaluationRunner(spark_session, runner_config)
        result = runner.run(
            spark_session.read.format("delta").load(input_path),
            task,
        )

        # Verify results
        assert result is not None
        assert "exact_match" in result.metrics
        assert "f1" in result.metrics

    @pytest.mark.google
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_full_qa_evaluation_gemini(self, spark_session, sample_qa_df, google_config, temp_delta_path):
        """Test complete QA evaluation with Google Gemini."""
        from spark_llm_eval.core.config import MetricConfig, StatisticsConfig, OutputConfig
        from spark_llm_eval.core.task import EvalTask
        from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig

        # Setup paths
        input_path = os.path.join(temp_delta_path, "qa_input_gemini")
        output_path = os.path.join(temp_delta_path, "qa_output_gemini")

        # Save input data (use subset for speed)
        sample_qa_df.limit(5).write.format("delta").mode("overwrite").save(input_path)

        # Define task
        task = EvalTask(
            task_id="integration-test-gemini",
            name="QA Integration Test Gemini",
            dataset_path=input_path,
            model_config=google_config,
            prompt_template="""Answer the following question concisely in a few words.

Question: {{ question }}

Answer:""",
            input_column="question",
            reference_column="answer",
            metrics=[
                MetricConfig(name="exact_match"),
                MetricConfig(name="f1"),
            ],
        )

        # Configure runner
        runner_config = RunnerConfig(
            model_config=google_config,
            metrics=task.metrics,
            statistics_config=StatisticsConfig(
                confidence_level=0.95,
                bootstrap_iterations=100,
            ),
            output_config=OutputConfig(results_path=output_path, save_results=True),
        )

        # Run evaluation
        runner = EvaluationRunner(spark_session, runner_config)
        result = runner.run(
            spark_session.read.format("delta").load(input_path),
            task,
        )

        # Verify results
        assert result is not None
        assert "exact_match" in result.metrics


class TestMetricsComputation:
    """Test various metrics in the pipeline."""

    @pytest.mark.openai
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_all_lexical_metrics(self, spark_session, sample_qa_df, openai_config, temp_delta_path):
        """Test all lexical metrics."""
        from spark_llm_eval.core.config import MetricConfig, StatisticsConfig
        from spark_llm_eval.core.task import EvalTask
        from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig

        input_path = os.path.join(temp_delta_path, "lexical_input")
        sample_qa_df.limit(5).write.format("delta").mode("overwrite").save(input_path)

        task = EvalTask(
            task_id="lexical-test",
            name="Lexical Metrics Test",
            dataset_path=input_path,
            model_config=openai_config,
            prompt_template="Answer: {{ question }}",
            input_column="question",
            reference_column="answer",
            metrics=[
                MetricConfig(name="exact_match"),
                MetricConfig(name="f1"),
                MetricConfig(name="bleu"),
                MetricConfig(name="rouge_l"),
                MetricConfig(name="contains"),
                MetricConfig(name="length_ratio"),
            ],
        )

        runner_config = RunnerConfig(
            model_config=openai_config,
            metrics=task.metrics,
            statistics_config=StatisticsConfig(bootstrap_iterations=100),
        )

        runner = EvaluationRunner(spark_session, runner_config)
        result = runner.run(
            spark_session.read.format("delta").load(input_path),
            task,
        )

        # Verify all metrics computed
        assert len(result.metrics) == 6
        for metric_name in ["exact_match", "f1", "bleu", "rouge_l", "contains", "length_ratio"]:
            assert metric_name in result.metrics
            assert result.metrics[metric_name].value >= 0


class TestModelComparison:
    """Test comparing multiple models."""

    @pytest.mark.openai
    @pytest.mark.anthropic
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_compare_openai_vs_anthropic(self, spark_session, sample_qa_df, openai_config, anthropic_config, temp_delta_path):
        """Test comparing OpenAI and Anthropic on same dataset."""
        from spark_llm_eval.core.config import MetricConfig, StatisticsConfig
        from spark_llm_eval.core.task import EvalTask
        from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig

        input_path = os.path.join(temp_delta_path, "comparison_input")
        sample_qa_df.limit(5).write.format("delta").mode("overwrite").save(input_path)

        prompt_template = """Answer the question concisely.
Question: {{ question }}
Answer:"""

        metrics = [MetricConfig(name="f1")]
        stats_config = StatisticsConfig(bootstrap_iterations=100)

        results = {}

        for name, config in [("openai", openai_config), ("anthropic", anthropic_config)]:
            task = EvalTask(
                task_id=f"comparison-{name}",
                name=f"Comparison {name}",
                dataset_path=input_path,
                model_config=config,
                prompt_template=prompt_template,
                input_column="question",
                reference_column="answer",
                metrics=metrics,
            )

            runner_config = RunnerConfig(
                model_config=config,
                metrics=metrics,
                statistics_config=stats_config,
            )

            runner = EvaluationRunner(spark_session, runner_config)
            results[name] = runner.run(
                spark_session.read.format("delta").load(input_path),
                task,
            )

        # Both should have results
        assert "openai" in results
        assert "anthropic" in results

        # Compare F1 scores
        openai_f1 = results["openai"].metrics["f1"]
        anthropic_f1 = results["anthropic"].metrics["f1"]

        print(f"OpenAI F1: {openai_f1.value:.4f}")
        print(f"Anthropic F1: {anthropic_f1.value:.4f}")

        # Both should have valid metric values
        assert openai_f1.value >= 0
        assert anthropic_f1.value >= 0


class TestResultsSerialization:
    """Test that results can be serialized and persisted."""

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_results_to_dataframe(self, spark_session, sample_qa_df, openai_config, temp_delta_path):
        """Test converting results to DataFrame for storage."""
        from spark_llm_eval.core.config import MetricConfig, StatisticsConfig, OutputConfig
        from spark_llm_eval.core.task import EvalTask
        from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig

        input_path = os.path.join(temp_delta_path, "serialize_input")
        output_path = os.path.join(temp_delta_path, "serialize_output")

        sample_qa_df.limit(3).write.format("delta").mode("overwrite").save(input_path)

        task = EvalTask(
            task_id="serialize-test",
            name="Serialization Test",
            dataset_path=input_path,
            model_config=openai_config,
            prompt_template="Answer: {{ question }}",
            input_column="question",
            reference_column="answer",
            metrics=[MetricConfig(name="f1")],
        )

        runner_config = RunnerConfig(
            model_config=openai_config,
            metrics=task.metrics,
            statistics_config=StatisticsConfig(bootstrap_iterations=100),
            output_config=OutputConfig(results_path=output_path, save_results=True),
        )

        runner = EvaluationRunner(spark_session, runner_config)
        result = runner.run(
            spark_session.read.format("delta").load(input_path),
            task,
        )

        # Convert result to dict for storage
        result_dict = {
            "task_id": task.task_id,
            "f1_value": result.metrics["f1"].value,
            "f1_ci_lower": result.metrics["f1"].confidence_interval[0] if result.metrics["f1"].confidence_interval else None,
            "f1_ci_upper": result.metrics["f1"].confidence_interval[1] if result.metrics["f1"].confidence_interval else None,
        }

        # Create DataFrame and save
        result_df = spark_session.createDataFrame([result_dict])
        result_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(output_path)

        # Read back
        loaded_df = spark_session.read.format("delta").load(output_path)
        assert loaded_df.count() == 1
        row = loaded_df.collect()[0]
        assert row.task_id == "serialize-test"
        assert row.f1_value >= 0
