"""Tests for EvalTask."""

import pytest
from spark_llm_eval.core.task import EvalTask
from spark_llm_eval.core.config import (
    ModelProvider,
    ModelConfig,
    MetricConfig,
    SamplingConfig,
)


class TestEvalTask:
    """Tests for EvalTask."""

    @pytest.fixture
    def basic_model_config(self):
        return ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
        )

    @pytest.fixture
    def basic_metrics(self):
        return [
            MetricConfig(name="exact_match", metric_type="lexical"),
            MetricConfig(name="f1", metric_type="lexical"),
        ]

    def test_create_basic_task(self, basic_model_config, basic_metrics):
        task = EvalTask(
            task_id="test-001",
            name="Basic Test",
            dataset_path="/delta/test_data",
            model_config=basic_model_config,
            prompt_template="Question: {{ input }}\nAnswer:",
            metrics=basic_metrics,
        )
        assert task.task_id == "test-001"
        assert task.name == "Basic Test"
        assert len(task.metrics) == 2

    def test_create_task_with_all_options(self, basic_model_config, basic_metrics):
        task = EvalTask(
            task_id="full-test",
            name="Full Options Test",
            description="Testing all the options",
            dataset_path="/delta/test_data",
            dataset_version=5,
            input_column="question",
            reference_column="answer",
            context_columns=["context", "metadata"],
            model_config=basic_model_config,
            prompt_template="Context: {{ context }}\nQ: {{ question }}\nA:",
            metrics=basic_metrics,
            sampling_config=SamplingConfig(sample_size=500),
            stratify_by=["category"],
            mlflow_experiment="/Shared/tests",
            tags={"env": "dev", "version": "0.1"},
        )
        assert task.dataset_version == 5
        assert task.input_column == "question"
        assert len(task.context_columns) == 2
        assert task.tags["env"] == "dev"

    def test_empty_task_id_raises(self, basic_model_config, basic_metrics):
        with pytest.raises(ValueError, match="task_id"):
            EvalTask(
                task_id="",
                name="Test",
                dataset_path="/delta/test",
                model_config=basic_model_config,
                prompt_template="{{ input }}",
                metrics=basic_metrics,
            )

    def test_empty_metrics_raises(self, basic_model_config):
        with pytest.raises(ValueError, match="metric"):
            EvalTask(
                task_id="test",
                name="Test",
                dataset_path="/delta/test",
                model_config=basic_model_config,
                prompt_template="{{ input }}",
                metrics=[],
            )

    def test_get_template_columns(self, basic_model_config, basic_metrics):
        task = EvalTask(
            task_id="test",
            name="Test",
            dataset_path="/delta/test",
            input_column="question",
            context_columns=["context", "doc_id"],
            model_config=basic_model_config,
            prompt_template="{{ context }}\n{{ question }}",
            metrics=basic_metrics,
        )
        cols = task.get_template_columns()
        assert "question" in cols
        assert "context" in cols
        assert "doc_id" in cols

    def test_to_dict(self, basic_model_config, basic_metrics):
        task = EvalTask(
            task_id="test",
            name="Test",
            dataset_path="/delta/test",
            model_config=basic_model_config,
            prompt_template="{{ input }}",
            metrics=basic_metrics,
        )
        d = task.to_dict()
        assert d["task_id"] == "test"
        assert d["name"] == "Test"
        assert "model_config" in d

    def test_default_values(self, basic_model_config, basic_metrics):
        task = EvalTask(
            task_id="test",
            name="Test",
            dataset_path="/delta/test",
            model_config=basic_model_config,
            prompt_template="{{ input }}",
            metrics=basic_metrics,
        )
        # check defaults are set correctly
        assert task.input_column == "input"
        assert task.reference_column == "reference"
        assert task.context_columns == []
        assert task.stratify_by == []
        assert task.parallelism is None
