"""EvalTask definition - the main input to an evaluation run."""

from dataclasses import dataclass, field
from typing import Any

from spark_llm_eval.core.config import (
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    OutputConfig,
    SamplingConfig,
    StatisticsConfig,
)


@dataclass
class EvalTask:
    """Defines a complete evaluation task.

    This is the main entry point for running an evaluation. It ties together
    the dataset, model, metrics, and configuration.

    Basic usage:
        task = EvalTask(
            task_id="my-eval",
            name="QA Evaluation",
            dataset_path="/mnt/delta/datasets/qa_test",
            model_config=ModelConfig(...),
            prompt_template="Answer: {{ question }}",
            metrics=[MetricConfig(name="exact_match", metric_type="lexical")],
        )
        result = runner.run(task)

    Args:
        task_id: Unique identifier for this evaluation. Used in tracking.
        name: Human-readable name for display.
        description: Optional longer description.
        dataset_path: Path to Delta table with evaluation data.
        dataset_version: Specific version to use (None = latest).
        input_column: Column containing the input text/question.
        reference_column: Column with ground truth answers. Can be None for
            metrics that don't need references (e.g., fluency).
        context_columns: Additional columns to include in prompt rendering.
        model_config: Configuration for the model being evaluated.
        prompt_template: Jinja2 template for constructing prompts. Column names
            become template variables.
        metrics: List of metrics to compute.
        statistics_config: Settings for CIs and significance tests.
        sampling_config: Optional - for evaluating a subset.
        stratify_by: Columns to break down results by (e.g., ["category"]).
        inference_config: Rate limiting, batching, caching settings.
        output_config: Where to save results.
        mlflow_experiment: MLflow experiment path for tracking.
        tags: Arbitrary key-value tags for filtering/searching.
        checkpoint_path: For resuming interrupted evaluations.
        parallelism: Number of Spark partitions. None = auto.
    """

    # required fields
    task_id: str
    name: str
    dataset_path: str
    model_config: ModelConfig
    prompt_template: str
    metrics: list[MetricConfig]

    # dataset config
    description: str | None = None
    dataset_version: int | None = None
    input_column: str = "input"
    reference_column: str | None = "reference"
    context_columns: list[str] = field(default_factory=list)

    # analysis config
    statistics_config: StatisticsConfig = field(default_factory=StatisticsConfig)
    sampling_config: SamplingConfig | None = None
    stratify_by: list[str] = field(default_factory=list)

    # execution config
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    parallelism: int | None = None
    checkpoint_path: str | None = None

    # tracking
    mlflow_experiment: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        # validate prompt template has the input column
        # just a basic check, Jinja will give better errors at render time
        if f"{{{{ {self.input_column}" not in self.prompt_template.replace(" ", ""):
            # this check is a bit fragile with whitespace but catches obvious issues
            pass  # TODO: maybe warn instead of silently passing

    def get_template_columns(self) -> list[str]:
        """Returns all columns needed for prompt rendering."""
        cols = [self.input_column]
        cols.extend(self.context_columns)
        return cols

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/storage.

        Note: This is a shallow conversion. ModelConfig etc. are converted
        to dicts but nested objects may still be dataclasses.
        """
        # TODO: proper recursive serialization
        from dataclasses import asdict

        return asdict(self)
