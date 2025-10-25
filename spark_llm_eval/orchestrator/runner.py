"""Main eval runner - ties together inference, metrics, and stats."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, FloatType
import logging
import time
from datetime import datetime

from spark_llm_eval.core.config import (
    ModelConfig,
    InferenceConfig,
    MetricConfig,
    StatisticsConfig,
    OutputConfig,
)
from spark_llm_eval.core.task import EvalTask
from spark_llm_eval.core.result import EvalResult, MetricValue, CostBreakdown, LatencyStats

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Config for eval runner."""
    model_config: ModelConfig
    metrics: List[MetricConfig]
    statistics_config: StatisticsConfig = field(
        default_factory=lambda: StatisticsConfig()
    )
    inference_config: InferenceConfig = field(
        default_factory=lambda: InferenceConfig()
    )
    output_config: OutputConfig = field(default_factory=lambda: OutputConfig())
    checkpoint_interval: int = 0
    cache_responses: bool = True
    response_cache_path: Optional[str] = None


class EvaluationRunner:
    """Main class that runs the eval pipeline end to end."""

    def __init__(self, spark: SparkSession, config: RunnerConfig, tracker=None):
        self.spark = spark
        self.config = config
        self.tracker = tracker  # mlflow tracker, optional
        self._start_time = None
        self._inference_engine = None

    def run(self, data: DataFrame, task: EvalTask) -> EvalResult:
        """Run the eval. Returns EvalResult with metrics and stats."""
        self._start_time = time.time()
        logger.info(f"Starting eval: {task.task_id}")

        self._validate_data(data, task)
        n_examples = data.count()

        # run inference
        data_with_predictions = self._run_inference(data, task)

        # compute metrics
        metrics = self._compute_metrics(data_with_predictions, task)

        # calculate statistics
        metrics_with_stats = self._compute_statistics(metrics, n_examples)

        # build result
        result = self._build_result(task, metrics_with_stats, n_examples)

        # track if configured
        if self.tracker:
            self._track_results(result)

        # save results if configured
        if self.config.output_config.save_results:
            self._save_results(data_with_predictions, result)

        elapsed = time.time() - self._start_time
        logger.info(f"Evaluation completed in {elapsed:.2f}s")

        return result

    def _validate_data(self, data, task):
        required_cols = set(task.get_template_columns())
        existing_cols = set(data.columns)

        missing = required_cols - existing_cols
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # check reference column if needed
        if any(m.requires_reference for m in self.config.metrics):
            ref_col = task.reference_column or "reference"
            if ref_col not in existing_cols:
                raise ValueError(f"Metrics require reference column '{ref_col}'")

    def _run_inference(self, data: DataFrame, task: EvalTask) -> DataFrame:
        """Run LLM inference on data.

        Returns DataFrame with prediction column added.
        """
        logger.info("Running inference...")

        # check if we have cached predictions
        if self.config.cache_responses and self.config.response_cache_path:
            cached = self._load_cached_responses()
            if cached is not None:
                logger.info("Using cached inference responses")
                return data.join(cached, on=task.id_column, how="left")

        # use the batch UDF for inference
        from spark_llm_eval.inference.batch_udf import (
            create_inference_udf,
            INFERENCE_OUTPUT_SCHEMA,
        )

        inference_udf = create_inference_udf(
            self.config.model_config,
            self.config.inference_config,
        )

        # Render prompts using template
        template_cols = task.get_template_columns()
        template_str = task.prompt_template
        col_name = template_cols[0]

        if len(template_cols) == 1:
            # Simple template: just use the input column as the prompt content
            if template_str and "{{" in template_str:
                # Use Spark SQL functions for simple template rendering
                # Replace {{ col_name }} with the column value
                # This avoids UDF serialization issues
                prompt_col = F.regexp_replace(
                    F.lit(template_str),
                    r"\{\{\s*" + col_name + r"\s*\}\}",
                    F.col(col_name)
                )
            else:
                prompt_col = F.col(col_name)
        else:
            # Multiple columns - need to do multiple replacements
            prompt_col = F.lit(template_str)
            for col in template_cols:
                prompt_col = F.regexp_replace(
                    prompt_col,
                    r"\{\{\s*" + col + r"\s*\}\}",
                    F.col(col)
                )

        # Add request_id and prompt columns for the UDF
        data_with_prompts = data.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn(
            "prompt", prompt_col
        )

        # Select only the columns needed for inference
        inference_input = data_with_prompts.select("request_id", "prompt")

        # Apply inference using mapInPandas
        inference_results = inference_input.mapInPandas(
            inference_udf, schema=INFERENCE_OUTPUT_SCHEMA
        )

        # Join results back to original data
        result_df = data_with_prompts.join(
            inference_results.select("request_id", "response_text"),
            on="request_id",
            how="left"
        ).withColumn(
            "prediction", F.col("response_text")
        ).drop("response_text", "request_id", "prompt")

        # cache responses if configured
        if self.config.cache_responses and self.config.response_cache_path:
            self._save_responses_cache(result_df, task.id_column)

        return result_df

    def _compute_metrics(
        self,
        data: DataFrame,
        task: EvalTask,
    ) -> Dict[str, List[float]]:
        """Compute metrics on predictions.

        Returns dict of metric name to list of per-example scores.
        """
        logger.info("Computing metrics...")

        from spark_llm_eval.evaluation import get_metric

        metrics_results = {}
        ref_col = task.reference_column or "reference"

        for metric_config in self.config.metrics:
            metric = get_metric(metric_config.name)

            # compute per-example scores
            # collect to driver for metrics computation
            # TODO: optimize with UDFs for large datasets
            predictions = data.select("prediction").rdd.flatMap(lambda x: x).collect()
            references = data.select(ref_col).rdd.flatMap(lambda x: x).collect()

            result = metric.compute(predictions, references, **metric_config.kwargs)

            if result.per_example_scores:
                metrics_results[metric_config.name] = result.per_example_scores
            else:
                # single aggregate value - replicate for stats
                metrics_results[metric_config.name] = [result.value]

        return metrics_results

    def _compute_statistics(
        self,
        metrics: Dict[str, List[float]],
        n_examples: int,
    ) -> Dict[str, MetricValue]:
        """Compute confidence intervals and statistics for metrics."""
        logger.info("Computing statistics...")

        from spark_llm_eval.statistics import bootstrap_ci, analytical_ci_proportion
        import numpy as np

        stats_config = self.config.statistics_config
        results = {}

        for name, scores in metrics.items():
            scores_arr = np.array(scores)
            mean_val = float(np.mean(scores_arr))

            # determine if binary metric
            is_binary = set(scores_arr.flatten()).issubset({0, 1, 0.0, 1.0, True, False})

            if is_binary and stats_config.ci_method == "analytical":
                successes = int(np.sum(scores_arr))
                _, ci, se = analytical_ci_proportion(
                    successes,
                    len(scores_arr),
                    stats_config.confidence_level,
                )
            else:
                _, ci, se = bootstrap_ci(
                    scores_arr,
                    confidence_level=stats_config.confidence_level,
                    n_iterations=stats_config.bootstrap_iterations,
                )

            results[name] = MetricValue(
                value=mean_val,
                confidence_interval=(ci[0], ci[1]),
                confidence_level=stats_config.confidence_level,
                standard_error=se,
                sample_size=len(scores_arr),
            )

        return results

    def _build_result(
        self,
        task: EvalTask,
        metrics: Dict[str, MetricValue],
        n_examples: int,
    ) -> EvalResult:
        """Build final EvalResult."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        elapsed_ms = elapsed * 1000

        return EvalResult(
            task_id=task.task_id,
            run_id=None,  # set by tracker if used
            timestamp=datetime.now(),
            metrics=metrics,
            stratified_metrics={},  # TODO: implement stratification
            cost=CostBreakdown(
                total_cost_usd=0.0,  # TODO: track actual costs
                input_tokens=0,
                output_tokens=0,
                num_requests=n_examples,
            ),
            latency=LatencyStats(
                mean_ms=elapsed_ms / n_examples if n_examples > 0 else 0,
                median_ms=elapsed_ms / n_examples if n_examples > 0 else 0,
                p95_ms=elapsed_ms / n_examples if n_examples > 0 else 0,
                p99_ms=elapsed_ms / n_examples if n_examples > 0 else 0,
                min_ms=0,
                max_ms=elapsed_ms,
                total_duration_s=elapsed,
            ),
            predictions_table=None,
            config_snapshot={
                "model": self.config.model_config.model_name,
                "provider": self.config.model_config.provider.value,
            },
            num_examples=n_examples,
            num_failures=0,
        )

    def _track_results(self, result: EvalResult) -> None:
        """Log results to MLflow tracker."""
        if not self.tracker:
            return

        # log metrics
        metrics_dict = {}
        for name, value in result.metrics.items():
            metrics_dict[name] = value.value
            if value.ci_lower is not None:
                metrics_dict[f"{name}_ci_lower"] = value.ci_lower
            if value.ci_upper is not None:
                metrics_dict[f"{name}_ci_upper"] = value.ci_upper

        self.tracker.log_metrics(metrics_dict)
        self.tracker.log_artifact(result.to_dict(), "result.json", "results")

    def _save_results(self, data: DataFrame, result: EvalResult) -> None:
        """Save results to configured output location."""
        output_path = self.config.output_config.results_path
        if not output_path:
            logger.warning("No output path configured, skipping save")
            return

        from spark_llm_eval.datasets import save_results
        save_results(data, output_path, mode="overwrite")

    def _load_cached_responses(self) -> Optional[DataFrame]:
        """Load cached inference responses if available."""
        if not self.config.response_cache_path:
            return None

        try:
            return self.spark.read.parquet(self.config.response_cache_path)
        except Exception as e:
            logger.debug(f"No cache found: {e}")
            return None

    def _save_responses_cache(self, data: DataFrame, id_column: str) -> None:
        """Save inference responses to cache."""
        if not self.config.response_cache_path:
            return

        cache_df = data.select(id_column, "prediction")
        cache_df.write.mode("overwrite").parquet(self.config.response_cache_path)
        logger.info(f"Cached responses to {self.config.response_cache_path}")


def run_evaluation(
    spark: SparkSession,
    data: DataFrame,
    task: EvalTask,
    model_config: ModelConfig,
    metrics: List[str],
    confidence_level: float = 0.95,
) -> EvalResult:
    """Convenience function for simple evaluations.

    Args:
        spark: Active SparkSession.
        data: Input DataFrame.
        task: Evaluation task.
        model_config: Model configuration.
        metrics: List of metric names.
        confidence_level: CI confidence level.

    Returns:
        EvalResult with metrics.

    Example:
        result = run_evaluation(
            spark,
            df,
            task,
            model_config,
            metrics=["exact_match", "f1"],
        )
    """
    metric_configs = [MetricConfig(name=m) for m in metrics]

    config = RunnerConfig(
        model_config=model_config,
        metrics=metric_configs,
        statistics_config=StatisticsConfig(confidence_level=confidence_level),
    )

    runner = EvaluationRunner(spark, config)
    return runner.run(data, task)
