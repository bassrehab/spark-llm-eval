"""Main eval runner - ties together inference, metrics, and stats."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_llm_eval.core.config import (
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    OutputConfig,
    StatisticsConfig,
)
from spark_llm_eval.core.result import CostBreakdown, EvalResult, LatencyStats, MetricValue
from spark_llm_eval.core.task import EvalTask

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Config for eval runner."""

    model_config: ModelConfig
    metrics: list[MetricConfig]
    statistics_config: StatisticsConfig = field(default_factory=lambda: StatisticsConfig())
    inference_config: InferenceConfig = field(default_factory=lambda: InferenceConfig())
    output_config: OutputConfig = field(default_factory=lambda: OutputConfig())
    checkpoint_interval: int = 0
    cache_responses: bool = True
    response_cache_path: str | None = None


class EvaluationRunner:
    """Main class that runs the eval pipeline end to end."""

    def __init__(self, spark: SparkSession, config: RunnerConfig, tracker=None):
        self.spark = spark
        self.config = config
        self.tracker = tracker  # mlflow tracker, optional
        self._start_time = None
        self._inference_engine = None
        self._cache_stats = None  # CacheStatistics if caching enabled

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

        Supports both legacy Parquet caching (via response_cache_path) and
        new Delta-backed caching (via InferenceConfig.cache_config).
        """
        logger.info("Running inference...")

        # Check for new Delta-backed caching
        cache_config = self.config.inference_config.get_effective_cache_config()

        from spark_llm_eval.cache.config import CachePolicy

        # Legacy Parquet caching path (backward compatibility)
        if cache_config.policy == CachePolicy.DISABLED and self.config.response_cache_path:
            return self._run_inference_legacy(data, task)

        # Prepare prompts first (needed for both caching and inference)
        data_with_prompts = self._prepare_prompts(data, task)

        # Delta-backed caching
        if cache_config.policy != CachePolicy.DISABLED:
            return self._run_inference_with_cache(data_with_prompts, task, cache_config)

        # No caching - run inference directly
        return self._run_inference_direct(data_with_prompts, task)

    def _prepare_prompts(self, data: DataFrame, task: EvalTask) -> DataFrame:
        """Prepare prompts from template and add request_id."""
        template_cols = task.get_template_columns()
        template_str = task.prompt_template
        col_name = template_cols[0]

        if len(template_cols) == 1:
            if template_str and "{{" in template_str:
                prompt_col = F.regexp_replace(
                    F.lit(template_str), r"\{\{\s*" + col_name + r"\s*\}\}", F.col(col_name)
                )
            else:
                prompt_col = F.col(col_name)
        else:
            prompt_col = F.lit(template_str)
            for col in template_cols:
                prompt_col = F.regexp_replace(prompt_col, r"\{\{\s*" + col + r"\s*\}\}", F.col(col))

        return data.withColumn(
            "request_id", F.monotonically_increasing_id().cast("string")
        ).withColumn("prompt", prompt_col)

    def _run_inference_with_cache(
        self,
        data_with_prompts: DataFrame,
        task: EvalTask,
        cache_config,
    ) -> DataFrame:
        """Run inference with Delta-backed caching."""
        from spark_llm_eval.cache import CacheError, DeltaCacheManager
        from spark_llm_eval.cache.config import CachePolicy

        cache_manager = DeltaCacheManager(
            spark=self.spark,
            config=cache_config,
            model_config=self.config.model_config,
        )

        # Add cache keys
        data_with_keys = cache_manager.add_cache_keys(data_with_prompts)

        # Phase 1: Cache lookup
        if cache_config.policy in (
            CachePolicy.ENABLED,
            CachePolicy.READ_ONLY,
            CachePolicy.REPLAY,
        ):
            cached_df, uncached_df = cache_manager.lookup(data_with_keys)

            if cache_config.policy == CachePolicy.REPLAY:
                uncached_count = uncached_df.count()
                if uncached_count > 0:
                    raise CacheError(
                        f"Replay mode requires all cache hits, "
                        f"but {uncached_count} prompts were not cached"
                    )
                # All hits - return cached results
                self._cache_stats = cache_manager.get_statistics()
                return cached_df.drop("cache_key", "from_cache")
        else:
            # WRITE_ONLY - no lookup
            cached_df = None
            uncached_df = data_with_keys

        # Phase 2: Run inference on uncached data
        uncached_count = uncached_df.count()
        if uncached_count > 0:
            logger.info(f"Running inference on {uncached_count} uncached prompts")
            fresh_results = self._run_inference_direct(uncached_df, task, keep_prompt=True)

            # Phase 3: Store in cache
            if cache_config.policy in (CachePolicy.ENABLED, CachePolicy.WRITE_ONLY):
                cache_manager.store(
                    fresh_results.withColumn("response_text", F.col("prediction")),
                    task_id=task.task_id,
                )
        else:
            fresh_results = None
            logger.info("All responses served from cache!")

        # Phase 4: Combine cached and fresh results
        if cached_df is not None and fresh_results is not None:
            # Ensure matching columns
            cached_cols = set(cached_df.columns)
            fresh_cols = set(fresh_results.columns)

            # Add missing columns with nulls
            for col in cached_cols - fresh_cols:
                if col not in ("cache_key", "from_cache"):
                    fresh_results = fresh_results.withColumn(col, F.lit(None))
            for col in fresh_cols - cached_cols:
                if col not in ("cache_key", "from_cache"):
                    cached_df = cached_df.withColumn(col, F.lit(None))

            result_df = cached_df.unionByName(
                fresh_results,
                allowMissingColumns=True,
            )
        elif cached_df is not None:
            result_df = cached_df
        else:
            result_df = fresh_results

        # Clean up temporary columns
        result_df = result_df.drop("cache_key", "from_cache", "request_id", "prompt")

        self._cache_stats = cache_manager.get_statistics()
        return result_df

    def _run_inference_direct(
        self,
        data_with_prompts: DataFrame,
        task: EvalTask,
        keep_prompt: bool = False,
    ) -> DataFrame:
        """Run inference directly without caching."""
        from spark_llm_eval.inference.batch_udf import (
            INFERENCE_OUTPUT_SCHEMA,
            create_inference_udf,
        )

        inference_udf = create_inference_udf(
            self.config.model_config,
            self.config.inference_config,
        )

        # Select only columns needed for inference
        inference_input = data_with_prompts.select("request_id", "prompt")

        # Apply inference
        inference_results = inference_input.mapInPandas(
            inference_udf, schema=INFERENCE_OUTPUT_SCHEMA
        )

        # Join results back
        result_df = data_with_prompts.join(
            inference_results.select(
                "request_id",
                "response_text",
                "input_tokens",
                "output_tokens",
                "latency_ms",
                "cost_usd",
            ),
            on="request_id",
            how="left",
        ).withColumn("prediction", F.col("response_text"))

        if not keep_prompt:
            result_df = result_df.drop("response_text", "request_id", "prompt")
        else:
            result_df = result_df.drop("response_text", "request_id")

        return result_df

    def _run_inference_legacy(self, data: DataFrame, task: EvalTask) -> DataFrame:
        """Legacy inference path using Parquet caching."""
        # Check for cached predictions
        if self.config.cache_responses and self.config.response_cache_path:
            cached = self._load_cached_responses()
            if cached is not None:
                logger.info("Using cached inference responses (legacy Parquet)")
                return data.join(cached, on=task.id_column, how="left")

        # Prepare and run inference
        data_with_prompts = self._prepare_prompts(data, task)
        result_df = self._run_inference_direct(data_with_prompts, task)

        # Save to legacy cache
        if self.config.cache_responses and self.config.response_cache_path:
            self._save_responses_cache(result_df, task.id_column)

        return result_df

    def _compute_metrics(
        self,
        data: DataFrame,
        task: EvalTask,
    ) -> dict[str, list[float]]:
        """Compute metrics on predictions.

        Returns dict of metric name to list of per-example scores.
        """
        logger.info("Computing metrics...")

        from spark_llm_eval.evaluation import get_metric

        metrics_results = {}
        ref_col = task.reference_column or "reference"

        # Collect base columns
        predictions = data.select("prediction").rdd.flatMap(lambda x: x).collect()
        references = data.select(ref_col).rdd.flatMap(lambda x: x).collect()

        # Collect RAG-specific columns for metrics that need them
        extra_kwargs: dict[str, Any] = {}

        # Query column (use input_column)
        if task.input_column in data.columns:
            extra_kwargs["queries"] = (
                data.select(task.input_column).rdd.flatMap(lambda x: x).collect()
            )

        # Context columns
        if task.context_columns:
            context_cols = [c for c in task.context_columns if c in data.columns]
            if len(context_cols) == 1:
                extra_kwargs["contexts"] = (
                    data.select(context_cols[0]).rdd.flatMap(lambda x: x).collect()
                )
            elif context_cols:
                # Multiple context columns -> list per example
                rows = data.select(*context_cols).collect()
                extra_kwargs["contexts"] = [list(row.asDict().values()) for row in rows]

        for metric_config in self.config.metrics:
            # Pass kwargs to metric constructor
            metric = get_metric(metric_config.name, **metric_config.kwargs)

            # Merge extra kwargs with metric config kwargs
            compute_kwargs = {**extra_kwargs, **metric_config.kwargs}
            result = metric.compute(predictions, references, **compute_kwargs)

            if result.per_example_scores:
                metrics_results[metric_config.name] = result.per_example_scores
            else:
                # single aggregate value - replicate for stats
                metrics_results[metric_config.name] = [result.value]

        return metrics_results

    def _compute_statistics(
        self,
        metrics: dict[str, list[float]],
        n_examples: int,
    ) -> dict[str, MetricValue]:
        """Compute confidence intervals and statistics for metrics."""
        logger.info("Computing statistics...")

        import numpy as np

        from spark_llm_eval.statistics import analytical_ci_proportion, bootstrap_ci

        stats_config = self.config.statistics_config
        results = {}

        for name, scores in metrics.items():
            scores_arr = np.array(scores)
            mean_val = float(np.mean(scores_arr))

            # determine if binary metric
            is_binary = set(scores_arr.flatten()).issubset({0, 1})

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
        metrics: dict[str, MetricValue],
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

    def _load_cached_responses(self) -> DataFrame | None:
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
    metrics: list[str],
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
