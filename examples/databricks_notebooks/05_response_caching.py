# Databricks notebook source
# MAGIC %md
# MAGIC # Response Caching with spark-llm-eval
# MAGIC
# MAGIC This notebook demonstrates Delta Lake-backed response caching to reduce LLM API costs.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - Configure Delta-backed response caching
# MAGIC - Run evaluations with cache warm-up (first run)
# MAGIC - Use cached responses for subsequent runs (cache hits)
# MAGIC - Use replay mode for metrics-only iteration
# MAGIC - View cache statistics and cost savings
# MAGIC - Manage cache (invalidation, vacuum)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks Runtime 14.0+ with ML
# MAGIC - API key for OpenAI stored in Databricks secrets

# COMMAND ----------

# MAGIC %pip install spark-llm-eval
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from spark_llm_eval.core.config import (
    ModelConfig,
    ModelProvider,
    MetricConfig,
    StatisticsConfig,
    InferenceConfig,
)
from spark_llm_eval.core.task import EvalTask
from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig
from spark_llm_eval.cache import CacheConfig, CachePolicy

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Model and Cache

# COMMAND ----------

# Model configuration
model_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini",
    api_key_secret="openai-api-key",  # Or use dbutils.secrets.get("scope", "key")
    temperature=0.0,
    max_tokens=512,
)

# Cache configuration - Delta table for response caching
CACHE_TABLE_PATH = "/tmp/spark_llm_eval/response_cache"

cache_config = CacheConfig(
    policy=CachePolicy.ENABLED,
    table_path=CACHE_TABLE_PATH,
    ttl_hours=24,  # Cache expires after 24 hours
    cache_version="1.0",
    track_statistics=True,
)

# Inference configuration with caching
inference_config = InferenceConfig(
    batch_size=32,
    max_retries=3,
    cache_config=cache_config,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sample Evaluation Dataset

# COMMAND ----------

# Create sample QA dataset
qa_data = [
    {"id": "q1", "question": "What is the capital of France?", "answer": "Paris"},
    {"id": "q2", "question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"id": "q3", "question": "What is the speed of light?", "answer": "299,792,458 meters per second"},
    {"id": "q4", "question": "What is photosynthesis?", "answer": "The process by which plants convert sunlight into energy"},
    {"id": "q5", "question": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
    {"id": "q6", "question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"id": "q7", "question": "What causes the seasons?", "answer": "The tilt of Earth's axis"},
    {"id": "q8", "question": "What is the Mona Lisa?", "answer": "A painting by Leonardo da Vinci"},
]

df = spark.createDataFrame(qa_data)
display(df)

# COMMAND ----------

# Save to Delta Lake
eval_table_path = "/tmp/spark_llm_eval/caching_demo_data"
df.write.format("delta").mode("overwrite").save(eval_table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. First Run - Cache Warm-Up
# MAGIC
# MAGIC The first run will call the LLM API for all examples and store responses in the cache.

# COMMAND ----------

# Define evaluation task
task = EvalTask(
    task_id="cache-demo",
    name="Caching Demo Evaluation",
    dataset_path=eval_table_path,
    model_config=model_config,
    prompt_template="Answer the following question concisely:\n\nQuestion: {{ question }}\n\nAnswer:",
    input_column="question",
    reference_column="answer",
    id_column="id",
    metrics=[
        MetricConfig(name="exact_match"),
        MetricConfig(name="f1"),
    ],
)

# Configure runner with caching
runner_config = RunnerConfig(
    model_config=model_config,
    metrics=task.metrics,
    inference_config=inference_config,
    statistics_config=StatisticsConfig(confidence_level=0.95),
)

runner = EvaluationRunner(spark, runner_config)

# COMMAND ----------

# Run evaluation (first run - all cache misses)
print("Running evaluation (first run - populating cache)...")
print("=" * 60)

eval_df = spark.read.format("delta").load(eval_table_path)
result1 = runner.run(eval_df, task)

# COMMAND ----------

# View results
print("\nFirst Run Results:")
print("=" * 60)
for metric_name, metric_value in result1.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}: {metric_value.value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# COMMAND ----------

# View cache statistics after first run
if runner._cache_stats:
    stats = runner._cache_stats.to_dict()
    print("\nCache Statistics (First Run):")
    print("-" * 40)
    print(f"Lookups: {stats['cache_lookup_count']}")
    print(f"Hits: {stats['cache_hit_count']}")
    print(f"Misses: {stats['cache_miss_count']}")
    print(f"Hit Rate: {stats['cache_hit_rate_pct']:.1f}%")
    print(f"Stored: {stats['cache_store_count']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Second Run - Using Cache
# MAGIC
# MAGIC The second run will use cached responses - no API calls needed!

# COMMAND ----------

# Run evaluation again (second run - all cache hits)
print("Running evaluation (second run - using cache)...")
print("=" * 60)

# Create new runner to reset stats
runner2 = EvaluationRunner(spark, runner_config)
result2 = runner2.run(eval_df, task)

# COMMAND ----------

# View cache statistics after second run
if runner2._cache_stats:
    stats = runner2._cache_stats.to_dict()
    print("\nCache Statistics (Second Run):")
    print("-" * 40)
    print(f"Lookups: {stats['cache_lookup_count']}")
    print(f"Hits: {stats['cache_hit_count']}")
    print(f"Misses: {stats['cache_miss_count']}")
    print(f"Hit Rate: {stats['cache_hit_rate_pct']:.1f}%")
    print(f"Stored: {stats['cache_store_count']}")
    print(f"\nCost Saved: ${stats['estimated_cost_saved_usd']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Replay Mode - Metrics-Only Iteration
# MAGIC
# MAGIC Replay mode requires all prompts to be cached. If any are missing, it raises an error.
# MAGIC This is useful when iterating on metrics without incurring API costs.

# COMMAND ----------

# Configure replay mode
replay_cache_config = CacheConfig(
    policy=CachePolicy.REPLAY,
    table_path=CACHE_TABLE_PATH,
    cache_version="1.0",
)

replay_inference_config = InferenceConfig(
    batch_size=32,
    cache_config=replay_cache_config,
)

# Different metrics for replay (no API calls)
replay_task = EvalTask(
    task_id="cache-demo-replay",
    name="Replay Mode Demo",
    dataset_path=eval_table_path,
    model_config=model_config,
    prompt_template="Answer the following question concisely:\n\nQuestion: {{ question }}\n\nAnswer:",
    input_column="question",
    reference_column="answer",
    id_column="id",
    metrics=[
        MetricConfig(name="exact_match"),
        MetricConfig(name="f1"),
        MetricConfig(name="contains"),  # Additional metric
        MetricConfig(name="length_ratio"),  # Additional metric
    ],
)

replay_runner_config = RunnerConfig(
    model_config=model_config,
    metrics=replay_task.metrics,
    inference_config=replay_inference_config,
    statistics_config=StatisticsConfig(confidence_level=0.95),
)

replay_runner = EvaluationRunner(spark, replay_runner_config)

# COMMAND ----------

# Run in replay mode (all from cache)
print("Running in replay mode (metrics-only, no API calls)...")
print("=" * 60)

result_replay = replay_runner.run(eval_df, replay_task)

# COMMAND ----------

# View replay results with additional metrics
print("\nReplay Mode Results (4 metrics, 0 API calls):")
print("=" * 60)
for metric_name, metric_value in result_replay.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}: {metric_value.value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. View Cache Contents

# COMMAND ----------

# Read cache table directly
cache_df = spark.read.format("delta").load(CACHE_TABLE_PATH)

print(f"Total cached responses: {cache_df.count()}")
display(cache_df.select(
    "cache_key",
    "model_name",
    "provider",
    "response_text",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "created_at",
    "access_count",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cache Management

# COMMAND ----------

# MAGIC %md
# MAGIC ### Invalidate Old Cache Entries

# COMMAND ----------

from spark_llm_eval.cache import DeltaCacheManager
from datetime import datetime, timedelta

# Create cache manager for management operations
mgmt_cache_config = CacheConfig(
    policy=CachePolicy.ENABLED,
    table_path=CACHE_TABLE_PATH,
)

cache_manager = DeltaCacheManager(
    spark=spark,
    config=mgmt_cache_config,
    model_config=model_config,
)

# View cache info
cache_info = cache_manager.get_cache_info()
print("Cache Info:")
print("-" * 40)
print(f"Table Path: {cache_info['table_path']}")
print(f"Total Entries: {cache_info['total_entries']}")
print(f"Cache Version: {cache_info['cache_version']}")
print(f"TTL Hours: {cache_info['ttl_hours']}")
print("\nEntries by Model:")
for entry in cache_info.get('entries_by_model', []):
    print(f"  {entry['provider']}/{entry['model']}: {entry['count']}")

# COMMAND ----------

# Example: Invalidate entries older than 7 days
# (Uncomment to run)
#
# cutoff = datetime.utcnow() - timedelta(days=7)
# invalidated = cache_manager.invalidate(before_timestamp=cutoff)
# print(f"Invalidated {invalidated} old entries")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vacuum Cache Table

# COMMAND ----------

# Clean up old Delta files (default 7-day retention)
# (Uncomment to run)
#
# cache_manager.vacuum(retention_hours=168)  # 7 days

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Cache Policies Summary
# MAGIC
# MAGIC | Policy | Lookup | Store | Use Case |
# MAGIC |--------|--------|-------|----------|
# MAGIC | `ENABLED` | Yes | Yes | Normal operation - save costs on repeated prompts |
# MAGIC | `DISABLED` | No | No | Fresh responses every time |
# MAGIC | `READ_ONLY` | Yes | No | Use cache but don't add new entries |
# MAGIC | `WRITE_ONLY` | No | Yes | Warm up cache without using it |
# MAGIC | `REPLAY` | Yes (required) | No | Metrics iteration without API calls |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cost Comparison

# COMMAND ----------

print("Cost Comparison:")
print("=" * 60)
print("\nWithout Caching:")
print(f"  8 examples x $0.00015/call = ${8 * 0.00015:.5f} per evaluation")
print(f"  10 evaluations = ${8 * 0.00015 * 10:.4f}")
print()
print("With Caching (after warm-up):")
print(f"  8 examples cached = $0.00 per evaluation")
print(f"  10 evaluations = ${8 * 0.00015:.5f} (just warm-up cost)")
print()
print(f"Savings: ${8 * 0.00015 * 9:.4f} (90% reduction)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC
# MAGIC 1. **Cache Configuration**: Using `CacheConfig` with `CachePolicy` to control caching behavior
# MAGIC 2. **First Run (Warm-up)**: All cache misses, responses stored in Delta table
# MAGIC 3. **Second Run (Cached)**: All cache hits, no API calls needed
# MAGIC 4. **Replay Mode**: Iterate on metrics without incurring API costs
# MAGIC 5. **Cache Management**: View, invalidate, and vacuum cache entries
# MAGIC
# MAGIC **Key Benefits:**
# MAGIC - **Cost Reduction**: Avoid redundant API calls for identical prompts
# MAGIC - **Faster Iteration**: Replay mode enables rapid metrics experimentation
# MAGIC - **ACID Guarantees**: Delta Lake ensures consistent, reliable caching
# MAGIC - **TTL Support**: Automatic expiration of stale cached responses
