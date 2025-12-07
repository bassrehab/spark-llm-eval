# Databricks notebook source
# MAGIC %md
# MAGIC # Model Comparison with Statistical Analysis
# MAGIC
# MAGIC This notebook demonstrates how to compare multiple LLM providers and determine
# MAGIC if differences are statistically significant.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks Runtime 14.0+ with ML
# MAGIC - API keys for OpenAI and Anthropic stored in Databricks secrets

# COMMAND ----------

# MAGIC %pip install spark-llm-eval
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

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
from spark_llm_eval.statistics import paired_ttest, bootstrap_test, cohens_d

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Multiple Models

# COMMAND ----------

# OpenAI Configuration
openai_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini",
    api_key_secret="openai-api-key",
    temperature=0.0,
    max_tokens=256,
)

# Anthropic Configuration
anthropic_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307",
    api_key_secret="anthropic-api-key",
    temperature=0.0,
    max_tokens=256,
)

models = {
    "gpt-4o-mini": openai_config,
    "claude-3-haiku": anthropic_config,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare Evaluation Dataset

# COMMAND ----------

# Larger dataset for statistical power
eval_data = [
    {"question": "What is the capital of France?", "answer": "Paris", "category": "geography"},
    {"question": "What is 2 + 2?", "answer": "4", "category": "math"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare", "category": "literature"},
    {"question": "What is the largest planet?", "answer": "Jupiter", "category": "science"},
    {"question": "What year did WWII end?", "answer": "1945", "category": "history"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "category": "science"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci", "category": "art"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo", "category": "geography"},
    {"question": "What is 10 x 10?", "answer": "100", "category": "math"},
    {"question": "Who wrote 1984?", "answer": "George Orwell", "category": "literature"},
    {"question": "What is the smallest planet?", "answer": "Mercury", "category": "science"},
    {"question": "What year did WWI start?", "answer": "1914", "category": "history"},
    {"question": "What is the chemical symbol for silver?", "answer": "Ag", "category": "science"},
    {"question": "Who sculpted David?", "answer": "Michelangelo", "category": "art"},
    {"question": "What is the capital of Germany?", "answer": "Berlin", "category": "geography"},
    {"question": "What is 15 / 3?", "answer": "5", "category": "math"},
    {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen", "category": "literature"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars", "category": "science"},
    {"question": "What year did the Berlin Wall fall?", "answer": "1989", "category": "history"},
    {"question": "What is the chemical symbol for iron?", "answer": "Fe", "category": "science"},
]

df = spark.createDataFrame(eval_data)
eval_table_path = "/tmp/spark_llm_eval/comparison_eval_data"
df.write.format("delta").mode("overwrite").save(eval_table_path)

print(f"Dataset size: {df.count()} examples")
display(df.groupBy("category").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run Evaluation for Each Model

# COMMAND ----------

results = {}

prompt_template = """Answer the following question concisely in a few words.

Question: {{ question }}

Answer:"""

metrics = [
    MetricConfig(name="exact_match"),
    MetricConfig(name="f1"),
    MetricConfig(name="contains"),
]

stats_config = StatisticsConfig(
    confidence_level=0.95,
    bootstrap_iterations=1000,
)

inference_config = InferenceConfig(
    batch_size=5,
    max_retries=3,
    rate_limit_rpm=100,
)

# COMMAND ----------

for model_name, model_config in models.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    task = EvalTask(
        task_id=f"comparison-{model_name}",
        name=f"Model Comparison - {model_name}",
        dataset_path=eval_table_path,
        model_config=model_config,
        prompt_template=prompt_template,
        input_column="question",
        reference_column="answer",
        metrics=metrics,
    )

    runner_config = RunnerConfig(
        model_config=model_config,
        metrics=metrics,
        statistics_config=stats_config,
        inference_config=inference_config,
    )

    runner = EvaluationRunner(spark, runner_config)
    eval_df = spark.read.format("delta").load(eval_table_path)

    results[model_name] = runner.run(eval_df, task)

    # Print results
    print(f"\nResults for {model_name}:")
    for metric_name, metric_value in results[model_name].metrics.items():
        ci = metric_value.confidence_interval
        print(f"  {metric_name}: {metric_value.value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare Results

# COMMAND ----------

import pandas as pd

# Create comparison table
comparison_data = []
for model_name, result in results.items():
    for metric_name, metric_value in result.metrics.items():
        comparison_data.append({
            "Model": model_name,
            "Metric": metric_name,
            "Value": metric_value.value,
            "CI Lower": metric_value.confidence_interval[0],
            "CI Upper": metric_value.confidence_interval[1],
            "CI Width": metric_value.ci_width,
        })

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Statistical Significance Testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if Confidence Intervals Overlap
# MAGIC
# MAGIC A quick (but conservative) check - non-overlapping CIs suggest significant difference.

# COMMAND ----------

model_names = list(results.keys())
if len(model_names) >= 2:
    model_a, model_b = model_names[0], model_names[1]

    print(f"Comparing {model_a} vs {model_b}")
    print("=" * 50)

    for metric_name in results[model_a].metrics.keys():
        metric_a = results[model_a].metrics[metric_name]
        metric_b = results[model_b].metrics[metric_name]

        overlaps = metric_a.overlaps(metric_b)
        diff = metric_b.value - metric_a.value

        print(f"\n{metric_name}:")
        print(f"  {model_a}: {metric_a.value:.4f} [{metric_a.confidence_interval[0]:.4f}, {metric_a.confidence_interval[1]:.4f}]")
        print(f"  {model_b}: {metric_b.value:.4f} [{metric_b.confidence_interval[0]:.4f}, {metric_b.confidence_interval[1]:.4f}]")
        print(f"  Difference: {diff:+.4f}")
        print(f"  CIs Overlap: {overlaps}")
        if not overlaps:
            print(f"  ** Likely significant difference **")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cost Comparison

# COMMAND ----------

cost_data = []
for model_name, result in results.items():
    cost_data.append({
        "Model": model_name,
        "Total Cost ($)": result.cost.total_cost_usd,
        "Input Tokens": result.cost.input_tokens,
        "Output Tokens": result.cost.output_tokens,
        "Requests": result.cost.num_requests,
        "Cost per Example ($)": result.cost.cost_per_example,
    })

cost_df = pd.DataFrame(cost_data)
display(cost_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and Recommendations

# COMMAND ----------

print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

for metric_name in ["f1", "exact_match", "contains"]:
    print(f"\n{metric_name.upper()}:")
    best_model = max(results.keys(), key=lambda m: results[m].metrics[metric_name].value)
    best_value = results[best_model].metrics[metric_name].value

    for model_name, result in results.items():
        value = result.metrics[metric_name].value
        ci = result.metrics[metric_name].confidence_interval
        marker = " <-- BEST" if model_name == best_model else ""
        print(f"  {model_name}: {value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]{marker}")

print("\n" + "=" * 60)
print("COST EFFICIENCY:")
cheapest = min(results.keys(), key=lambda m: results[m].cost.total_cost_usd)
print(f"  Most cost-effective: {cheapest} (${results[cheapest].cost.total_cost_usd:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC - Increase sample size for more statistical power
# MAGIC - Add stratified analysis by category
# MAGIC - Run significance tests with paired data
# MAGIC - See `03_agent_evaluation` for agent evaluation
