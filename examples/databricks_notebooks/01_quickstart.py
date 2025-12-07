# Databricks notebook source
# MAGIC %md
# MAGIC # spark-llm-eval Quickstart
# MAGIC
# MAGIC This notebook demonstrates the basic usage of spark-llm-eval for evaluating LLM outputs at scale.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks Runtime 14.0+ with ML
# MAGIC - OpenAI API key stored in Databricks secrets

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

# COMMAND ----------

# Configure the model - using Databricks secrets for API key
model_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini",
    api_key_secret="openai-api-key",  # Or use dbutils.secrets.get("scope", "key")
    temperature=0.0,
    max_tokens=256,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare Evaluation Dataset
# MAGIC
# MAGIC Create a sample QA dataset. In production, load from Delta Lake.

# COMMAND ----------

# Sample evaluation data
eval_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the speed of light?", "answer": "299,792,458 meters per second"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
]

df = spark.createDataFrame(eval_data)
display(df)

# COMMAND ----------

# Save to Delta Lake (optional - for reproducibility)
eval_table_path = "/tmp/spark_llm_eval/quickstart_eval_data"
df.write.format("delta").mode("overwrite").save(eval_table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Evaluation Task

# COMMAND ----------

# Define the evaluation task
task = EvalTask(
    task_id="quickstart-eval-001",
    name="QA Quickstart Evaluation",
    dataset_path=eval_table_path,
    model_config=model_config,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configure and Run Evaluation

# COMMAND ----------

# Configure the runner
runner_config = RunnerConfig(
    model_config=model_config,
    metrics=task.metrics,
    statistics_config=StatisticsConfig(
        confidence_level=0.95,
        bootstrap_iterations=1000,
    ),
    inference_config=InferenceConfig(
        batch_size=5,
        max_retries=3,
        rate_limit_rpm=500,
    ),
)

# Create runner and execute
runner = EvaluationRunner(spark, runner_config)

# COMMAND ----------

# Load data and run evaluation
eval_df = spark.read.format("delta").load(eval_table_path)
result = runner.run(eval_df, task)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. View Results

# COMMAND ----------

# Print summary
print(result.summary())

# COMMAND ----------

# Access individual metrics with confidence intervals
for metric_name, metric_value in result.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}:")
    print(f"  Value: {metric_value.value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Sample size: {metric_value.sample_size}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. View Predictions (Optional)
# MAGIC
# MAGIC If you saved results, you can view the per-example predictions.

# COMMAND ----------

# Check cost breakdown
print(f"Total cost: ${result.cost.total_cost_usd:.4f}")
print(f"Total requests: {result.cost.num_requests}")
print(f"Input tokens: {result.cost.input_tokens}")
print(f"Output tokens: {result.cost.output_tokens}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC - See `02_model_comparison` for comparing multiple models
# MAGIC - See `03_agent_evaluation` for evaluating agent trajectories
# MAGIC - Check the [documentation](https://github.com/bassrehab/spark-llm-eval) for advanced features
