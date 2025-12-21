# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Evaluation with spark-llm-eval
# MAGIC
# MAGIC This notebook demonstrates how to evaluate Retrieval-Augmented Generation (RAG) systems using spark-llm-eval.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - Evaluate RAG outputs with specialized metrics (context relevance, faithfulness, answer relevance)
# MAGIC - Compare LLM-as-judge across multiple providers (OpenAI, Claude, Gemini)
# MAGIC - Use embedding-based metrics for faster, cheaper evaluation
# MAGIC - Analyze results with confidence intervals
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks Runtime 14.0+ with ML
# MAGIC - API keys for OpenAI, Anthropic, and Google stored in Databricks secrets

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
from spark_llm_eval.evaluation import RAGJudgeConfig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Judge Models (OpenAI, Claude, Gemini)
# MAGIC
# MAGIC We'll evaluate RAG outputs using judges from all three major providers.

# COMMAND ----------

# OpenAI Judge
openai_model = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini",
    api_key_secret="openai-api-key",  # Or use dbutils.secrets.get("scope", "key")
    temperature=0.0,
)
openai_judge = RAGJudgeConfig(model_config=openai_model)

# Anthropic Claude Judge
claude_model = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307",
    api_key_secret="anthropic-api-key",
    temperature=0.0,
)
claude_judge = RAGJudgeConfig(model_config=claude_model)

# Google Gemini Judge
gemini_model = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-1.5-flash",
    api_key_secret="google-api-key",
    temperature=0.0,
)
gemini_judge = RAGJudgeConfig(model_config=gemini_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Sample RAG Dataset
# MAGIC
# MAGIC Create a sample RAG evaluation dataset with queries, retrieved context, generated answers, and ground truth.

# COMMAND ----------

# Sample RAG evaluation data
rag_data = [
    {
        "query": "What is the capital of France?",
        "context": "France is a country in Western Europe. Paris is the capital and largest city of France. The city has a population of over 2 million people.",
        "answer": "The capital of France is Paris.",
        "ground_truth": "Paris",
    },
    {
        "query": "Who invented the telephone?",
        "context": "Alexander Graham Bell was a Scottish-born inventor. He is credited with inventing the first practical telephone in 1876. Bell also worked on other communication devices.",
        "answer": "Alexander Graham Bell invented the telephone in 1876.",
        "ground_truth": "Alexander Graham Bell",
    },
    {
        "query": "What is photosynthesis?",
        "context": "Photosynthesis is the process by which plants convert sunlight into energy. Plants use carbon dioxide and water to produce glucose and oxygen. This process occurs in the chloroplasts.",
        "answer": "Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        "ground_truth": "The process by which plants convert sunlight into energy",
    },
    {
        "query": "What is the largest planet in our solar system?",
        "context": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than twice that of all other planets combined. Jupiter has at least 95 known moons.",
        "answer": "Jupiter is the largest planet in our solar system. It is a gas giant.",
        "ground_truth": "Jupiter",
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "context": "William Shakespeare was an English playwright and poet. He wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet. He lived during the Elizabethan era.",
        "answer": "William Shakespeare wrote Romeo and Juliet.",
        "ground_truth": "William Shakespeare",
    },
    {
        "query": "What is the speed of light?",
        "context": "Light travels at approximately 299,792,458 meters per second in a vacuum. This is considered the universal speed limit. Nothing can travel faster than light according to Einstein's theory of relativity.",
        "answer": "The speed of light is approximately 299,792,458 meters per second.",
        "ground_truth": "299,792,458 meters per second",
    },
    {
        "query": "What causes the seasons?",
        "context": "The Earth's axis is tilted at about 23.5 degrees. As Earth orbits the Sun, different parts receive varying amounts of sunlight. This axial tilt is the primary cause of seasons.",
        "answer": "Seasons are caused by the Earth's axial tilt of about 23.5 degrees, which affects how much sunlight different parts of Earth receive.",
        "ground_truth": "The tilt of Earth's axis",
    },
    {
        "query": "What is the Mona Lisa?",
        "context": "The Mona Lisa is a famous painting by Leonardo da Vinci. It was painted in the early 16th century. The painting is displayed at the Louvre Museum in Paris.",
        "answer": "The Mona Lisa is a famous painting by Leonardo da Vinci, currently displayed at the Louvre Museum.",
        "ground_truth": "A painting by Leonardo da Vinci",
    },
]

df = spark.createDataFrame(rag_data)
display(df)

# COMMAND ----------

# Save to Delta Lake for reproducibility
rag_table_path = "/tmp/spark_llm_eval/rag_eval_data"
df.write.format("delta").mode("overwrite").save(rag_table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure RAG Metrics
# MAGIC
# MAGIC We'll use both LLM-as-judge and embedding-based metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Available RAG Metrics
# MAGIC
# MAGIC | Metric | Type | Description |
# MAGIC |--------|------|-------------|
# MAGIC | `context_relevance` | LLM-Judge | Is the retrieved context relevant to the query? |
# MAGIC | `faithfulness` | LLM-Judge | Is the answer grounded in the context (no hallucinations)? |
# MAGIC | `answer_relevance` | LLM-Judge | Does the answer address the query? |
# MAGIC | `context_precision` | LLM-Judge | Ranking quality of retrieved chunks |
# MAGIC | `context_recall` | LLM-Judge | Coverage of ground truth in context |
# MAGIC | `context_relevance_embedding` | Embedding | Cosine similarity (faster, cheaper) |
# MAGIC | `answer_relevance_embedding` | Embedding | Cosine similarity (faster, cheaper) |
# MAGIC | `faithfulness_nli` | NLI | Entailment-based (faster, cheaper) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate with OpenAI Judge

# COMMAND ----------

# Define task with OpenAI judge metrics
task_openai = EvalTask(
    task_id="rag-eval-openai",
    name="RAG Evaluation (OpenAI Judge)",
    dataset_path=rag_table_path,
    model_config=openai_model,  # For any generation if needed
    prompt_template="{{ query }}",  # Simple pass-through
    input_column="query",
    reference_column="ground_truth",
    context_columns=["context"],  # RAG context column
    metrics=[
        MetricConfig(name="context_relevance", params={"judge_config": openai_judge}),
        MetricConfig(name="faithfulness", params={"judge_config": openai_judge}),
        MetricConfig(name="answer_relevance", params={"judge_config": openai_judge}),
    ],
)

# Configure runner
runner_config_openai = RunnerConfig(
    model_config=openai_model,
    metrics=task_openai.metrics,
    statistics_config=StatisticsConfig(confidence_level=0.95, bootstrap_iterations=500),
)

runner_openai = EvaluationRunner(spark, runner_config_openai)

# COMMAND ----------

# For RAG evaluation, we already have predictions (answers) - add them as a column
eval_df = spark.read.format("delta").load(rag_table_path)
eval_df_with_pred = eval_df.withColumn("prediction", F.col("answer"))

# Run evaluation
result_openai = runner_openai.run(eval_df_with_pred, task_openai)

# COMMAND ----------

print("=" * 60)
print("OpenAI Judge Results")
print("=" * 60)
for metric_name, metric_value in result_openai.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}:")
    print(f"  Value: {metric_value.value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate with Claude Judge

# COMMAND ----------

task_claude = EvalTask(
    task_id="rag-eval-claude",
    name="RAG Evaluation (Claude Judge)",
    dataset_path=rag_table_path,
    model_config=claude_model,
    prompt_template="{{ query }}",
    input_column="query",
    reference_column="ground_truth",
    context_columns=["context"],
    metrics=[
        MetricConfig(name="context_relevance", params={"judge_config": claude_judge}),
        MetricConfig(name="faithfulness", params={"judge_config": claude_judge}),
        MetricConfig(name="answer_relevance", params={"judge_config": claude_judge}),
    ],
)

runner_config_claude = RunnerConfig(
    model_config=claude_model,
    metrics=task_claude.metrics,
    statistics_config=StatisticsConfig(confidence_level=0.95, bootstrap_iterations=500),
)

runner_claude = EvaluationRunner(spark, runner_config_claude)
result_claude = runner_claude.run(eval_df_with_pred, task_claude)

# COMMAND ----------

print("=" * 60)
print("Claude Judge Results")
print("=" * 60)
for metric_name, metric_value in result_claude.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}:")
    print(f"  Value: {metric_value.value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate with Gemini Judge

# COMMAND ----------

task_gemini = EvalTask(
    task_id="rag-eval-gemini",
    name="RAG Evaluation (Gemini Judge)",
    dataset_path=rag_table_path,
    model_config=gemini_model,
    prompt_template="{{ query }}",
    input_column="query",
    reference_column="ground_truth",
    context_columns=["context"],
    metrics=[
        MetricConfig(name="context_relevance", params={"judge_config": gemini_judge}),
        MetricConfig(name="faithfulness", params={"judge_config": gemini_judge}),
        MetricConfig(name="answer_relevance", params={"judge_config": gemini_judge}),
    ],
)

runner_config_gemini = RunnerConfig(
    model_config=gemini_model,
    metrics=task_gemini.metrics,
    statistics_config=StatisticsConfig(confidence_level=0.95, bootstrap_iterations=500),
)

runner_gemini = EvaluationRunner(spark, runner_config_gemini)
result_gemini = runner_gemini.run(eval_df_with_pred, task_gemini)

# COMMAND ----------

print("=" * 60)
print("Gemini Judge Results")
print("=" * 60)
for metric_name, metric_value in result_gemini.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}:")
    print(f"  Value: {metric_value.value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Compare Judges

# COMMAND ----------

# Create comparison table
comparison_data = []
for metric in ["context_relevance", "faithfulness", "answer_relevance"]:
    comparison_data.append({
        "Metric": metric,
        "OpenAI": f"{result_openai.metrics[metric].value:.3f}",
        "Claude": f"{result_claude.metrics[metric].value:.3f}",
        "Gemini": f"{result_gemini.metrics[metric].value:.3f}",
    })

comparison_df = spark.createDataFrame(comparison_data)
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Embedding-Based Metrics (Faster & Cheaper)
# MAGIC
# MAGIC Embedding metrics don't require LLM API calls - they use local sentence transformers.

# COMMAND ----------

task_embedding = EvalTask(
    task_id="rag-eval-embedding",
    name="RAG Evaluation (Embedding)",
    dataset_path=rag_table_path,
    model_config=openai_model,  # Not used for embedding metrics
    prompt_template="{{ query }}",
    input_column="query",
    reference_column="ground_truth",
    context_columns=["context"],
    metrics=[
        MetricConfig(name="context_relevance_embedding"),
        MetricConfig(name="answer_relevance_embedding"),
        # faithfulness_nli requires sentence-transformers with cross-encoder
        # MetricConfig(name="faithfulness_nli"),
    ],
)

runner_config_embedding = RunnerConfig(
    model_config=openai_model,
    metrics=task_embedding.metrics,
    statistics_config=StatisticsConfig(confidence_level=0.95),
)

runner_embedding = EvaluationRunner(spark, runner_config_embedding)
result_embedding = runner_embedding.run(eval_df_with_pred, task_embedding)

# COMMAND ----------

print("=" * 60)
print("Embedding-Based Metrics Results")
print("=" * 60)
for metric_name, metric_value in result_embedding.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}:")
    print(f"  Value: {metric_value.value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cost Comparison
# MAGIC
# MAGIC Embedding metrics are significantly cheaper than LLM-as-judge.

# COMMAND ----------

print("Cost Comparison (per 1000 evaluations):")
print("-" * 50)
print("LLM-as-Judge (GPT-4o-mini): ~$0.15-0.30")
print("LLM-as-Judge (Claude Haiku): ~$0.10-0.20")
print("LLM-as-Judge (Gemini Flash): ~$0.05-0.10")
print("Embedding-based: ~$0.00 (local compute)")
print()
print("Trade-off: LLM judges are more nuanced but expensive.")
print("Embedding metrics are fast and free but less accurate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Precision and Recall Metrics (Require Ground Truth)

# COMMAND ----------

# Context precision and recall require ground truth reference
task_precision_recall = EvalTask(
    task_id="rag-eval-precision-recall",
    name="RAG Precision/Recall Evaluation",
    dataset_path=rag_table_path,
    model_config=openai_model,
    prompt_template="{{ query }}",
    input_column="query",
    reference_column="ground_truth",
    context_columns=["context"],
    metrics=[
        MetricConfig(name="context_precision", params={"judge_config": openai_judge}),
        MetricConfig(name="context_recall", params={"judge_config": openai_judge}),
    ],
)

runner_config_pr = RunnerConfig(
    model_config=openai_model,
    metrics=task_precision_recall.metrics,
    statistics_config=StatisticsConfig(confidence_level=0.95),
)

runner_pr = EvaluationRunner(spark, runner_config_pr)
result_pr = runner_pr.run(eval_df_with_pred, task_precision_recall)

# COMMAND ----------

print("=" * 60)
print("Precision/Recall Metrics Results")
print("=" * 60)
for metric_name, metric_value in result_pr.metrics.items():
    ci_lower, ci_upper = metric_value.confidence_interval
    print(f"{metric_name}:")
    print(f"  Value: {metric_value.value:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC
# MAGIC 1. **Multi-provider evaluation**: Compare RAG outputs using OpenAI, Claude, and Gemini as judges
# MAGIC 2. **Core RAG metrics**: context_relevance, faithfulness, answer_relevance
# MAGIC 3. **Advanced metrics**: context_precision and context_recall (require ground truth)
# MAGIC 4. **Embedding alternatives**: Faster, cheaper metrics using sentence transformers
# MAGIC 5. **Statistical rigor**: All metrics include confidence intervals
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Scale to larger datasets using Spark's distributed execution
# MAGIC - Track experiments with MLflow integration
# MAGIC - Build automated RAG quality monitoring pipelines
