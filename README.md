# spark-llm-eval

**Distributed LLM Evaluation Framework for Apache Spark**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySpark 3.5+](https://img.shields.io/badge/pyspark-3.5+-orange.svg)](https://spark.apache.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/spark-llm-eval?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/project/spark-llm-eval)

---

## Why spark-llm-eval?

Current LLM evaluation tools are designed for single-machine execution. When you need to evaluate models against millions of examples - customer support tickets, documents, transactions - they don't scale.

**spark-llm-eval** is built Spark-native from the ground up:

| Feature | Single-Machine Tools | spark-llm-eval |
|---------|---------------------|----------------|
| Scale | ~10K examples | Millions of examples |
| Statistical Rigor | Point estimates | Confidence intervals, significance tests |
| Reproducibility | Manual tracking | Delta Lake versioning + MLflow |
| Cost | Naive API calls | Caching, batching, rate limiting |
| Enterprise Ready | Limited | Unity Catalog, governance, audit |

## Features

- **Distributed Inference**: Spark-native with Pandas UDFs, scales linearly with executors
- **Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini
- **Statistical Rigor**: Bootstrap CIs, paired t-tests, McNemar's test, Wilcoxon, effect sizes
- **Smart Rate Limiting**: Token bucket algorithm with RPM and TPM limits
- **Comprehensive Metrics**: Lexical, semantic (BERTScore, embeddings), LLM-as-judge
- **Response Caching**: Delta-backed content-addressable cache with TTL, replay mode for metrics iteration
- **MLflow Integration**: Full experiment tracking, artifact logging, model comparison
- **Delta Lake Native**: Versioned datasets, time travel, ACID transactions

## Installation

### PyPI (Recommended)

```bash
pip install spark-llm-eval
```

### Databricks

In a Databricks notebook:
```python
%pip install spark-llm-eval
```

Or attach to your cluster via the Libraries UI using PyPI coordinates: `spark-llm-eval`

### Development

```bash
git clone https://github.com/bassrehab/spark-llm-eval.git
cd spark-llm-eval
pip install -e ".[dev]"
```

## Quick Start

```python
from pyspark.sql import SparkSession
from spark_llm_eval.core.config import ModelConfig, ModelProvider, MetricConfig, StatisticsConfig
from spark_llm_eval.core.task import EvalTask
from spark_llm_eval.orchestrator import EvaluationRunner, RunnerConfig
from spark_llm_eval.datasets import load_dataset

# Initialize Spark
spark = SparkSession.builder.appName("llm-eval").getOrCreate()

# Load evaluation dataset
data = load_dataset(
    spark,
    table_path="/mnt/delta/datasets/qa_test",
    input_column="question",
    reference_column="answer",
)

# Configure model
model_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o",
    api_key_secret="secrets/openai_key",
)

# Define evaluation task
task = EvalTask(
    task_id="qa-eval-001",
    name="QA Model Evaluation",
    dataset_path="/mnt/delta/datasets/qa_test",
    model_config=model_config,
    prompt_template="""Answer the following question concisely.

Question: {{ input }}

Answer:""",
    metrics=[
        MetricConfig(name="exact_match"),
        MetricConfig(name="f1"),
    ],
)

# Configure runner
runner_config = RunnerConfig(
    model_config=model_config,
    metrics=task.metrics,
    statistics_config=StatisticsConfig(confidence_level=0.95),
)

# Run evaluation
runner = EvaluationRunner(spark, runner_config)
result = runner.run(data, task)

# Results include confidence intervals
for name, metric in result.metrics.items():
    ci = metric.confidence_interval
    print(f"{name}: {metric.value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
```

## Supported Metrics

### Lexical Metrics
- **exact_match**: Exact string match (case-insensitive)
- **f1**: Token-level F1 score
- **bleu**: BLEU score (1-4 grams)
- **rouge_l**: ROUGE-L score
- **contains**: Substring containment check
- **length_ratio**: Response length relative to reference

### Semantic Metrics
- **bertscore**: BERTScore (precision, recall, F1) using contextual embeddings
- **embedding_similarity**: Cosine similarity of sentence embeddings
- **semantic_similarity**: Overall semantic similarity score

### LLM-as-Judge
- **llm_judge**: Customizable LLM-based evaluation with rubrics
- **pairwise_comparison**: Compare two outputs head-to-head
- **pointwise_grading**: Score individual outputs on defined criteria

### RAG Metrics
- **context_relevance**: Is retrieved context relevant to the query?
- **faithfulness**: Is the answer grounded in context (no hallucinations)?
- **answer_relevance**: Does the answer address the query?
- **context_precision**: Ranking quality of retrieved chunks
- **context_recall**: Coverage of ground truth in retrieved context
- **context_relevance_embedding**: Embedding-based relevance (faster, cheaper)
- **answer_relevance_embedding**: Embedding-based relevance (faster, cheaper)
- **faithfulness_nli**: NLI-based faithfulness (faster, cheaper)

## Statistical Features

All metrics include:
- **Confidence intervals** (bootstrap or analytical)
- **Standard error**
- **Sample size**

Model comparisons include:
- **Paired t-test** for continuous metrics
- **McNemar's test** for binary outcomes
- **Wilcoxon signed-rank** for non-parametric comparison
- **Effect size** (Cohen's d, Hedges' g)

```python
from spark_llm_eval.statistics import paired_ttest, cohens_d

# Compare two models
result = paired_ttest(model_a_scores, model_b_scores)
print(f"p-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")

# Compute effect size
effect = cohens_d(model_a_scores, model_b_scores)
print(f"Cohen's d: {effect.value:.3f} ({effect.interpretation})")
```

## Configuration

### Inference Configuration

```python
from spark_llm_eval.core.config import InferenceConfig

inference_config = InferenceConfig(
    batch_size=32,              # Examples per batch
    max_retries=3,              # Retry on failure
    timeout=60.0,               # Request timeout (seconds)
    rate_limit_rpm=10000,       # Requests per minute
    rate_limit_tpm=1000000,     # Tokens per minute
)
```

### Statistics Configuration

```python
from spark_llm_eval.core.config import StatisticsConfig

stats_config = StatisticsConfig(
    confidence_level=0.95,      # 95% confidence intervals
    bootstrap_iterations=1000,  # Bootstrap samples
    significance_threshold=0.05,# p-value threshold
    compute_effect_size=True    # Include Cohen's d
)
```

## MLflow Integration

```python
from spark_llm_eval.tracking import create_tracker

tracker = create_tracker(
    experiment_name="llm-evaluations",
    run_name="gpt4-baseline",
)

with tracker.start_run():
    result = runner.run(data, task)
    tracker.log_results_summary({
        name: {"value": m.value, "ci_lower": m.confidence_interval[0], "ci_upper": m.confidence_interval[1]}
        for name, m in result.metrics.items()
    })
```

## Development

```bash
# Clone repository
git clone https://github.com/bassrehab/spark-llm-eval.git
cd spark-llm-eval

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit -v

# Run linting
ruff check spark_llm_eval
black --check spark_llm_eval
```

## Roadmap

### Completed
- [x] Core framework with OpenAI support
- [x] Lexical metrics with statistical rigor
- [x] MLflow integration
- [x] Delta Lake dataset integration
- [x] Multi-provider support (Anthropic, Google Gemini)
- [x] Semantic metrics (BERTScore, embeddings)
- [x] LLM-as-judge
- [x] Agent evaluation (trajectories, tool use, multi-agent debate)
- [x] RAG evaluation metrics (context relevance, faithfulness, answer relevance, precision, recall)
- [x] Databricks notebook examples
- [x] PyPI package release
- [x] Response caching (Delta-backed) with replay mode

### Planned
- [ ] vLLM/local model support
- [ ] Unity Catalog integration
- [ ] Databricks Asset Bundle distribution

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use spark-llm-eval in your research, please cite:

```bibtex
@software{spark_llm_eval,
  author = {Mitra, Subhadip},
  title = {spark-llm-eval: Distributed LLM Evaluation Framework},
  year = {2025},
  url = {https://github.com/bassrehab/spark-llm-eval}
}
```

## Acknowledgments

- Inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [ragas](https://github.com/explodinggradients/ragas), and [deepeval](https://github.com/confident-ai/deepeval)
- Statistical methods based on [Berg-Kirkpatrick et al. (2012)](https://aclanthology.org/D12-1091/)
- Built for the [Databricks](https://databricks.com/) ecosystem
