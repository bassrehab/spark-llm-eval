"""Configuration dataclasses. Treat these as immutable after creation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DATABRICKS = "databricks"
    VLLM = "vllm"
    # TODO: azure openai - need to figure out the auth story


@dataclass(frozen=True)
class ModelConfig:
    """Config for the model being evaluated. Use api_key_secret for Databricks
    secret paths (like 'scope/key'), don't put actual keys here."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.0  # 0 for deterministic evals
    max_tokens: int = 1024
    api_key_secret: str | None = None
    endpoint: str | None = None  # for self-hosted/proxy
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"bad temperature: {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass
class MetricConfig:
    """Config for a metric like exact_match, bleu, bertscore etc."""
    name: str
    metric_type: str = "lexical"  # lexical, semantic, llm_judge, custom
    params: dict[str, Any] = field(default_factory=dict)
    requires_reference: bool = True
    judge_model: ModelConfig | None = None  # only for llm_judge type

    @property
    def kwargs(self):
        return self.params  # alias

    def __post_init__(self):
        valid = {"lexical", "semantic", "llm_judge", "custom"}
        if self.metric_type not in valid:
            raise ValueError(f"invalid metric_type: {self.metric_type}")
        if self.metric_type == "llm_judge" and not self.judge_model:
            raise ValueError("need judge_model for llm_judge metrics")


@dataclass(frozen=True)
class SamplingConfig:
    """For when you dont want to eval the whole dataset."""
    strategy: str = "random"  # random, stratified, systematic
    sample_size: int | None = None
    sample_fraction: float | None = None
    stratify_columns: list[str] = field(default_factory=list)
    seed: int = 42

    def __post_init__(self):
        if self.sample_size is None and self.sample_fraction is None:
            raise ValueError("need sample_size or sample_fraction")
        if self.sample_size and self.sample_fraction:
            raise ValueError("pick one: sample_size or sample_fraction")
        if self.strategy == "stratified" and not self.stratify_columns:
            raise ValueError("stratified needs stratify_columns")


@dataclass(frozen=True)
class StatisticsConfig:
    """Controls CIs, significance testing etc."""
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000  # 10000 for final results
    significance_threshold: float = 0.05
    compute_effect_size: bool = True
    ci_method: str = "bootstrap"  # or "analytical"

    def __post_init__(self):
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level should be between 0 and 1")
        # technically <100 works but the CIs will be garbage
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations too low, need at least 100")
        if self.ci_method not in ("bootstrap", "analytical"):
            raise ValueError(f"unknown ci_method: {self.ci_method}")


@dataclass(frozen=True)
class InferenceConfig:
    """Inference settings - batch size, retries, rate limits."""
    batch_size: int = 32  # tune based on your rate limits
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    rate_limit_rpm: int | None = None  # requests/min
    rate_limit_tpm: int | None = None  # tokens/min
    enable_caching: bool = True
    cache_table: str | None = None

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size cant be < 1")
        # XXX: should we warn if caching enabled but no cache_table?
        # for now the orchestrator sets it if missing


@dataclass
class OutputConfig:
    """Where to save stuff."""
    results_path: str | None = None
    predictions_table: str | None = None
    metrics_table: str | None = None
    save_results: bool = False
    save_predictions: bool = True
    include_prompts: bool = False  # can get big
