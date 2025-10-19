"""Statistical analysis tools for evaluation results."""

from spark_llm_eval.statistics.confidence import (
    bootstrap_ci,
    bootstrap_ci_bca,
    analytical_ci_mean,
    analytical_ci_proportion,
    compare_cis,
)
from spark_llm_eval.statistics.significance import (
    SignificanceResult,
    paired_ttest,
    mcnemar_test,
    bootstrap_significance,
    wilcoxon_signed_rank,
    choose_test,
)
from spark_llm_eval.statistics.effect_size import (
    EffectSizeResult,
    cohens_d,
    hedges_g,
    odds_ratio,
    relative_improvement,
)

__all__ = [
    # confidence
    "bootstrap_ci",
    "bootstrap_ci_bca",
    "analytical_ci_mean",
    "analytical_ci_proportion",
    "compare_cis",
    # significance
    "SignificanceResult",
    "paired_ttest",
    "mcnemar_test",
    "bootstrap_significance",
    "wilcoxon_signed_rank",
    "choose_test",
    # effect size
    "EffectSizeResult",
    "cohens_d",
    "hedges_g",
    "odds_ratio",
    "relative_improvement",
]
