"""Effect size calculations for model comparisons.

Effect sizes quantify the magnitude of differences between models,
independent of sample size. This is crucial for understanding
practical significance vs statistical significance.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EffectSizeResult:
    """Result of effect size calculation.

    Args:
        name: Name of the effect size measure.
        value: Effect size value.
        ci: Confidence interval (lower, upper).
        interpretation: Qualitative interpretation (small/medium/large).
        details: Additional information.
    """

    name: str
    value: float
    ci: tuple[float, float] | None
    interpretation: str
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def __str__(self) -> str:
        if self.ci:
            return f"{self.name}={self.value:.3f} [{self.ci[0]:.3f}, {self.ci[1]:.3f}] ({self.interpretation})"
        return f"{self.name}={self.value:.3f} ({self.interpretation})"


def cohens_d(
    values_a: np.ndarray,
    values_b: np.ndarray,
    paired: bool = True,
    confidence_level: float = 0.95,
) -> EffectSizeResult:
    """Compute Cohen's d effect size.

    Measures the standardized difference between two means.

    Interpretation (rough guidelines):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        values_a: Scores from model A.
        values_b: Scores from model B.
        paired: Whether samples are paired (uses different pooling).
        confidence_level: For CI calculation.

    Returns:
        EffectSizeResult with Cohen's d value.

    Example:
        model_a_scores = [0.8, 0.7, 0.9, 0.75]
        model_b_scores = [0.85, 0.72, 0.88, 0.80]
        result = cohens_d(model_a_scores, model_b_scores)
        print(f"Effect size: {result.value:.3f} ({result.interpretation})")
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    mean_diff = mean_a - mean_b

    if paired:
        # for paired samples, use SD of differences
        diff = values_a - values_b
        sd = np.std(diff, ddof=1)
        n = len(diff)
    else:
        # pooled SD for independent samples
        n_a = len(values_a)
        n_b = len(values_b)
        var_a = np.var(values_a, ddof=1)
        var_b = np.var(values_b, ddof=1)
        sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        n = n_a + n_b

    if sd == 0:
        d = 0.0 if mean_diff == 0 else np.sign(mean_diff) * np.inf
    else:
        d = mean_diff / sd

    # calculate CI using noncentral t-distribution approximation
    ci = _cohens_d_ci(d, n, confidence_level, paired)

    interpretation = _interpret_cohens_d(abs(d))

    return EffectSizeResult(
        name="cohens_d",
        value=float(d),
        ci=ci,
        interpretation=interpretation,
        details={
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "mean_difference": float(mean_diff),
            "pooled_sd": float(sd),
            "paired": paired,
        },
    )


def _cohens_d_ci(
    d: float,
    n: int,
    confidence_level: float,
    paired: bool,
) -> tuple[float, float]:
    """Compute confidence interval for Cohen's d."""
    if not np.isfinite(d):
        return (float("-inf"), float("inf"))

    # approximate SE for d
    if paired:
        se = np.sqrt(1 / n + d**2 / (2 * n))
    else:
        se = np.sqrt(2 / n + d**2 / (2 * n))

    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)

    lower = d - z * se
    upper = d + z * se

    return (float(lower), float(upper))


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def hedges_g(
    values_a: np.ndarray,
    values_b: np.ndarray,
    confidence_level: float = 0.95,
) -> EffectSizeResult:
    """Compute Hedges' g effect size.

    Similar to Cohen's d but with correction for small sample bias.
    Preferred over Cohen's d for small samples (n < 20).

    Args:
        values_a: Scores from model A.
        values_b: Scores from model B.
        confidence_level: For CI calculation.

    Returns:
        EffectSizeResult with Hedges' g value.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    # compute Cohen's d first
    d_result = cohens_d(values_a, values_b, paired=False, confidence_level=confidence_level)
    d = d_result.value

    # apply Hedges' correction
    n_a = len(values_a)
    n_b = len(values_b)
    df = n_a + n_b - 2

    # correction factor (approximate for large df)
    if df > 0:
        correction = 1 - 3 / (4 * df - 1)
    else:
        correction = 1.0

    g = d * correction

    # adjust CI
    if d_result.ci:
        ci = (d_result.ci[0] * correction, d_result.ci[1] * correction)
    else:
        ci = None

    interpretation = _interpret_cohens_d(abs(g))

    return EffectSizeResult(
        name="hedges_g",
        value=float(g),
        ci=ci,
        interpretation=interpretation,
        details={
            "cohens_d": float(d),
            "correction_factor": float(correction),
            "df": df,
        },
    )


def odds_ratio(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    confidence_level: float = 0.95,
) -> EffectSizeResult:
    """Compute odds ratio for binary outcomes.

    For paired binary data, computes the ratio of odds that A
    outperforms B vs B outperforms A.

    Args:
        correct_a: Boolean array - True where model A was correct.
        correct_b: Boolean array - True where model B was correct.
        confidence_level: For CI calculation.

    Returns:
        EffectSizeResult with odds ratio.

    Interpretation:
    - OR = 1: no difference
    - OR > 1: A tends to be correct when B is wrong
    - OR < 1: B tends to be correct when A is wrong
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)

    # discordant pairs
    b01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B correct

    # handle edge cases
    if b10 == 0 and b01 == 0:
        or_value = 1.0
        ci = (1.0, 1.0)
        interpretation = "no difference"
    elif b10 == 0:
        or_value = float("inf")
        ci = None
        interpretation = "A always wins on discordant"
    elif b01 == 0:
        or_value = 0.0
        ci = None
        interpretation = "B always wins on discordant"
    else:
        or_value = b01 / b10

        # CI using log transformation
        log_or = np.log(or_value)
        se_log = np.sqrt(1 / b01 + 1 / b10)

        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        log_lower = log_or - z * se_log
        log_upper = log_or + z * se_log

        ci = (np.exp(log_lower), np.exp(log_upper))

        if or_value > 1.5:
            interpretation = "A better"
        elif or_value < 0.67:
            interpretation = "B better"
        else:
            interpretation = "similar"

    return EffectSizeResult(
        name="odds_ratio",
        value=float(or_value),
        ci=ci,
        interpretation=interpretation,
        details={
            "a_only_correct": int(b01),
            "b_only_correct": int(b10),
        },
    )


def relative_improvement(
    value_a: float,
    value_b: float,
    baseline_is_b: bool = True,
) -> EffectSizeResult:
    """Compute relative improvement percentage.

    Simple but useful metric for reporting.

    Args:
        value_a: Metric value for model A.
        value_b: Metric value for model B.
        baseline_is_b: If True, computes improvement of A over B.

    Returns:
        EffectSizeResult with relative improvement.
    """
    if baseline_is_b:
        baseline = value_b
        comparison = value_a
    else:
        baseline = value_a
        comparison = value_b

    if baseline == 0:
        if comparison == 0:
            rel_imp = 0.0
            interpretation = "no change"
        else:
            rel_imp = float("inf")
            interpretation = "infinite improvement"
    else:
        rel_imp = (comparison - baseline) / abs(baseline) * 100

        if rel_imp > 10:
            interpretation = "large improvement"
        elif rel_imp > 2:
            interpretation = "moderate improvement"
        elif rel_imp > -2:
            interpretation = "negligible change"
        elif rel_imp > -10:
            interpretation = "moderate decline"
        else:
            interpretation = "large decline"

    return EffectSizeResult(
        name="relative_improvement_pct",
        value=float(rel_imp),
        ci=None,  # would need bootstrap for CI
        interpretation=interpretation,
        details={
            "baseline_value": float(baseline),
            "comparison_value": float(comparison),
            "absolute_difference": float(comparison - baseline),
        },
    )
