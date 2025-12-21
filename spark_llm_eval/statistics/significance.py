"""Hypothesis testing for model comparison.

Provides statistical tests for determining if differences between
models are significant.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SignificanceResult:
    """Result of a significance test.

    Args:
        test_name: Name of the test performed.
        statistic: Test statistic value.
        p_value: p-value from the test.
        is_significant: Whether result is significant at given threshold.
        confidence_level: Confidence level used for determination.
        details: Additional test-specific information.
    """

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def __str__(self) -> str:
        sig = "significant" if self.is_significant else "not significant"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f} ({sig})"


def paired_ttest(
    values_a: np.ndarray,
    values_b: np.ndarray,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
) -> SignificanceResult:
    """Paired t-test for comparing two models on the same examples.

    Use this when you have per-example scores from two models and want
    to test if one is significantly better than the other.

    Args:
        values_a: Per-example scores from model A.
        values_b: Per-example scores from model B.
        confidence_level: Confidence level for significance.
        alternative: "two-sided", "less", or "greater".

    Returns:
        SignificanceResult with test outcome.

    Example:
        model_a_scores = [0.8, 0.7, 0.9, 0.75]
        model_b_scores = [0.85, 0.72, 0.88, 0.80]
        result = paired_ttest(model_a_scores, model_b_scores)
        if result.is_significant:
            print("Models are significantly different")
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length for paired test")

    if len(values_a) < 2:
        raise ValueError("Need at least 2 samples for t-test")

    diff = values_a - values_b

    # handle case where all differences are zero (identical values)
    if np.allclose(diff, 0):
        return SignificanceResult(
            test_name="paired_ttest",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            confidence_level=confidence_level,
            details={
                "mean_difference": 0.0,
                "std_difference": 0.0,
                "n_samples": len(values_a),
                "alternative": alternative,
            },
        )

    statistic, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)

    alpha = 1 - confidence_level
    is_significant = p_value < alpha

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    return SignificanceResult(
        test_name="paired_ttest",
        statistic=float(statistic),
        p_value=float(p_value),
        is_significant=is_significant,
        confidence_level=confidence_level,
        details={
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "n_samples": len(values_a),
            "alternative": alternative,
        },
    )


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    confidence_level: float = 0.95,
) -> SignificanceResult:
    """McNemar's test for comparing binary outcomes.

    Use this when you have binary (correct/incorrect) results from two
    models and want to test if the error patterns differ significantly.

    Args:
        correct_a: Boolean array - True where model A was correct.
        correct_b: Boolean array - True where model B was correct.
        confidence_level: Confidence level.

    Returns:
        SignificanceResult with test outcome.

    Example:
        a_correct = [True, True, False, True, False]
        b_correct = [True, False, False, True, True]
        result = mcnemar_test(a_correct, b_correct)
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)

    if len(correct_a) != len(correct_b):
        raise ValueError("Arrays must have same length")

    # build contingency table
    # b01: A correct, B wrong
    # b10: A wrong, B correct
    b01 = np.sum(correct_a & ~correct_b)
    b10 = np.sum(~correct_a & correct_b)

    # McNemar's test focuses on discordant pairs
    n_discordant = b01 + b10

    if n_discordant < 10:
        # use exact binomial test for small samples
        # under null, discordant pairs are equally likely
        binom_result = stats.binomtest(b01, n_discordant, 0.5)
        p_value = binom_result.pvalue
        statistic = float(b01)
        test_name = "mcnemar_exact"
    else:
        # use chi-squared approximation with continuity correction
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        test_name = "mcnemar_chi2"

    alpha = 1 - confidence_level
    is_significant = p_value < alpha

    return SignificanceResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        is_significant=is_significant,
        confidence_level=confidence_level,
        details={
            "a_only_correct": int(b01),
            "b_only_correct": int(b10),
            "both_correct": int(np.sum(correct_a & correct_b)),
            "both_wrong": int(np.sum(~correct_a & ~correct_b)),
            "n_discordant": int(n_discordant),
        },
    )


def bootstrap_significance(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic: callable = np.mean,
    n_iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = 42,
) -> SignificanceResult:
    """Bootstrap permutation test for significance.

    Non-parametric test that makes fewer assumptions than t-test.
    Good for non-normal distributions or small samples.

    Args:
        values_a: Scores from model/condition A.
        values_b: Scores from model/condition B.
        statistic: Function to compute statistic (default: mean).
        n_iterations: Number of permutations.
        confidence_level: Confidence level.
        seed: Random seed.

    Returns:
        SignificanceResult with test outcome.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    rng = np.random.default_rng(seed)
    n_a = len(values_a)
    n_b = len(values_b)

    observed_diff = statistic(values_a) - statistic(values_b)

    # permutation test under null hypothesis
    pooled = np.concatenate([values_a, values_b])
    null_diffs = np.zeros(n_iterations)

    for i in range(n_iterations):
        perm = rng.permutation(pooled)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]
        null_diffs[i] = statistic(perm_a) - statistic(perm_b)

    # two-tailed p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    alpha = 1 - confidence_level
    is_significant = p_value < alpha

    return SignificanceResult(
        test_name="bootstrap_permutation",
        statistic=float(observed_diff),
        p_value=float(p_value),
        is_significant=is_significant,
        confidence_level=confidence_level,
        details={
            "observed_difference": float(observed_diff),
            "n_iterations": n_iterations,
            "n_samples_a": n_a,
            "n_samples_b": n_b,
        },
    )


def wilcoxon_signed_rank(
    values_a: np.ndarray,
    values_b: np.ndarray,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
) -> SignificanceResult:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test. Doesn't assume
    normal distribution of differences.

    Args:
        values_a: Per-example scores from model A.
        values_b: Per-example scores from model B.
        confidence_level: Confidence level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        SignificanceResult with test outcome.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length")

    # need to handle zero differences
    diff = values_a - values_b
    non_zero = diff != 0

    if np.sum(non_zero) < 2:
        # not enough non-zero differences
        return SignificanceResult(
            test_name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            confidence_level=confidence_level,
            details={"warning": "insufficient non-zero differences"},
        )

    statistic, p_value = stats.wilcoxon(
        values_a,
        values_b,
        alternative=alternative,
        zero_method="wilcox",
    )

    alpha = 1 - confidence_level
    is_significant = p_value < alpha

    return SignificanceResult(
        test_name="wilcoxon",
        statistic=float(statistic),
        p_value=float(p_value),
        is_significant=is_significant,
        confidence_level=confidence_level,
        details={
            "alternative": alternative,
            "n_non_zero_diffs": int(np.sum(non_zero)),
        },
    )


def choose_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    metric_type: str = "continuous",
    paired: bool = True,
) -> str:
    """Recommend appropriate test based on data characteristics.

    Args:
        values_a: First set of values.
        values_b: Second set of values.
        metric_type: "continuous" or "binary".
        paired: Whether samples are paired (same examples).

    Returns:
        Recommended test name.
    """
    n = len(values_a)

    if metric_type == "binary":
        return "mcnemar" if paired else "chi2"

    if not paired:
        return "bootstrap_permutation"

    # for paired continuous data
    if n < 20:
        return "wilcoxon"  # non-parametric for small samples

    # check normality of differences
    diff = np.asarray(values_a) - np.asarray(values_b)
    _, p_normal = stats.shapiro(diff[: min(5000, len(diff))])

    if p_normal < 0.05:
        return "wilcoxon"  # non-parametric if clearly non-normal

    return "paired_ttest"  # default to parametric
