"""CI computation - bootstrap and analytical methods."""

import logging
from collections.abc import Callable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_iterations: int = 1000,
    seed: int | None = 42,
) -> tuple[float, tuple[float, float], float]:
    """Bootstrap CI using percentile method. Returns (point, (lo, hi), se)."""
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return 0.0, (0.0, 0.0), 0.0

    if n == 1:
        val = float(statistic(values))
        return val, (val, val), 0.0

    rng = np.random.default_rng(seed)

    # compute bootstrap distribution
    boot_stats = np.zeros(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    # point estimate on original data
    point_estimate = float(statistic(values))

    # percentile CI
    alpha = 1 - confidence_level
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    # standard error is std of bootstrap distribution
    se = float(np.std(boot_stats, ddof=1))

    return point_estimate, (lower, upper), se


def bootstrap_ci_bca(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_iterations: int = 1000,
    seed: int | None = 42,
) -> tuple[float, tuple[float, float], float]:
    """BCa bootstrap - better for skewed data but falls back to percentile if it fails."""
    values = np.asarray(values)
    n = len(values)

    if n < 2:
        return bootstrap_ci(values, statistic, confidence_level, n_iterations, seed)

    rng = np.random.default_rng(seed)

    # bootstrap distribution
    boot_stats = np.zeros(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    point_estimate = float(statistic(values))

    # bias correction - can blow up sometimes hence the fallback
    z0 = stats.norm.ppf(np.mean(boot_stats < point_estimate))
    if not np.isfinite(z0):
        return bootstrap_ci(values, statistic, confidence_level, n_iterations, seed)

    # acceleration factor using jackknife
    jack_stats = np.zeros(n)
    for i in range(n):
        jack_sample = np.delete(values, i)
        jack_stats[i] = statistic(jack_sample)

    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)

    if den == 0:
        return bootstrap_ci(values, statistic, confidence_level, n_iterations, seed)

    a = num / den

    # adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa adjustment
    alpha_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    alpha_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

    if not (0 < alpha_lower < 1 and 0 < alpha_upper < 1):
        return bootstrap_ci(values, statistic, confidence_level, n_iterations, seed)

    lower = float(np.percentile(boot_stats, 100 * alpha_lower))
    upper = float(np.percentile(boot_stats, 100 * alpha_upper))
    se = float(np.std(boot_stats, ddof=1))

    return point_estimate, (lower, upper), se


def analytical_ci_mean(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, tuple[float, float], float]:
    """CI for mean using t-dist. Assumes roughly normal data."""
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return 0.0, (0.0, 0.0), 0.0

    if n == 1:
        val = float(values[0])
        return val, (val, val), 0.0

    mean = float(np.mean(values))
    se = float(stats.sem(values))

    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin = t_crit * se
    lower = mean - margin
    upper = mean + margin

    return mean, (lower, upper), se


def analytical_ci_proportion(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
    method: str = "wilson",
) -> tuple[float, tuple[float, float], float]:
    """CI for proportions. Wilson is usually what you want."""
    if total == 0:
        return 0.0, (0.0, 0.0), 0.0

    p = successes / total

    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)

    if method == "wilson":
        # wilson score - handles edge cases better than wald
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
    elif method == "normal":
        # wald interval - simple but can give bounds outside [0,1] lol
        se = np.sqrt(p * (1 - p) / total)
        margin = z * se
        lower = max(0.0, p - margin)
        upper = min(1.0, p + margin)
    elif method == "clopper-pearson":
        # exact but overly conservative
        lower = (
            stats.beta.ppf(alpha / 2, successes, total - successes + 1) if successes > 0 else 0.0
        )
        upper = (
            stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
            if successes < total
            else 1.0
        )
    else:
        raise ValueError(f"unknown method {method}, try wilson/normal/clopper-pearson")

    # standard error approximation
    se = np.sqrt(p * (1 - p) / total) if total > 0 else 0.0

    return float(p), (float(lower), float(upper)), float(se)


def compare_cis(ci1: tuple[float, float], ci2: tuple[float, float]) -> dict:
    """Quick comparison of two CIs - do they overlap, by how much, etc."""
    lo1, hi1 = ci1
    lo2, hi2 = ci2

    overlap = not (hi1 < lo2 or hi2 < lo1)

    # calculate overlap amount if they do overlap
    if overlap:
        overlap_start = max(lo1, lo2)
        overlap_end = min(hi1, hi2)
        overlap_width = overlap_end - overlap_start
    else:
        overlap_width = 0.0

    return {
        "overlaps": overlap,
        "overlap_width": overlap_width,
        "ci1_width": hi1 - lo1,
        "ci2_width": hi2 - lo2,
        "likely_different": not overlap,  # rough heuristic
    }
