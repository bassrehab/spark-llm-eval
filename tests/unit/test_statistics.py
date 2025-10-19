"""Tests for statistics module."""

import pytest
import numpy as np
from spark_llm_eval.statistics.confidence import (
    bootstrap_ci,
    bootstrap_ci_bca,
    analytical_ci_mean,
    analytical_ci_proportion,
    compare_cis,
)
from spark_llm_eval.statistics.significance import (
    paired_ttest,
    mcnemar_test,
    bootstrap_significance,
    wilcoxon_signed_rank,
    choose_test,
)
from spark_llm_eval.statistics.effect_size import (
    cohens_d,
    hedges_g,
    odds_ratio,
    relative_improvement,
)


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_basic_bootstrap(self):
        values = np.array([0.7, 0.8, 0.75, 0.72, 0.78, 0.76])
        point, ci, se = bootstrap_ci(values, confidence_level=0.95)

        assert ci[0] <= point <= ci[1]
        assert se > 0

    def test_ci_contains_true_mean(self):
        # generate data from known distribution
        np.random.seed(42)
        true_mean = 0.75
        values = np.random.normal(true_mean, 0.1, 100)

        point, ci, se = bootstrap_ci(values, confidence_level=0.95, n_iterations=2000)

        # CI should contain true mean (with high probability)
        assert ci[0] < true_mean < ci[1]

    def test_wider_ci_for_higher_confidence(self):
        values = np.array([0.6, 0.7, 0.8, 0.65, 0.75])

        _, ci_90, _ = bootstrap_ci(values, confidence_level=0.90)
        _, ci_99, _ = bootstrap_ci(values, confidence_level=0.99)

        width_90 = ci_90[1] - ci_90[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_99 > width_90

    def test_single_value(self):
        values = np.array([0.5])
        point, ci, se = bootstrap_ci(values)

        assert point == 0.5
        assert ci == (0.5, 0.5)
        assert se == 0.0

    def test_empty_array(self):
        values = np.array([])
        point, ci, se = bootstrap_ci(values)

        assert point == 0.0
        assert ci == (0.0, 0.0)


class TestBootstrapBCA:
    """Tests for BCa bootstrap."""

    def test_bca_basic(self):
        values = np.array([0.7, 0.8, 0.75, 0.72, 0.78, 0.76, 0.73])
        point, ci, se = bootstrap_ci_bca(values, confidence_level=0.95)

        assert ci[0] <= point <= ci[1]


class TestAnalyticalCIMean:
    """Tests for analytical CI for mean."""

    def test_basic(self):
        values = np.array([0.7, 0.8, 0.75, 0.72, 0.78])
        mean, ci, se = analytical_ci_mean(values, confidence_level=0.95)

        assert ci[0] < mean < ci[1]
        assert se > 0

    def test_known_values(self):
        # with known values we can check the math
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, ci, se = analytical_ci_mean(values, confidence_level=0.95)

        assert mean == pytest.approx(3.0)
        assert se == pytest.approx(np.std(values, ddof=1) / np.sqrt(5))


class TestAnalyticalCIProportion:
    """Tests for proportion CI."""

    def test_wilson_interval(self):
        p, ci, se = analytical_ci_proportion(70, 100, confidence_level=0.95)

        assert p == 0.7
        assert ci[0] < 0.7 < ci[1]
        assert 0 <= ci[0]
        assert ci[1] <= 1

    def test_extreme_proportions(self):
        # all successes
        p, ci, se = analytical_ci_proportion(100, 100, confidence_level=0.95)
        assert p == 1.0
        assert ci[1] == 1.0

        # no successes
        p, ci, se = analytical_ci_proportion(0, 100, confidence_level=0.95)
        assert p == 0.0
        assert ci[0] == pytest.approx(0.0, abs=1e-10)  # floating point tolerance

    def test_different_methods(self):
        successes, total = 60, 100

        _, ci_wilson, _ = analytical_ci_proportion(successes, total, method="wilson")
        _, ci_normal, _ = analytical_ci_proportion(successes, total, method="normal")

        # both should give similar results for moderate proportions
        assert abs(ci_wilson[0] - ci_normal[0]) < 0.05


class TestCompareCIs:
    """Tests for CI comparison."""

    def test_overlapping_cis(self):
        result = compare_cis((0.6, 0.8), (0.7, 0.9))
        assert result["overlaps"] is True

    def test_non_overlapping_cis(self):
        result = compare_cis((0.6, 0.7), (0.8, 0.9))
        assert result["overlaps"] is False
        assert result["likely_different"] is True


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_identical_values(self):
        values = [0.7, 0.8, 0.75, 0.72]
        result = paired_ttest(values, values)

        assert result.p_value == pytest.approx(1.0)
        assert not result.is_significant

    def test_different_values(self):
        a = np.array([0.8, 0.85, 0.9, 0.82, 0.88, 0.91])
        b = np.array([0.6, 0.65, 0.7, 0.62, 0.68, 0.71])

        result = paired_ttest(a, b)

        assert result.p_value < 0.05
        assert result.is_significant
        assert result.details["mean_difference"] > 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_ttest([0.7, 0.8], [0.7])


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_no_difference(self):
        # equal discordant pairs
        a = [True, True, False, False, True]
        b = [True, False, True, False, True]

        result = mcnemar_test(a, b)
        assert not result.is_significant

    def test_clear_difference(self):
        # A much better than B on discordant
        a = [True] * 50 + [False] * 10
        b = [False] * 50 + [True] * 10

        result = mcnemar_test(a, b)
        assert result.is_significant


class TestBootstrapSignificance:
    """Tests for bootstrap permutation test."""

    def test_identical_distributions(self):
        np.random.seed(42)
        a = np.random.normal(0.7, 0.1, 50)
        b = np.random.normal(0.7, 0.1, 50)

        result = bootstrap_significance(a, b, n_iterations=1000)
        # should not be significant
        assert result.p_value > 0.05

    def test_different_distributions(self):
        np.random.seed(42)
        a = np.random.normal(0.8, 0.1, 50)
        b = np.random.normal(0.6, 0.1, 50)

        result = bootstrap_significance(a, b, n_iterations=1000)
        # should be significant
        assert result.p_value < 0.05


class TestWilcoxon:
    """Tests for Wilcoxon signed-rank test."""

    def test_identical_values(self):
        values = [0.7, 0.8, 0.75, 0.72, 0.78]
        result = wilcoxon_signed_rank(values, values)

        # no non-zero differences
        assert not result.is_significant

    def test_different_values(self):
        a = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.87, 0.89]
        b = [0.6, 0.65, 0.7, 0.62, 0.68, 0.71, 0.67, 0.69]

        result = wilcoxon_signed_rank(a, b)
        assert result.is_significant


class TestChooseTest:
    """Tests for test recommendation."""

    def test_binary_paired(self):
        a = [True, False, True]
        b = [True, True, False]
        assert choose_test(a, b, metric_type="binary", paired=True) == "mcnemar"

    def test_continuous_small_sample(self):
        a = list(range(10))
        b = list(range(10))
        result = choose_test(a, b, metric_type="continuous", paired=True)
        assert result == "wilcoxon"


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_no_difference(self):
        values = [0.7, 0.8, 0.75, 0.72, 0.78]
        result = cohens_d(values, values)

        assert result.value == pytest.approx(0.0)
        assert result.interpretation == "negligible"

    def test_large_difference(self):
        a = [0.8, 0.85, 0.9, 0.82, 0.88]
        b = [0.5, 0.55, 0.6, 0.52, 0.58]

        result = cohens_d(a, b)

        assert abs(result.value) > 0.8
        assert result.interpretation == "large"

    def test_ci_contains_point(self):
        a = [0.7, 0.75, 0.8, 0.72, 0.78]
        b = [0.6, 0.65, 0.7, 0.62, 0.68]

        result = cohens_d(a, b)

        assert result.ci[0] <= result.value <= result.ci[1]


class TestHedgesG:
    """Tests for Hedges' g."""

    def test_correction_applied(self):
        a = [0.7, 0.75, 0.8, 0.72, 0.78]
        b = [0.6, 0.65, 0.7, 0.62, 0.68]

        d_result = cohens_d(a, b, paired=False)
        g_result = hedges_g(a, b)

        # Hedges g should be slightly smaller due to correction
        assert abs(g_result.value) < abs(d_result.value)
        assert g_result.details["cohens_d"] == pytest.approx(d_result.value)


class TestOddsRatio:
    """Tests for odds ratio."""

    def test_equal_discordant(self):
        a = [True, True, False, False]
        b = [True, False, True, False]

        result = odds_ratio(a, b)
        assert result.value == pytest.approx(1.0)

    def test_a_better(self):
        # A correct when B wrong, more often than vice versa
        a = [True] * 20 + [False] * 5 + [True] * 25 + [False] * 50
        b = [False] * 20 + [True] * 5 + [True] * 25 + [False] * 50

        result = odds_ratio(a, b)
        assert result.value > 1


class TestRelativeImprovement:
    """Tests for relative improvement."""

    def test_improvement(self):
        result = relative_improvement(0.8, 0.7, baseline_is_b=True)
        # (0.8 - 0.7) / 0.7 * 100 ≈ 14.3%
        assert result.value == pytest.approx(14.286, rel=0.01)

    def test_decline(self):
        result = relative_improvement(0.6, 0.7, baseline_is_b=True)
        # (0.6 - 0.7) / 0.7 * 100 ≈ -14.3%
        assert result.value == pytest.approx(-14.286, rel=0.01)

    def test_no_change(self):
        result = relative_improvement(0.7, 0.7, baseline_is_b=True)
        assert result.value == pytest.approx(0.0)
