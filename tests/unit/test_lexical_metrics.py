"""Tests for lexical metrics."""

import pytest
from spark_llm_eval.evaluation.lexical import (
    ExactMatchMetric,
    F1Metric,
    ContainsMetric,
    BLEUMetric,
    ROUGELMetric,
    LengthRatioMetric,
    normalize_text,
    tokenize,
)
from spark_llm_eval.evaluation.base import get_metric, list_metrics


class TestNormalization:
    """Tests for text normalization utilities."""

    def test_normalize_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_normalize_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_normalize_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_tokenize(self):
        tokens = tokenize("Hello, World!")
        assert tokens == ["hello", "world"]


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    def test_exact_match(self):
        metric = ExactMatchMetric()
        result = metric.compute(
            predictions=["hello world", "foo bar"],
            references=["hello world", "foo bar"],
        )
        assert result.value == 1.0
        assert result.per_example_scores == [1.0, 1.0]

    def test_no_match(self):
        metric = ExactMatchMetric()
        result = metric.compute(
            predictions=["hello world", "foo bar"],
            references=["goodbye world", "baz qux"],
        )
        assert result.value == 0.0

    def test_partial_match(self):
        metric = ExactMatchMetric()
        result = metric.compute(
            predictions=["hello world", "foo bar"],
            references=["hello world", "baz qux"],
        )
        assert result.value == 0.5
        assert result.per_example_scores == [1.0, 0.0]

    def test_normalization(self):
        metric = ExactMatchMetric(normalize=True)
        result = metric.compute(
            predictions=["Hello, World!"],
            references=["hello world"],
        )
        assert result.value == 1.0

    def test_case_sensitive(self):
        metric = ExactMatchMetric(normalize=False, case_sensitive=True)
        result = metric.compute(
            predictions=["Hello"],
            references=["hello"],
        )
        assert result.value == 0.0

    def test_case_insensitive(self):
        metric = ExactMatchMetric(normalize=False, case_sensitive=False)
        result = metric.compute(
            predictions=["Hello"],
            references=["hello"],
        )
        assert result.value == 1.0


class TestF1Metric:
    """Tests for F1Metric."""

    def test_perfect_match(self):
        metric = F1Metric()
        result = metric.compute(
            predictions=["the quick brown fox"],
            references=["the quick brown fox"],
        )
        assert result.value == 1.0

    def test_no_overlap(self):
        metric = F1Metric()
        result = metric.compute(
            predictions=["hello world"],
            references=["foo bar"],
        )
        assert result.value == 0.0

    def test_partial_overlap(self):
        metric = F1Metric()
        result = metric.compute(
            predictions=["the quick brown fox"],
            references=["the slow brown dog"],
        )
        # overlap: "the", "brown" (2 tokens)
        # precision: 2/4 = 0.5, recall: 2/4 = 0.5, f1 = 0.5
        assert result.value == pytest.approx(0.5)

    def test_empty_prediction(self):
        metric = F1Metric()
        result = metric.compute(
            predictions=[""],
            references=["hello world"],
        )
        assert result.value == 0.0

    def test_both_empty(self):
        metric = F1Metric()
        result = metric.compute(
            predictions=[""],
            references=[""],
        )
        assert result.value == 1.0

    def test_metadata_has_precision_recall(self):
        metric = F1Metric()
        result = metric.compute(
            predictions=["a b c d"],
            references=["a b e f"],
        )
        assert "avg_precision" in result.metadata
        assert "avg_recall" in result.metadata


class TestContainsMetric:
    """Tests for ContainsMetric."""

    def test_contains(self):
        metric = ContainsMetric()
        result = metric.compute(
            predictions=["the answer is 42"],
            references=["42"],
        )
        assert result.value == 1.0

    def test_not_contains(self):
        metric = ContainsMetric()
        result = metric.compute(
            predictions=["the answer is 42"],
            references=["43"],
        )
        assert result.value == 0.0

    def test_contains_with_normalization(self):
        metric = ContainsMetric(normalize=True)
        result = metric.compute(
            predictions=["The Answer Is 42!"],
            references=["answer is"],
        )
        assert result.value == 1.0


class TestBLEUMetric:
    """Tests for BLEUMetric."""

    def test_perfect_match(self):
        metric = BLEUMetric()
        result = metric.compute(
            predictions=["the cat sat on the mat"],
            references=["the cat sat on the mat"],
        )
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        metric = BLEUMetric()
        result = metric.compute(
            predictions=["hello world"],
            references=["foo bar baz"],
        )
        assert result.value < 0.1

    def test_partial_overlap(self):
        metric = BLEUMetric()
        result = metric.compute(
            predictions=["the cat sat on the mat"],
            references=["the dog sat on the rug"],
        )
        # some n-gram overlap
        assert 0.1 < result.value < 0.9

    def test_empty_prediction(self):
        metric = BLEUMetric()
        result = metric.compute(
            predictions=[""],
            references=["hello world"],
        )
        assert result.value == 0.0


class TestROUGELMetric:
    """Tests for ROUGELMetric."""

    def test_perfect_match(self):
        metric = ROUGELMetric()
        result = metric.compute(
            predictions=["the cat sat on the mat"],
            references=["the cat sat on the mat"],
        )
        assert result.value == 1.0

    def test_no_overlap(self):
        metric = ROUGELMetric()
        result = metric.compute(
            predictions=["hello world"],
            references=["foo bar"],
        )
        assert result.value == 0.0

    def test_subsequence(self):
        metric = ROUGELMetric()
        result = metric.compute(
            predictions=["a b c d e"],
            references=["a c e"],
        )
        # LCS = "a c e" (length 3)
        # precision = 3/5, recall = 3/3 = 1.0
        # f1 = 2 * 0.6 * 1.0 / 1.6 = 0.75
        assert result.value == pytest.approx(0.75)


class TestLengthRatioMetric:
    """Tests for LengthRatioMetric."""

    def test_same_length(self):
        metric = LengthRatioMetric()
        result = metric.compute(
            predictions=["hello world"],
            references=["hello world"],
        )
        assert result.value == pytest.approx(1.0)

    def test_double_length(self):
        metric = LengthRatioMetric()
        result = metric.compute(
            predictions=["a b c d"],
            references=["a b"],
        )
        assert result.value == pytest.approx(2.0)

    def test_half_length(self):
        metric = LengthRatioMetric()
        result = metric.compute(
            predictions=["a b"],
            references=["a b c d"],
        )
        assert result.value == pytest.approx(0.5)


class TestMetricRegistry:
    """Tests for metric registry."""

    def test_get_metric(self):
        metric = get_metric("exact_match")
        assert isinstance(metric, ExactMatchMetric)

    def test_get_metric_with_kwargs(self):
        metric = get_metric("exact_match", normalize=False)
        assert metric.normalize is False

    def test_list_metrics(self):
        metrics = list_metrics()
        assert "exact_match" in metrics
        assert "f1" in metrics
        assert "bleu" in metrics
        assert "rouge_l" in metrics

    def test_get_unknown_metric(self):
        with pytest.raises(KeyError, match="Unknown metric"):
            get_metric("nonexistent_metric")


class TestValidation:
    """Tests for input validation."""

    def test_mismatched_lengths(self):
        metric = ExactMatchMetric()
        with pytest.raises(ValueError, match="same length"):
            metric.compute(
                predictions=["a", "b"],
                references=["a"],
            )

    def test_empty_inputs(self):
        metric = ExactMatchMetric()
        with pytest.raises(ValueError, match="empty"):
            metric.compute(predictions=[], references=[])
