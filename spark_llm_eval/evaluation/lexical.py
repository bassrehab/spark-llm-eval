"""Lexical metrics - string/token matching without semantic understanding."""

import logging
import string
from collections import Counter

from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric

logger = logging.getLogger(__name__)


def normalize_text(text: str | None) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def tokenize(text):
    # just whitespace split after normalizing
    return normalize_text(text).split()


@register_metric
class ExactMatchMetric(Metric):
    """1.0 if exact match, 0.0 otherwise."""

    name = "exact_match"

    def __init__(self, normalize: bool = True, case_sensitive: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            if self.normalize:
                pred = normalize_text(pred)
                ref = normalize_text(ref)
            elif not self.case_sensitive:
                pred = pred.lower()
                ref = ref.lower()

            scores.append(1.0 if pred == ref else 0.0)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
        )


@register_metric
class F1Metric(Metric):
    """Token-level F1 (SQuAD style)."""

    name = "f1"

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        self.validate_inputs(predictions, references)

        scores = []
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            if self.normalize:
                pred_tokens = tokenize(pred)
                ref_tokens = tokenize(ref)
            else:
                pred_tokens = pred.split()
                ref_tokens = ref.split()

            if not pred_tokens and not ref_tokens:
                # both empty, consider it a match
                scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
                continue

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            # counter intersection gives us the overlap
            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())

            if num_common == 0:
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)

            scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={
                "avg_precision": sum(precisions) / len(precisions),
                "avg_recall": sum(recalls) / len(recalls),
            },
        )


@register_metric
class ContainsMetric(Metric):
    """1.0 if reference substring is in prediction."""

    name = "contains"

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            if self.normalize:
                pred = normalize_text(pred)
                ref = normalize_text(ref)

            scores.append(1.0 if ref in pred else 0.0)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
        )


@register_metric
class BLEUMetric(Metric):
    """BLEU score - ngram precision with brevity penalty.
    See the original Papineni et al. paper for details."""

    name = "bleu"

    def __init__(self, max_n: int = 4, smooth: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.smooth = smooth

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            score = self._sentence_bleu(pred, ref)
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
        )

    def _sentence_bleu(self, prediction: str, reference: str) -> float:
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)

        if not pred_tokens:
            return 0.0

        # compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            matches = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())

            if self.smooth:
                # add-1 smoothing
                precision = (matches + 1) / (total + 1)
            else:
                precision = matches / total if total > 0 else 0.0

            precisions.append(precision)

        # geometric mean of precisions
        if all(p > 0 for p in precisions):
            import math

            log_precision = sum(math.log(p) for p in precisions) / len(precisions)
            geo_mean = math.exp(log_precision)
        else:
            geo_mean = 0.0

        # brevity penalty
        bp = self._brevity_penalty(len(pred_tokens), len(ref_tokens))

        return bp * geo_mean

    def _get_ngrams(self, tokens, n):
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)

    def _brevity_penalty(self, pred_len, ref_len):
        if pred_len >= ref_len:
            return 1.0
        if pred_len == 0:
            return 0.0
        import math

        return math.exp(1 - ref_len / pred_len)


@register_metric
class ROUGELMetric(Metric):
    """ROUGE-L using longest common subsequence."""

    name = "rouge_l"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        self.validate_inputs(predictions, references)

        scores = []
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            lcs_len = self._lcs_length(pred_tokens, ref_tokens)

            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(ref_tokens)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={
                "avg_precision": sum(precisions) / len(precisions),
                "avg_recall": sum(recalls) / len(recalls),
            },
        )

    def _lcs_length(self, x, y):
        # classic DP but space optimized - only keep 2 rows
        m, n = len(x), len(y)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]


@register_metric
class LengthRatioMetric(Metric):
    """pred_len / ref_len - useful to catch verbose or terse outputs."""

    name = "length_ratio"

    def __init__(self, use_tokens: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_tokens = use_tokens

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            if self.use_tokens:
                pred_len = len(tokenize(pred))
                ref_len = len(tokenize(ref))
            else:
                pred_len = len(pred)
                ref_len = len(ref)

            if ref_len == 0:
                ratio = 0.0 if pred_len == 0 else float("inf")
            else:
                ratio = pred_len / ref_len

            scores.append(ratio)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
        )
