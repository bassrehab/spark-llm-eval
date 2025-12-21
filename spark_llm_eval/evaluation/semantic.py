"""Semantic similarity metrics for evaluation.

These metrics use embeddings or language models to measure
semantic similarity rather than exact lexical overlap.
"""

import logging

import numpy as np

from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric

logger = logging.getLogger(__name__)

# lazy imports for optional dependencies
_transformers = None
_torch = None
_sentence_transformers = None


def _get_transformers():
    """Lazy import transformers."""
    global _transformers
    if _transformers is None:
        try:
            import transformers

            _transformers = transformers
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")
    return _transformers


def _get_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
    return _torch


def _get_sentence_transformers():
    """Lazy import sentence-transformers."""
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            import sentence_transformers

            _sentence_transformers = sentence_transformers
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformers


@register_metric("bertscore")
class BERTScoreMetric(Metric):
    """BERTScore metric for semantic similarity.

    Uses contextual embeddings from BERT-like models to compute
    similarity between predictions and references.

    Returns precision, recall, and F1 scores.
    """

    name = "bertscore"
    requires_reference = True

    def __init__(
        self,
        model_name: str = "microsoft/deberta-xlarge-mnli",
        batch_size: int = 32,
        device: str | None = None,
        use_fast_tokenizer: bool = True,
    ):
        """Initialize BERTScore metric.

        Args:
            model_name: HuggingFace model to use for embeddings.
            batch_size: Batch size for encoding.
            device: Device to use (cuda/cpu). Auto-detected if None.
            use_fast_tokenizer: Whether to use fast tokenizer.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fast_tokenizer = use_fast_tokenizer

        torch = _get_torch()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load model and tokenizer lazily."""
        if self._model is not None:
            return

        transformers = _get_transformers()
        torch = _get_torch()

        logger.info(f"Loading BERTScore model: {self.model_name}")

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=self.use_fast_tokenizer,
        )
        self._model = transformers.AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get token embeddings for texts."""
        torch = _get_torch()
        self._load_model()

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # use last hidden state
                embeddings = outputs.last_hidden_state

            # move to cpu and convert to numpy
            all_embeddings.append(embeddings.cpu().numpy())

        return all_embeddings

    def _compute_bertscore(
        self,
        pred_embeddings: np.ndarray,
        ref_embeddings: np.ndarray,
        pred_mask: np.ndarray,
        ref_mask: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute BERTScore P/R/F1 for a single example."""
        # normalize embeddings
        pred_norm = pred_embeddings / (
            np.linalg.norm(pred_embeddings, axis=-1, keepdims=True) + 1e-8
        )
        ref_norm = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=-1, keepdims=True) + 1e-8)

        # compute similarity matrix
        similarity = np.matmul(pred_norm, ref_norm.T)

        # apply masks (ignore padding tokens)
        pred_len = int(pred_mask.sum())
        ref_len = int(ref_mask.sum())

        if pred_len == 0 or ref_len == 0:
            return 0.0, 0.0, 0.0

        similarity = similarity[:pred_len, :ref_len]

        # precision: max similarity for each prediction token
        precision = similarity.max(axis=1).mean()

        # recall: max similarity for each reference token
        recall = similarity.max(axis=0).mean()

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return float(precision), float(recall), float(f1)

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        """Compute BERTScore for predictions vs references.

        Args:
            predictions: Model predictions.
            references: Ground truth references.

        Returns:
            MetricResult with F1 as primary score, P/R in details.
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")

        if len(predictions) == 0:
            return MetricResult(
                name=self.name,
                value=0.0,
                per_example_scores=[],
                metadata={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            )

        torch = _get_torch()
        self._load_model()

        precisions = []
        recalls = []
        f1_scores = []

        # process in batches
        for i in range(0, len(predictions), self.batch_size):
            batch_preds = predictions[i : i + self.batch_size]
            batch_refs = references[i : i + self.batch_size]

            # tokenize
            pred_inputs = self._tokenizer(
                batch_preds,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            ref_inputs = self._tokenizer(
                batch_refs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                pred_outputs = self._model(**pred_inputs)
                ref_outputs = self._model(**ref_inputs)

            pred_embeddings = pred_outputs.last_hidden_state.cpu().numpy()
            ref_embeddings = ref_outputs.last_hidden_state.cpu().numpy()
            pred_mask = pred_inputs.attention_mask.cpu().numpy()
            ref_mask = ref_inputs.attention_mask.cpu().numpy()

            # compute scores for each example in batch
            for j in range(len(batch_preds)):
                p, r, f = self._compute_bertscore(
                    pred_embeddings[j],
                    ref_embeddings[j],
                    pred_mask[j],
                    ref_mask[j],
                )
                precisions.append(p)
                recalls.append(r)
                f1_scores.append(f)

        avg_precision = float(np.mean(precisions))
        avg_recall = float(np.mean(recalls))
        avg_f1 = float(np.mean(f1_scores))

        return MetricResult(
            name=self.name,
            value=avg_f1,  # F1 as primary metric
            per_example_scores=f1_scores,
            metadata={
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "per_example_precision": precisions,
                "per_example_recall": recalls,
            },
        )


@register_metric("embedding_similarity")
class EmbeddingSimilarityMetric(Metric):
    """Cosine similarity using sentence embeddings.

    Uses sentence-transformers for efficient sentence embeddings,
    then computes cosine similarity between predictions and references.
    """

    name = "embedding_similarity"
    requires_reference = True

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ):
        """Initialize embedding similarity metric.

        Args:
            model_name: Sentence-transformers model name.
            batch_size: Batch size for encoding.
            device: Device to use.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        """Load sentence transformer model lazily."""
        if self._model is not None:
            return

        st = _get_sentence_transformers()
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = st.SentenceTransformer(self.model_name, device=self.device)

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        """Compute cosine similarity between predictions and references.

        Args:
            predictions: Model predictions.
            references: Ground truth references.

        Returns:
            MetricResult with average cosine similarity.
        """
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have same length")

        if len(predictions) == 0:
            return MetricResult(
                name=self.name,
                value=0.0,
                per_example_scores=[],
            )

        self._load_model()

        # encode predictions and references
        pred_embeddings = self._model.encode(
            predictions,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ref_embeddings = self._model.encode(
            references,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # compute cosine similarity for each pair
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            # normalize
            pred_norm = pred_emb / (np.linalg.norm(pred_emb) + 1e-8)
            ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)
            # cosine similarity
            sim = float(np.dot(pred_norm, ref_norm))
            similarities.append(sim)

        avg_similarity = float(np.mean(similarities))

        return MetricResult(
            name=self.name,
            value=avg_similarity,
            per_example_scores=similarities,
            metadata={
                "model": self.model_name,
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
            },
        )


@register_metric("semantic_similarity")
class SemanticSimilarityMetric(EmbeddingSimilarityMetric):
    """Alias for embedding_similarity for convenience."""

    name = "semantic_similarity"
