"""RAG (Retrieval-Augmented Generation) evaluation metrics.

This module provides metrics for evaluating RAG systems:
- Context relevance: Is retrieved context relevant to the query?
- Faithfulness: Is the answer grounded in the context?
- Answer relevance: Does the answer address the query?
- Context precision: Ranking quality of retrieved chunks
- Context recall: Coverage of ground truth in retrieved context

Both LLM-as-judge and embedding-based variants are provided.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from spark_llm_eval.core.config import ModelConfig
from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RAGJudgeConfig:
    """Configuration for RAG LLM judge metrics.

    Args:
        model_config: Model to use as judge.
        temperature: Sampling temperature (0 for deterministic).
        max_retries: Number of retries on failure.
    """

    model_config: ModelConfig
    temperature: float = 0.0
    max_retries: int = 3


# =============================================================================
# Prompt Templates
# =============================================================================

CONTEXT_RELEVANCE_PROMPT = """You are an expert evaluator assessing how relevant retrieved context is to a given query.

Query: {query}

Retrieved Context:
{context}

Task: Evaluate how relevant the retrieved context is for answering the query.

Consider:
- Does the context contain information needed to answer the query?
- Is the context on-topic and related to the query?
- Would this context help in generating a correct answer?

Rate the context relevance on a scale of 1 to 5:
1 - Completely irrelevant, no connection to the query
2 - Mostly irrelevant, only tangentially related
3 - Partially relevant, contains some useful information
4 - Mostly relevant, addresses the query well
5 - Highly relevant, directly addresses the query with comprehensive information

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>"}}"""


FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an answer is faithful to the given context (no hallucinations).

Context:
{context}

Answer to evaluate:
{answer}

Task: Determine if EVERY claim in the answer can be supported by the context.

For each statement in the answer:
1. Identify the claim being made
2. Check if the context supports this claim
3. Flag any claims not supported by the context

Rate the faithfulness on a scale of 1 to 5:
1 - Completely unfaithful, major hallucinations
2 - Mostly unfaithful, significant unsupported claims
3 - Partially faithful, some unsupported claims
4 - Mostly faithful, minor unsupported details
5 - Completely faithful, all claims supported by context

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "unsupported_claims": ["<claim1>", "<claim2>"]}}"""


ANSWER_RELEVANCE_PROMPT = """You are an expert evaluator assessing how well an answer addresses a given query.

Query: {query}

Answer:
{answer}

Task: Evaluate how well the answer addresses the query.

Consider:
- Does the answer directly address the question asked?
- Is the answer complete and comprehensive?
- Is the answer concise without unnecessary information?

Rate the answer relevance on a scale of 1 to 5:
1 - Completely irrelevant, does not address the query
2 - Mostly irrelevant, only tangentially addresses the query
3 - Partially relevant, addresses some aspects of the query
4 - Mostly relevant, addresses the query well with minor gaps
5 - Highly relevant, fully addresses the query

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>"}}"""


CONTEXT_PRECISION_PROMPT = """You are an expert evaluator assessing the precision of retrieved context chunks.

Query: {query}

Retrieved Context Chunks:
{context}

Ground Truth Answer: {reference}

Task: For each chunk, determine if it is relevant to answering the query correctly.
A chunk is relevant if it contains information that helps produce the ground truth answer.

Evaluate each chunk and identify which ones are truly relevant.

Respond in JSON format:
{{"relevant_chunks": [<indices of relevant chunks, 0-indexed>], "total_chunks": <number>, "reasoning": "<explanation>"}}"""


CONTEXT_RECALL_PROMPT = """You are an expert evaluator assessing the recall of retrieved context.

Query: {query}

Retrieved Context:
{context}

Ground Truth Answer: {reference}

Task: Determine what fraction of the ground truth answer can be attributed to the retrieved context.

Break down the ground truth answer into key claims/facts, then check which ones are supported by the context.

Respond in JSON format:
{{"supported_claims": ["<claim1>", "<claim2>"], "unsupported_claims": ["<claim3>"], "recall_score": <0.0-1.0>, "reasoning": "<explanation>"}}"""


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_score_response(response: str, scale: tuple = (1, 5)) -> tuple[int, dict]:
    """Parse JSON response with score."""
    try:
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = int(data.get("score", scale[0]))
            score = max(scale[0], min(scale[1], score))
            return score, data
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract number
    numbers = re.findall(r"\b(\d+)\b", response)
    for num_str in numbers:
        num = int(num_str)
        if scale[0] <= num <= scale[1]:
            return num, {"reasoning": response}

    return (scale[0] + scale[1]) // 2, {"reasoning": response}


# =============================================================================
# RAG Metric Base Class
# =============================================================================


class RAGMetric(Metric):
    """Base class for RAG evaluation metrics.

    RAG metrics evaluate retrieval-augmented generation systems by
    measuring the quality of retrieved context and generated answers.

    Subclasses must implement compute() and accept these kwargs:
    - queries: List of user queries/questions
    - contexts: List of retrieved contexts (can be list of lists for multiple chunks)
    - predictions: List of generated answers
    - references: List of ground truth answers (optional for some metrics)
    """

    requires_query: bool = True
    requires_context: bool = True
    requires_reference: bool = False

    def validate_rag_inputs(
        self,
        predictions: list[str],
        queries: list[str] | None = None,
        contexts: list[Any] | None = None,
        references: list[str] | None = None,
    ) -> None:
        """Validate RAG-specific inputs."""
        n = len(predictions)

        if self.requires_query:
            if queries is None:
                raise ValueError(f"{self.name} requires 'queries' parameter")
            if len(queries) != n:
                raise ValueError(f"queries length ({len(queries)}) != predictions ({n})")

        if self.requires_context:
            if contexts is None:
                raise ValueError(f"{self.name} requires 'contexts' parameter")
            if len(contexts) != n:
                raise ValueError(f"contexts length ({len(contexts)}) != predictions ({n})")

        if self.requires_reference and references is not None:
            if len(references) != n:
                raise ValueError(f"references length ({len(references)}) != predictions ({n})")

    @staticmethod
    def format_context(context: Any) -> str:
        """Format context (single string or list of chunks) into single string."""
        if isinstance(context, list):
            return "\n\n".join(f"[Chunk {i + 1}]: {c}" for i, c in enumerate(context))
        return str(context)

    @staticmethod
    def split_into_statements(text: str) -> list[str]:
        """Split text into individual statements/claims for analysis."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# LLM-as-Judge RAG Metrics
# =============================================================================


@register_metric("context_relevance")
class ContextRelevanceMetric(RAGMetric):
    """Measures how relevant retrieved context is to the query.

    Uses LLM-as-judge to evaluate if the context contains information
    needed to answer the query.
    """

    name = "context_relevance"
    requires_query = True
    requires_context = True
    requires_reference = False

    def __init__(
        self,
        judge_config: RAGJudgeConfig,
        prompt_template: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_config = judge_config
        self.prompt_template = prompt_template or CONTEXT_RELEVANCE_PROMPT
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from spark_llm_eval.inference import create_engine

            self._engine = create_engine(self.judge_config.model_config)
            self._engine.initialize()
        return self._engine

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        queries: list[str] | None = None,
        contexts: list[Any] | None = None,
        **kwargs,
    ) -> MetricResult:
        self.validate_rag_inputs(predictions, queries, contexts, references)

        engine = self._get_engine()
        scores = []
        metadata_list = []

        for query, context in zip(queries, contexts):
            formatted_context = self.format_context(context)
            prompt = self.prompt_template.format(
                query=query,
                context=formatted_context,
            )

            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=256,
                temperature=self.judge_config.temperature,
            )

            try:
                response = engine.infer(request)
                score, meta = _parse_score_response(response.text)
            except Exception as e:
                logger.warning(f"Context relevance failed: {e}")
                score, meta = 3, {"error": str(e)}

            scores.append(score)
            metadata_list.append(meta)

        # Normalize to 0-1
        normalized = [(s - 1) / 4 for s in scores]

        return MetricResult(
            name=self.name,
            value=sum(normalized) / len(normalized),
            per_example_scores=normalized,
            metadata={
                "raw_scores": scores,
                "details": metadata_list,
            },
        )


@register_metric("faithfulness")
class FaithfulnessMetric(RAGMetric):
    """Measures if the answer is grounded in the context (no hallucinations).

    Uses LLM-as-judge to verify each claim in the answer can be
    attributed to the provided context.
    """

    name = "faithfulness"
    requires_query = False
    requires_context = True
    requires_reference = False

    def __init__(
        self,
        judge_config: RAGJudgeConfig,
        prompt_template: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_config = judge_config
        self.prompt_template = prompt_template or FAITHFULNESS_PROMPT
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from spark_llm_eval.inference import create_engine

            self._engine = create_engine(self.judge_config.model_config)
            self._engine.initialize()
        return self._engine

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        contexts: list[Any] | None = None,
        **kwargs,
    ) -> MetricResult:
        self.validate_rag_inputs(predictions, contexts=contexts)

        engine = self._get_engine()
        scores = []
        metadata_list = []

        for answer, context in zip(predictions, contexts):
            formatted_context = self.format_context(context)
            prompt = self.prompt_template.format(
                context=formatted_context,
                answer=answer,
            )

            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=512,
                temperature=self.judge_config.temperature,
            )

            try:
                response = engine.infer(request)
                score, meta = _parse_score_response(response.text)
            except Exception as e:
                logger.warning(f"Faithfulness check failed: {e}")
                score, meta = 3, {"error": str(e)}

            scores.append(score)
            metadata_list.append(meta)

        normalized = [(s - 1) / 4 for s in scores]

        return MetricResult(
            name=self.name,
            value=sum(normalized) / len(normalized),
            per_example_scores=normalized,
            metadata={
                "raw_scores": scores,
                "details": metadata_list,
            },
        )


@register_metric("answer_relevance")
class AnswerRelevanceMetric(RAGMetric):
    """Measures how well the answer addresses the query."""

    name = "answer_relevance"
    requires_query = True
    requires_context = False
    requires_reference = False

    def __init__(
        self,
        judge_config: RAGJudgeConfig,
        prompt_template: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_config = judge_config
        self.prompt_template = prompt_template or ANSWER_RELEVANCE_PROMPT
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from spark_llm_eval.inference import create_engine

            self._engine = create_engine(self.judge_config.model_config)
            self._engine.initialize()
        return self._engine

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        queries: list[str] | None = None,
        **kwargs,
    ) -> MetricResult:
        self.validate_rag_inputs(predictions, queries=queries, contexts=predictions)

        engine = self._get_engine()
        scores = []
        metadata_list = []

        for query, answer in zip(queries, predictions):
            prompt = self.prompt_template.format(query=query, answer=answer)

            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=256,
                temperature=self.judge_config.temperature,
            )

            try:
                response = engine.infer(request)
                score, meta = _parse_score_response(response.text)
            except Exception as e:
                logger.warning(f"Answer relevance failed: {e}")
                score, meta = 3, {"error": str(e)}

            scores.append(score)
            metadata_list.append(meta)

        normalized = [(s - 1) / 4 for s in scores]

        return MetricResult(
            name=self.name,
            value=sum(normalized) / len(normalized),
            per_example_scores=normalized,
            metadata={"raw_scores": scores, "details": metadata_list},
        )


@register_metric("context_precision")
class ContextPrecisionMetric(RAGMetric):
    """Measures ranking quality - are relevant chunks ranked higher?

    Requires ground truth reference to determine chunk relevance.
    """

    name = "context_precision"
    requires_query = True
    requires_context = True
    requires_reference = True

    def __init__(
        self,
        judge_config: RAGJudgeConfig,
        prompt_template: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_config = judge_config
        self.prompt_template = prompt_template or CONTEXT_PRECISION_PROMPT
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from spark_llm_eval.inference import create_engine

            self._engine = create_engine(self.judge_config.model_config)
            self._engine.initialize()
        return self._engine

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        queries: list[str] | None = None,
        contexts: list[Any] | None = None,
        **kwargs,
    ) -> MetricResult:
        self.validate_rag_inputs(predictions, queries, contexts, references)

        engine = self._get_engine()
        scores = []
        metadata_list = []

        for query, context, reference in zip(queries, contexts, references):
            formatted_context = self.format_context(context)
            prompt = self.prompt_template.format(
                query=query,
                context=formatted_context,
                reference=reference,
            )

            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=512,
                temperature=self.judge_config.temperature,
            )

            try:
                response = engine.infer(request)
                json_match = re.search(r"\{[^{}]*\}", response.text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    relevant = data.get("relevant_chunks", [])
                    if isinstance(context, list) and len(context) > 0:
                        precision = len(relevant) / len(context)
                    else:
                        precision = 1.0 if relevant else 0.0
                    scores.append(precision)
                    metadata_list.append(data)
                else:
                    scores.append(0.5)
                    metadata_list.append({"error": "Failed to parse"})
            except Exception as e:
                logger.warning(f"Context precision failed: {e}")
                scores.append(0.5)
                metadata_list.append({"error": str(e)})

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={"details": metadata_list},
        )


@register_metric("context_recall")
class ContextRecallMetric(RAGMetric):
    """Measures coverage of ground truth in retrieved context."""

    name = "context_recall"
    requires_query = True
    requires_context = True
    requires_reference = True

    def __init__(
        self,
        judge_config: RAGJudgeConfig,
        prompt_template: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.judge_config = judge_config
        self.prompt_template = prompt_template or CONTEXT_RECALL_PROMPT
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from spark_llm_eval.inference import create_engine

            self._engine = create_engine(self.judge_config.model_config)
            self._engine.initialize()
        return self._engine

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        queries: list[str] | None = None,
        contexts: list[Any] | None = None,
        **kwargs,
    ) -> MetricResult:
        self.validate_rag_inputs(predictions, queries, contexts, references)

        engine = self._get_engine()
        scores = []
        metadata_list = []

        for query, context, reference in zip(queries, contexts, references):
            formatted_context = self.format_context(context)
            prompt = self.prompt_template.format(
                query=query,
                context=formatted_context,
                reference=reference,
            )

            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=512,
                temperature=self.judge_config.temperature,
            )

            try:
                response = engine.infer(request)
                json_match = re.search(r"\{[^{}]*\}", response.text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    recall = float(data.get("recall_score", 0.5))
                    scores.append(recall)
                    metadata_list.append(data)
                else:
                    scores.append(0.5)
                    metadata_list.append({"error": "Failed to parse"})
            except Exception as e:
                logger.warning(f"Context recall failed: {e}")
                scores.append(0.5)
                metadata_list.append({"error": str(e)})

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={"details": metadata_list},
        )


# =============================================================================
# Embedding-Based RAG Metrics
# =============================================================================


def _get_sentence_transformers():
    """Lazy import sentence-transformers."""
    try:
        import sentence_transformers

        return sentence_transformers
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        )


@register_metric("context_relevance_embedding")
class ContextRelevanceEmbeddingMetric(RAGMetric):
    """Embedding-based context relevance using cosine similarity.

    Faster and cheaper than LLM-based, but less nuanced.
    """

    name = "context_relevance_embedding"
    requires_query = True
    requires_context = True
    requires_reference = False

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        st = _get_sentence_transformers()
        self._model = st.SentenceTransformer(self.model_name, device=self.device)

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        queries: list[str] | None = None,
        contexts: list[Any] | None = None,
        **kwargs,
    ) -> MetricResult:
        import numpy as np

        self.validate_rag_inputs(predictions, queries, contexts)
        self._load_model()

        scores = []

        for query, context in zip(queries, contexts):
            formatted_context = self.format_context(context)

            embeddings = self._model.encode(
                [query, formatted_context],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            query_emb = embeddings[0] / (np.linalg.norm(embeddings[0]) + 1e-8)
            context_emb = embeddings[1] / (np.linalg.norm(embeddings[1]) + 1e-8)
            similarity = float(np.dot(query_emb, context_emb))

            score = (similarity + 1) / 2
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={"model": self.model_name},
        )


@register_metric("answer_relevance_embedding")
class AnswerRelevanceEmbeddingMetric(RAGMetric):
    """Embedding-based answer relevance using cosine similarity."""

    name = "answer_relevance_embedding"
    requires_query = True
    requires_context = False
    requires_reference = False

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        st = _get_sentence_transformers()
        self._model = st.SentenceTransformer(self.model_name, device=self.device)

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        queries: list[str] | None = None,
        **kwargs,
    ) -> MetricResult:
        import numpy as np

        self.validate_rag_inputs(predictions, queries=queries, contexts=predictions)
        self._load_model()

        scores = []

        for query, answer in zip(queries, predictions):
            embeddings = self._model.encode(
                [query, answer],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            query_emb = embeddings[0] / (np.linalg.norm(embeddings[0]) + 1e-8)
            answer_emb = embeddings[1] / (np.linalg.norm(embeddings[1]) + 1e-8)
            similarity = float(np.dot(query_emb, answer_emb))

            score = (similarity + 1) / 2
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={"model": self.model_name},
        )


@register_metric("faithfulness_nli")
class FaithfulnessNLIMetric(RAGMetric):
    """NLI-based faithfulness using entailment classification.

    Checks if each sentence in the answer is entailed by the context.
    """

    name = "faithfulness_nli"
    requires_query = False
    requires_context = True
    requires_reference = False

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        batch_size: int = 16,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers required for NLI metrics")

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        contexts: list[Any] | None = None,
        **kwargs,
    ) -> MetricResult:
        import numpy as np

        self.validate_rag_inputs(predictions, contexts=contexts)
        self._load_model()

        scores = []
        details = []

        for answer, context in zip(predictions, contexts):
            formatted_context = self.format_context(context)
            statements = self.split_into_statements(answer)

            if not statements:
                scores.append(1.0)
                details.append({"statements": [], "entailed": []})
                continue

            pairs = [(formatted_context, stmt) for stmt in statements]
            nli_scores = self._model.predict(pairs)

            if isinstance(nli_scores, np.ndarray) and len(nli_scores.shape) > 1:
                entailment_scores = nli_scores[:, 2]
                entailed = [s > 0.5 for s in entailment_scores]
            else:
                entailed = [s > 0.5 for s in nli_scores]

            faithfulness = sum(entailed) / len(statements) if statements else 1.0
            scores.append(faithfulness)
            details.append(
                {
                    "statements": statements,
                    "entailed": entailed,
                }
            )

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores),
            per_example_scores=scores,
            metadata={
                "model": self.model_name,
                "details": details,
            },
        )
