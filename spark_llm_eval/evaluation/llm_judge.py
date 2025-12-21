"""LLM-as-judge evaluation metrics.

Uses a language model to evaluate the quality of responses
based on custom rubrics and criteria.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from spark_llm_eval.core.config import ModelConfig
from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric

logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    """Configuration for LLM judge.

    Args:
        model_config: Model to use as judge.
        rubric: Evaluation criteria/rubric text.
        scale: Rating scale (e.g., 1-5, 1-10).
        criteria: List of specific criteria to evaluate.
        include_reasoning: Whether to request reasoning from judge.
        temperature: Sampling temperature for judge.
    """

    model_config: ModelConfig
    rubric: str
    scale: tuple[int, int] = (1, 5)
    criteria: list[str] = field(default_factory=list)
    include_reasoning: bool = True
    temperature: float = 0.0


# default prompt templates
DEFAULT_JUDGE_PROMPT = """You are an expert evaluator. Evaluate the following response based on the given criteria.

{rubric}

Input/Question: {input}

Response to evaluate: {prediction}

{reference_section}

Rate the response on a scale of {scale_min} to {scale_max}.

Provide your evaluation in the following JSON format:
{{
    "score": <integer between {scale_min} and {scale_max}>,
    "reasoning": "<brief explanation of your rating>"
}}

Your evaluation:"""

DEFAULT_PAIRWISE_PROMPT = """You are an expert evaluator. Compare the following two responses and determine which one is better.

{rubric}

Input/Question: {input}

Response A: {prediction_a}

Response B: {prediction_b}

{reference_section}

Which response is better? Respond with:
- "A" if Response A is better
- "B" if Response B is better
- "tie" if they are equally good

Provide your evaluation in the following JSON format:
{{
    "winner": "<A, B, or tie>",
    "reasoning": "<brief explanation of your choice>"
}}

Your evaluation:"""


def _parse_judge_response(response: str, scale: tuple[int, int]) -> tuple[int, str]:
    """Parse JSON response from judge.

    Returns:
        Tuple of (score, reasoning).
    """
    # try to extract JSON
    try:
        # find JSON in response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = int(data.get("score", scale[0]))
            reasoning = data.get("reasoning", "")
            # clamp score to valid range
            score = max(scale[0], min(scale[1], score))
            return score, reasoning
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # fallback: try to extract just the number
    numbers = re.findall(r"\b(\d+)\b", response)
    for num_str in numbers:
        num = int(num_str)
        if scale[0] <= num <= scale[1]:
            return num, response

    # default to middle of scale
    return (scale[0] + scale[1]) // 2, response


def _parse_pairwise_response(response: str) -> tuple[str, str]:
    """Parse pairwise comparison response.

    Returns:
        Tuple of (winner: A/B/tie, reasoning).
    """
    try:
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            winner = data.get("winner", "tie").upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE"
            reasoning = data.get("reasoning", "")
            return winner, reasoning
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # fallback: look for A or B in response
    response_upper = response.upper()
    if "RESPONSE A" in response_upper and "BETTER" in response_upper:
        return "A", response
    if "RESPONSE B" in response_upper and "BETTER" in response_upper:
        return "B", response
    if " A " in response_upper or response_upper.startswith("A"):
        return "A", response
    if " B " in response_upper or response_upper.startswith("B"):
        return "B", response

    return "TIE", response


@register_metric("llm_judge")
class LLMJudgeMetric(Metric):
    """LLM-as-judge evaluation metric.

    Uses a language model to evaluate response quality
    based on custom rubrics.
    """

    name = "llm_judge"
    requires_reference = False  # can work without reference

    def __init__(
        self,
        judge_config: JudgeConfig,
        prompt_template: str | None = None,
        input_column: str = "input",
    ):
        """Initialize LLM judge metric.

        Args:
            judge_config: Configuration for the judge.
            prompt_template: Custom prompt template.
            input_column: Name of input column in data.
        """
        self.judge_config = judge_config
        self.prompt_template = prompt_template or DEFAULT_JUDGE_PROMPT
        self.input_column = input_column
        self._inference_engine = None

    def _get_engine(self):
        """Get or create inference engine."""
        if self._inference_engine is not None:
            return self._inference_engine

        from spark_llm_eval.inference import create_engine

        self._inference_engine = create_engine(self.judge_config.model_config)
        self._inference_engine.initialize()
        return self._inference_engine

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        inputs: list[str] | None = None,
        **kwargs,
    ) -> MetricResult:
        """Compute LLM judge scores for predictions.

        Args:
            predictions: Model predictions to evaluate.
            references: Optional ground truth references.
            inputs: Original inputs/questions.

        Returns:
            MetricResult with normalized scores (0-1).
        """
        if inputs is None:
            inputs = [""] * len(predictions)

        if references is None:
            references = [None] * len(predictions)

        engine = self._get_engine()
        config = self.judge_config
        scale = config.scale

        scores = []
        reasonings = []

        for pred, ref, inp in zip(predictions, references, inputs):
            # build reference section
            if ref:
                ref_section = f"Reference/Expected answer: {ref}"
            else:
                ref_section = ""

            # format prompt
            prompt = self.prompt_template.format(
                rubric=config.rubric,
                input=inp,
                prediction=pred,
                reference_section=ref_section,
                scale_min=scale[0],
                scale_max=scale[1],
            )

            # call judge model
            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=256,
                temperature=config.temperature,
            )

            try:
                response = engine.infer(request)
                score, reasoning = _parse_judge_response(response.text, scale)
            except Exception as e:
                logger.warning(f"Judge inference failed: {e}")
                score = (scale[0] + scale[1]) // 2
                reasoning = f"Error: {e}"

            scores.append(score)
            reasonings.append(reasoning)

        # normalize scores to 0-1 range
        normalized_scores = [(s - scale[0]) / (scale[1] - scale[0]) for s in scores]

        avg_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0

        return MetricResult(
            name=self.name,
            value=avg_score,
            per_example_scores=normalized_scores,
            metadata={
                "raw_scores": scores,
                "reasonings": reasonings,
                "scale": scale,
                "rubric": config.rubric,
            },
        )


@register_metric("pairwise_judge")
class PairwiseJudgeMetric(Metric):
    """Pairwise comparison using LLM judge.

    Compares two responses and determines which is better.
    Useful for A/B testing models.
    """

    name = "pairwise_judge"
    requires_reference = False

    def __init__(
        self,
        judge_config: JudgeConfig,
        prompt_template: str | None = None,
    ):
        """Initialize pairwise judge.

        Args:
            judge_config: Configuration for the judge.
            prompt_template: Custom prompt template.
        """
        self.judge_config = judge_config
        self.prompt_template = prompt_template or DEFAULT_PAIRWISE_PROMPT
        self._inference_engine = None

    def _get_engine(self):
        """Get or create inference engine."""
        if self._inference_engine is not None:
            return self._inference_engine

        from spark_llm_eval.inference import create_engine

        self._inference_engine = create_engine(self.judge_config.model_config)
        self._inference_engine.initialize()
        return self._inference_engine

    def compute(
        self,
        predictions_a: list[str],
        predictions_b: list[str],
        inputs: list[str] | None = None,
        references: list[str] | None = None,
        **kwargs,
    ) -> MetricResult:
        """Compare two sets of predictions.

        Args:
            predictions_a: Predictions from model A.
            predictions_b: Predictions from model B.
            inputs: Original inputs.
            references: Optional references.

        Returns:
            MetricResult with win rate for A (0-1).
        """
        if len(predictions_a) != len(predictions_b):
            raise ValueError("prediction lists must have same length")

        if inputs is None:
            inputs = [""] * len(predictions_a)

        if references is None:
            references = [None] * len(predictions_a)

        engine = self._get_engine()
        config = self.judge_config

        results = []  # 1 = A wins, 0 = B wins, 0.5 = tie
        reasonings = []

        for pred_a, pred_b, inp, ref in zip(predictions_a, predictions_b, inputs, references):
            if ref:
                ref_section = f"Reference/Expected answer: {ref}"
            else:
                ref_section = ""

            prompt = self.prompt_template.format(
                rubric=config.rubric,
                input=inp,
                prediction_a=pred_a,
                prediction_b=pred_b,
                reference_section=ref_section,
            )

            from spark_llm_eval.inference.base import InferenceRequest

            request = InferenceRequest(
                prompt=prompt,
                max_tokens=256,
                temperature=config.temperature,
            )

            try:
                response = engine.infer(request)
                winner, reasoning = _parse_pairwise_response(response.text)
            except Exception as e:
                logger.warning(f"Judge inference failed: {e}")
                winner = "TIE"
                reasoning = f"Error: {e}"

            if winner == "A":
                results.append(1.0)
            elif winner == "B":
                results.append(0.0)
            else:
                results.append(0.5)

            reasonings.append(reasoning)

        # win rate for A
        win_rate_a = sum(results) / len(results) if results else 0.5

        a_wins = sum(1 for r in results if r == 1.0)
        b_wins = sum(1 for r in results if r == 0.0)
        ties = sum(1 for r in results if r == 0.5)

        return MetricResult(
            name=self.name,
            value=win_rate_a,
            per_example_scores=results,
            metadata={
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "reasonings": reasonings,
                "rubric": config.rubric,
            },
        )


@register_metric("g_eval")
class GEvalMetric(Metric):
    """G-Eval style evaluation.

    Based on the G-Eval paper, uses chain-of-thought prompting
    for more accurate evaluation.
    """

    name = "g_eval"
    requires_reference = False

    def __init__(
        self,
        judge_config: JudgeConfig,
        criteria_weights: dict[str, float] | None = None,
    ):
        """Initialize G-Eval metric.

        Args:
            judge_config: Configuration with criteria list.
            criteria_weights: Optional weights for each criterion.
        """
        self.judge_config = judge_config
        self.criteria_weights = criteria_weights or {}
        self._inference_engine = None

    def _get_engine(self):
        """Get or create inference engine."""
        if self._inference_engine is not None:
            return self._inference_engine

        from spark_llm_eval.inference import create_engine

        self._inference_engine = create_engine(self.judge_config.model_config)
        self._inference_engine.initialize()
        return self._inference_engine

    def compute(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        inputs: list[str] | None = None,
        **kwargs,
    ) -> MetricResult:
        """Compute G-Eval scores.

        Evaluates each criterion separately, then combines.

        Args:
            predictions: Predictions to evaluate.
            references: Optional references.
            inputs: Original inputs.

        Returns:
            MetricResult with combined score.
        """
        if not self.judge_config.criteria:
            # fallback to simple judge
            judge = LLMJudgeMetric(self.judge_config)
            return judge.compute(predictions, references, inputs, **kwargs)

        if inputs is None:
            inputs = [""] * len(predictions)

        if references is None:
            references = [None] * len(predictions)

        engine = self._get_engine()
        config = self.judge_config
        scale = config.scale

        # evaluate each criterion
        criteria_scores = {c: [] for c in config.criteria}

        for pred, ref, inp in zip(predictions, references, inputs):
            for criterion in config.criteria:
                prompt = f"""Evaluate the following response on the criterion: {criterion}

Input: {inp}
Response: {pred}
{f"Reference: {ref}" if ref else ""}

Rate on a scale of {scale[0]} to {scale[1]}.

Think step by step about how well the response meets this criterion.
Then provide your rating as JSON: {{"score": <number>, "reasoning": "<explanation>"}}

Your evaluation:"""

                from spark_llm_eval.inference.base import InferenceRequest

                request = InferenceRequest(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=config.temperature,
                )

                try:
                    response = engine.infer(request)
                    score, _ = _parse_judge_response(response.text, scale)
                except Exception as e:
                    logger.warning(f"G-Eval failed for {criterion}: {e}")
                    score = (scale[0] + scale[1]) // 2

                criteria_scores[criterion].append(score)

        # combine scores with weights
        combined_scores = []
        for i in range(len(predictions)):
            weighted_sum = 0.0
            total_weight = 0.0
            for criterion in config.criteria:
                weight = self.criteria_weights.get(criterion, 1.0)
                weighted_sum += criteria_scores[criterion][i] * weight
                total_weight += weight

            if total_weight > 0:
                combined = weighted_sum / total_weight
            else:
                combined = (scale[0] + scale[1]) / 2

            # normalize to 0-1
            normalized = (combined - scale[0]) / (scale[1] - scale[0])
            combined_scores.append(normalized)

        avg_score = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0

        return MetricResult(
            name=self.name,
            value=avg_score,
            per_example_scores=combined_scores,
            metadata={
                "criteria_scores": criteria_scores,
                "criteria": config.criteria,
                "weights": self.criteria_weights,
            },
        )
