"""Tool use evaluation metrics for agents."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a single tool call."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: str | None = None

    def __hash__(self):
        # need this for set operations - hash on name + params
        params_str = json.dumps(self.parameters, sort_keys=True) if self.parameters else ""
        return hash((self.name, params_str))

    def __eq__(self, other):
        if not isinstance(other, ToolCall):
            return False
        return self.name == other.name and self.parameters == other.parameters


@dataclass
class ToolCallSequence:
    """A sequence of tool calls from an agent."""

    calls: list[ToolCall]
    task_description: str = ""
    final_answer: str | None = None

    @property
    def tool_names(self):
        return [c.name for c in self.calls]

    @property
    def unique_tools(self):
        return set(self.tool_names)

    @property
    def num_calls(self):
        return len(self.calls)

    @property
    def num_failures(self):
        return sum(1 for c in self.calls if not c.success)


@register_metric
class ToolSelectionAccuracyMetric(Metric):
    """Measures if the agent selected the correct tools.

    Compares predicted tool names to reference tool names,
    ignoring order and parameters.
    """

    name = "tool_selection_accuracy"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute tool selection accuracy.

        Args:
            predictions: Predicted tool names (comma-separated strings).
            references: Reference tool names (comma-separated strings).

        Returns:
            MetricResult with accuracy scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tools = self._parse_tools(pred)
            ref_tools = self._parse_tools(ref)

            if not ref_tools:
                # No tools expected
                scores.append(1.0 if not pred_tools else 0.0)
                continue

            # Jaccard similarity
            intersection = pred_tools & ref_tools
            union = pred_tools | ref_tools
            score = len(intersection) / len(union) if union else 1.0
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )

    def _parse_tools(self, tools_str: str) -> set[str]:
        """Parse comma-separated tool names."""
        if not tools_str:
            return set()
        return {t.strip().lower() for t in tools_str.split(",") if t.strip()}


@register_metric
class ToolOrderAccuracyMetric(Metric):
    """Measures if tools were called in the correct order.

    Uses longest common subsequence (LCS) to measure ordering accuracy.
    """

    name = "tool_order_accuracy"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute tool ordering accuracy using LCS.

        Args:
            predictions: Predicted tool sequences (comma-separated).
            references: Reference tool sequences (comma-separated).

        Returns:
            MetricResult with LCS-based accuracy scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_seq = self._parse_sequence(pred)
            ref_seq = self._parse_sequence(ref)

            if not ref_seq:
                scores.append(1.0 if not pred_seq else 0.0)
                continue

            lcs_len = self._lcs_length(pred_seq, ref_seq)
            score = lcs_len / len(ref_seq)
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )

    def _parse_sequence(self, seq_str: str) -> list[str]:
        """Parse comma-separated sequence."""
        if not seq_str:
            return []
        return [t.strip().lower() for t in seq_str.split(",") if t.strip()]

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


@register_metric
class ToolParameterAccuracyMetric(Metric):
    """Measures accuracy of tool call parameters.

    Compares parameter keys and values between predicted and reference.
    """

    name = "tool_param_accuracy"

    def __init__(self, check_values: bool = True, **kwargs):
        """Initialize parameter accuracy metric.

        Args:
            check_values: Whether to check parameter values (not just keys).
        """
        super().__init__(**kwargs)
        self.check_values = check_values

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute parameter accuracy.

        Args:
            predictions: Predicted parameters as JSON strings.
            references: Reference parameters as JSON strings.

        Returns:
            MetricResult with accuracy scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_params = self._parse_params(pred)
            ref_params = self._parse_params(ref)

            if not ref_params:
                scores.append(1.0 if not pred_params else 0.0)
                continue

            if self.check_values:
                # Check both keys and values
                matching = sum(
                    1 for k, v in ref_params.items() if k in pred_params and pred_params[k] == v
                )
            else:
                # Check only keys
                matching = len(set(pred_params.keys()) & set(ref_params.keys()))

            score = matching / len(ref_params)
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )

    def _parse_params(self, params_str: str) -> dict[str, Any]:
        """Parse JSON parameter string."""
        if not params_str:
            return {}
        try:
            return json.loads(params_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse params JSON: {params_str}")
            return {}


@register_metric
class ToolCallEfficiencyMetric(Metric):
    """Measures efficiency of tool usage.

    Computes ratio of necessary tool calls to actual calls.
    Score of 1.0 means perfect efficiency (no redundant calls).
    """

    name = "tool_efficiency"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute tool call efficiency.

        Args:
            predictions: Number of tool calls made (as strings).
            references: Minimum necessary tool calls (as strings).

        Returns:
            MetricResult with efficiency scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            try:
                pred_count = int(pred) if pred else 0
                ref_count = int(ref) if ref else 0
            except ValueError:
                scores.append(0.0)
                continue

            if ref_count == 0:
                scores.append(1.0 if pred_count == 0 else 0.0)
                continue

            # Efficiency = min(1, reference / predicted)
            efficiency = min(1.0, ref_count / pred_count) if pred_count > 0 else 0.0
            scores.append(efficiency)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class ToolErrorRecoveryMetric(Metric):
    """Measures how well the agent recovers from tool errors.

    Evaluates whether the agent:
    - Retries failed calls appropriately
    - Uses fallback tools when needed
    - Doesn't get stuck in error loops
    """

    name = "tool_error_recovery"

    def __init__(self, max_retries: int = 3, **kwargs):
        """Initialize error recovery metric.

        Args:
            max_retries: Maximum expected retries before giving up.
        """
        super().__init__(**kwargs)
        self.max_retries = max_retries

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute error recovery scores.

        Args:
            predictions: Recovery status strings ("recovered", "stuck", "gave_up").
            references: Expected recovery outcomes.

        Returns:
            MetricResult with recovery scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_status = pred.lower().strip() if pred else ""
            ref_status = ref.lower().strip() if ref else ""

            # Score based on recovery behavior
            if pred_status == ref_status:
                scores.append(1.0)
            elif pred_status == "recovered" and ref_status in ("recovered", "fallback"):
                scores.append(0.8)  # Recovered when fallback was expected
            elif pred_status == "fallback" and ref_status == "recovered":
                scores.append(0.6)  # Used fallback when direct recovery was expected
            elif pred_status in ("stuck", "loop") and ref_status in ("recovered", "fallback"):
                scores.append(0.0)  # Failed to recover
            else:
                scores.append(0.5)  # Partial credit

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class ToolCallPrecisionRecallMetric(Metric):
    """Computes precision, recall, and F1 for tool calls.

    Treats tool calls as a set and computes standard IR metrics.
    """

    name = "tool_call_f1"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute precision, recall, and F1 for tool calls.

        Args:
            predictions: Predicted tool calls (comma-separated).
            references: Reference tool calls (comma-separated).

        Returns:
            MetricResult with F1 scores and precision/recall in metadata.
        """
        self.validate_inputs(predictions, references)

        scores = []
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            pred_calls = {t.strip().lower() for t in pred.split(",") if t.strip()}
            ref_calls = {t.strip().lower() for t in ref.split(",") if t.strip()}

            if not pred_calls and not ref_calls:
                scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
                continue

            true_positives = len(pred_calls & ref_calls)
            precision = true_positives / len(pred_calls) if pred_calls else 0.0
            recall = true_positives / len(ref_calls) if ref_calls else 0.0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
            metadata={
                "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
                "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            },
        )


def parse_tool_calls_from_messages(
    messages: list[dict[str, Any]],
) -> ToolCallSequence:
    """Parse tool calls from a list of chat messages.

    Args:
        messages: List of message dicts in OpenAI format.

    Returns:
        ToolCallSequence with extracted tool calls.
    """
    calls = []
    task_description = ""
    final_answer = None

    for msg in messages:
        role = msg.get("role", "")

        # Get task from first user message
        if role == "user" and not task_description:
            task_description = msg.get("content", "")

        # Extract tool calls from assistant messages
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                params_str = func.get("arguments", "{}")

                try:
                    params = json.loads(params_str) if isinstance(params_str, str) else params_str
                except json.JSONDecodeError:
                    params = {}

                calls.append(ToolCall(name=name, parameters=params))

            # Track final answer (last assistant message without tool calls)
            if not tool_calls and msg.get("content"):
                final_answer = msg.get("content")

        # Match tool results to calls
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")

            # Find matching call and update result
            # Note: In real implementation, would need to track IDs
            if calls:
                calls[-1].result = content
                if "error" in content.lower():
                    calls[-1].success = False
                    calls[-1].error = content

    return ToolCallSequence(
        calls=calls,
        task_description=task_description,
        final_answer=final_answer,
    )
