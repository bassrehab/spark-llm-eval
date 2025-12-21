"""Multi-turn trajectory evaluation for agent conversations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric


class TurnRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ActionType(Enum):
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    THINK = "think"
    DELEGATE = "delegate"
    TERMINATE = "terminate"


@dataclass
class Action:
    """An action taken by the agent."""

    action_type: ActionType
    content: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Result of an action (tool output etc)."""

    content: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """Single turn in a conversation."""

    role: TurnRole
    content: str
    action: Action | None = None
    observation: Observation | None = None
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A complete agent trajectory (multi-turn conversation).

    Args:
        trajectory_id: Unique identifier.
        turns: List of conversation turns.
        initial_goal: The user's initial goal/task.
        final_state: Final state achieved.
        goal_achieved: Whether the goal was achieved.
        metadata: Trajectory-level metadata.
    """

    trajectory_id: str
    turns: list[Turn]
    initial_goal: str
    final_state: str | None = None
    goal_achieved: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        """Total number of turns."""
        return len(self.turns)

    @property
    def num_assistant_turns(self) -> int:
        """Number of assistant turns."""
        return sum(1 for t in self.turns if t.role == TurnRole.ASSISTANT)

    @property
    def num_tool_calls(self) -> int:
        """Number of tool calls made."""
        return sum(
            1 for t in self.turns if t.action and t.action.action_type == ActionType.TOOL_CALL
        )

    @property
    def actions(self) -> list[Action]:
        """Extract all actions from the trajectory."""
        return [t.action for t in self.turns if t.action is not None]

    @property
    def tool_calls(self) -> list[Action]:
        """Extract all tool call actions."""
        return [a for a in self.actions if a.action_type == ActionType.TOOL_CALL]

    def get_turn(self, index: int) -> Turn | None:
        """Get turn by index."""
        if 0 <= index < len(self.turns):
            return self.turns[index]
        return None


@dataclass
class TrajectoryPair:
    """A trajectory with optional reference trajectory.

    Used for comparing predicted vs gold trajectories.

    Args:
        predicted: The predicted/actual trajectory.
        reference: Optional reference/gold trajectory.
        reference_goal_achieved: Whether reference achieved goal.
    """

    predicted: Trajectory
    reference: Trajectory | None = None
    reference_goal_achieved: bool | None = None


class TrajectoryMetric(ABC):
    """Base class for trajectory evaluation metrics.

    Unlike standard metrics that compare prediction strings to references,
    trajectory metrics operate on full Trajectory objects.
    """

    name: str = "trajectory_metric"

    def __init__(self, **kwargs):
        """Initialize with optional config."""
        self.config = kwargs

    @abstractmethod
    def compute(
        self,
        trajectories: list[TrajectoryPair],
    ) -> MetricResult:
        """Compute metric on trajectory pairs.

        Args:
            trajectories: List of (predicted, reference) trajectory pairs.

        Returns:
            MetricResult with aggregate and per-trajectory scores.
        """
        pass


@register_metric
class GoalCompletionMetric(TrajectoryMetric, Metric):
    """Measures whether the agent achieved its goal.

    This is a binary metric based on the trajectory's goal_achieved field.
    For automated evaluation, you can use an LLM judge to assess completion.
    """

    name = "goal_completion"

    def compute(
        self,
        predictions: list[str] | list[TrajectoryPair],
        references: list[str] | None = None,
    ) -> MetricResult:
        """Compute goal completion rate.

        Args:
            predictions: Either list of trajectory pairs or list of strings.
                If strings, expects "true"/"false" for goal achieved.
            references: Optional reference values.

        Returns:
            MetricResult with completion rate.
        """
        if not predictions:
            return MetricResult(name=self.name, value=0.0)

        # Handle TrajectoryPair input
        if isinstance(predictions[0], TrajectoryPair):
            scores = []
            for pair in predictions:
                if pair.predicted.goal_achieved is not None:
                    scores.append(1.0 if pair.predicted.goal_achieved else 0.0)
                else:
                    # Assume not achieved if not specified
                    scores.append(0.0)
        else:
            # Handle string input (e.g., from LLM judge)
            scores = []
            for pred in predictions:
                pred_lower = str(pred).lower().strip()
                if pred_lower in ("true", "yes", "1", "achieved", "success"):
                    scores.append(1.0)
                else:
                    scores.append(0.0)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class TrajectoryEfficiencyMetric(TrajectoryMetric, Metric):
    """Measures trajectory efficiency (fewer turns = more efficient).

    Computes ratio of reference turns to predicted turns.
    Score > 1 means predicted was more efficient.
    Score < 1 means predicted used more turns.

    If no reference is provided, uses max_turns config for normalization.
    """

    name = "trajectory_efficiency"

    def __init__(self, max_turns: int = 20, **kwargs):
        """Initialize with max turns for normalization.

        Args:
            max_turns: Maximum expected turns (for normalization when no reference).
        """
        super().__init__(**kwargs)
        self.max_turns = max_turns

    def compute(
        self,
        predictions: list[str] | list[TrajectoryPair],
        references: list[str] | None = None,
    ) -> MetricResult:
        """Compute efficiency scores.

        Args:
            predictions: Trajectory pairs or turn counts as strings.
            references: Optional reference turn counts.

        Returns:
            MetricResult with efficiency scores.
        """
        if not predictions:
            return MetricResult(name=self.name, value=0.0)

        scores = []

        if isinstance(predictions[0], TrajectoryPair):
            for pair in predictions:
                pred_turns = pair.predicted.num_turns
                if pair.reference:
                    ref_turns = pair.reference.num_turns
                    # Efficiency = reference / predicted (higher is better)
                    efficiency = ref_turns / pred_turns if pred_turns > 0 else 0.0
                else:
                    # Normalize by max_turns: (max - actual) / max
                    efficiency = (self.max_turns - pred_turns) / self.max_turns
                    efficiency = max(0.0, min(1.0, efficiency))
                scores.append(efficiency)
        else:
            # Handle numeric strings (turn counts)
            refs = references or [str(self.max_turns)] * len(predictions)
            for pred, ref in zip(predictions, refs):
                pred_turns = int(pred) if pred else self.max_turns
                ref_turns = int(ref) if ref else self.max_turns
                efficiency = ref_turns / pred_turns if pred_turns > 0 else 0.0
                scores.append(efficiency)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class ToolCallAccuracyMetric(TrajectoryMetric, Metric):
    """Measures accuracy of tool calls in trajectories.

    Computes what fraction of tool calls match the reference trajectory
    in terms of:
    - Tool name (required)
    - Parameters (optional, configurable)
    """

    name = "tool_call_accuracy"

    def __init__(self, check_params: bool = False, **kwargs):
        """Initialize tool call accuracy metric.

        Args:
            check_params: Whether to check parameter values match.
        """
        super().__init__(**kwargs)
        self.check_params = check_params

    def compute(
        self,
        predictions: list[str] | list[TrajectoryPair],
        references: list[str] | None = None,
    ) -> MetricResult:
        """Compute tool call accuracy.

        Args:
            predictions: Trajectory pairs or tool call sequences.
            references: Optional reference tool sequences.

        Returns:
            MetricResult with accuracy scores.
        """
        if not predictions:
            return MetricResult(name=self.name, value=0.0)

        scores = []

        if isinstance(predictions[0], TrajectoryPair):
            for pair in predictions:
                if not pair.reference:
                    # No reference, skip
                    continue

                pred_tools = pair.predicted.tool_calls
                ref_tools = pair.reference.tool_calls

                if not ref_tools:
                    # No reference tools expected
                    scores.append(1.0 if not pred_tools else 0.0)
                    continue

                # Count matching tool calls
                matches = 0
                for i, ref_tool in enumerate(ref_tools):
                    if i < len(pred_tools):
                        pred_tool = pred_tools[i]
                        if pred_tool.content == ref_tool.content:
                            if not self.check_params:
                                matches += 1
                            elif pred_tool.parameters == ref_tool.parameters:
                                matches += 1

                scores.append(matches / len(ref_tools))
        else:
            # Handle string input (comma-separated tool names)
            refs = references or [""] * len(predictions)
            for pred, ref in zip(predictions, refs):
                pred_tools = [t.strip() for t in str(pred).split(",") if t.strip()]
                ref_tools = [t.strip() for t in str(ref).split(",") if t.strip()]

                if not ref_tools:
                    scores.append(1.0 if not pred_tools else 0.0)
                    continue

                matches = sum(1 for p, r in zip(pred_tools, ref_tools) if p == r)
                scores.append(matches / len(ref_tools))

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class ActionSequenceF1Metric(TrajectoryMetric, Metric):
    """Computes F1 score for action sequences.

    Treats predicted and reference action sequences as sets
    and computes precision, recall, and F1.
    """

    name = "action_sequence_f1"

    def compute(
        self,
        predictions: list[str] | list[TrajectoryPair],
        references: list[str] | None = None,
    ) -> MetricResult:
        """Compute action sequence F1 scores.

        Args:
            predictions: Trajectory pairs or action sequences.
            references: Optional reference action sequences.

        Returns:
            MetricResult with F1 scores.
        """
        if not predictions:
            return MetricResult(name=self.name, value=0.0)

        scores = []

        if isinstance(predictions[0], TrajectoryPair):
            for pair in predictions:
                if not pair.reference:
                    continue

                pred_actions = set(
                    f"{a.action_type.value}:{a.content}" for a in pair.predicted.actions
                )
                ref_actions = set(
                    f"{a.action_type.value}:{a.content}" for a in pair.reference.actions
                )

                f1 = self._compute_f1(pred_actions, ref_actions)
                scores.append(f1)
        else:
            # Handle string input
            refs = references or [""] * len(predictions)
            for pred, ref in zip(predictions, refs):
                pred_actions = set(t.strip() for t in str(pred).split(",") if t.strip())
                ref_actions = set(t.strip() for t in str(ref).split(",") if t.strip())

                f1 = self._compute_f1(pred_actions, ref_actions)
                scores.append(f1)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
            metadata={"metric_type": "f1"},
        )

    def _compute_f1(self, pred: set, ref: set) -> float:
        """Compute F1 between two sets."""
        if not pred and not ref:
            return 1.0
        if not pred or not ref:
            return 0.0

        true_positives = len(pred & ref)
        precision = true_positives / len(pred)
        recall = true_positives / len(ref)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def parse_trajectory_from_messages(
    messages: list[dict[str, Any]],
    trajectory_id: str = "unknown",
) -> Trajectory:
    """Parse a trajectory from a list of messages.

    Converts standard chat message format to Trajectory object.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        trajectory_id: ID to assign to the trajectory.

    Returns:
        Trajectory object.
    """
    turns = []
    initial_goal = ""

    for i, msg in enumerate(messages):
        role_str = msg.get("role", "user").lower()
        content = msg.get("content", "")

        # Map role string to enum
        role_map = {
            "user": TurnRole.USER,
            "assistant": TurnRole.ASSISTANT,
            "system": TurnRole.SYSTEM,
            "tool": TurnRole.TOOL,
            "function": TurnRole.TOOL,
        }
        role = role_map.get(role_str, TurnRole.USER)

        # Extract initial goal from first user message
        if role == TurnRole.USER and not initial_goal:
            initial_goal = content

        # Check for tool calls in assistant messages
        action = None
        if role == TurnRole.ASSISTANT:
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Take first tool call as primary action
                tc = tool_calls[0]
                action = Action(
                    action_type=ActionType.TOOL_CALL,
                    content=tc.get("function", {}).get("name", ""),
                    parameters=tc.get("function", {}).get("arguments", {}),
                )
            else:
                action = Action(
                    action_type=ActionType.RESPONSE,
                    content=content,
                )

        # Check for tool response
        observation = None
        if role == TurnRole.TOOL:
            observation = Observation(
                content=content,
                success="error" not in content.lower(),
            )

        turn = Turn(
            role=role,
            content=content,
            action=action,
            observation=observation,
            turn_index=i,
            metadata=msg.get("metadata", {}),
        )
        turns.append(turn)

    return Trajectory(
        trajectory_id=trajectory_id,
        turns=turns,
        initial_goal=initial_goal,
    )
