"""Multi-agent debate evaluation metrics."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spark_llm_eval.evaluation.base import Metric, MetricResult, register_metric

logger = logging.getLogger(__name__)


class ArgumentType(Enum):
    CLAIM = "claim"
    SUPPORT = "support"
    COUNTER = "counter"
    CONCESSION = "concession"
    SYNTHESIS = "synthesis"
    CLARIFICATION = "clarification"


class DebateRole(Enum):
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    MEDIATOR = "mediator"
    JUDGE = "judge"


@dataclass
class Argument:
    """A single argument in a debate."""

    agent_id: str
    role: DebateRole
    argument_type: ArgumentType
    content: str
    references: list[int] = field(default_factory=list)  # refs to prior args
    strength: float | None = None  # 0-1 if scored
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    round_number: int
    arguments: list[Argument]
    topic: str = ""


@dataclass
class DebateSession:
    """Complete debate session between agents."""

    session_id: str
    topic: str
    agents: list[str]
    rounds: list[DebateRound]
    final_verdict: str | None = None
    winner_agent_id: str | None = None
    consensus_reached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_rounds(self):
        """Total number of rounds."""
        return len(self.rounds)

    @property
    def all_arguments(self) -> list[Argument]:
        """Get all arguments across rounds."""
        return [arg for round in self.rounds for arg in round.arguments]

    @property
    def num_arguments(self) -> int:
        """Total number of arguments."""
        return sum(len(r.arguments) for r in self.rounds)

    def arguments_by_agent(self, agent_id: str) -> list[Argument]:
        """Get all arguments from a specific agent."""
        return [a for a in self.all_arguments if a.agent_id == agent_id]

    def argument_type_counts(self) -> dict[ArgumentType, int]:
        """Count arguments by type."""
        counts: dict[ArgumentType, int] = {}
        for arg in self.all_arguments:
            counts[arg.argument_type] = counts.get(arg.argument_type, 0) + 1
        return counts


@register_metric
class ConsensusReachedMetric(Metric):
    """Measures whether agents reached consensus.

    Binary metric: 1.0 if consensus, 0.0 if not.
    """

    name = "consensus_reached"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute consensus rate.

        Args:
            predictions: Predicted outcomes ("consensus", "no_consensus").
            references: Reference outcomes.

        Returns:
            MetricResult with consensus rate.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_consensus = pred.lower().strip() in ("consensus", "yes", "true", "1")
            ref_consensus = ref.lower().strip() in ("consensus", "yes", "true", "1")

            if pred_consensus == ref_consensus:
                scores.append(1.0)
            else:
                scores.append(0.0)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class ArgumentDiversityMetric(Metric):
    """Measures diversity of argument types used.

    Higher diversity suggests richer debate with varied reasoning strategies.
    """

    name = "argument_diversity"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute argument diversity.

        Args:
            predictions: Argument types used (comma-separated).
            references: All possible argument types (comma-separated).

        Returns:
            MetricResult with diversity scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_types = {t.strip().lower() for t in pred.split(",") if t.strip()}
            ref_types = {t.strip().lower() for t in ref.split(",") if t.strip()}

            if not ref_types:
                # Use default argument types
                ref_types = {t.value for t in ArgumentType}

            diversity = len(pred_types) / len(ref_types) if ref_types else 0.0
            scores.append(min(1.0, diversity))

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class ContributionBalanceMetric(Metric):
    """Measures balance of contributions across agents.

    Uses entropy to measure how evenly distributed contributions are.
    Score of 1.0 means perfectly balanced, lower scores mean imbalanced.
    """

    name = "contribution_balance"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute contribution balance using entropy.

        Args:
            predictions: Contribution counts per agent (comma-separated numbers).
            references: Number of agents (as string).

        Returns:
            MetricResult with balance scores.
        """
        self.validate_inputs(predictions, references)

        import math

        scores = []
        for pred, ref in zip(predictions, references):
            try:
                # Parse contribution counts
                counts = [int(c.strip()) for c in pred.split(",") if c.strip()]
                num_agents = int(ref) if ref else len(counts)
            except ValueError:
                scores.append(0.0)
                continue

            if not counts or sum(counts) == 0:
                scores.append(0.0)
                continue

            # Compute normalized entropy
            total = sum(counts)
            probs = [c / total for c in counts]

            # Entropy
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)

            # Maximum entropy for uniform distribution
            max_entropy = math.log2(num_agents) if num_agents > 1 else 1

            # Normalized entropy (0 = imbalanced, 1 = perfectly balanced)
            balance = entropy / max_entropy if max_entropy > 0 else 1.0
            scores.append(balance)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )


@register_metric
class DebateProgressionMetric(Metric):
    """Measures logical progression of the debate.

    Evaluates whether arguments build on each other properly,
    with claims followed by support/counter arguments.
    """

    name = "debate_progression"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute debate progression quality.

        Args:
            predictions: Argument type sequence (comma-separated).
            references: Expected progression patterns (comma-separated).

        Returns:
            MetricResult with progression scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_seq = [t.strip().lower() for t in pred.split(",") if t.strip()]
            ref_seq = [t.strip().lower() for t in ref.split(",") if t.strip()]

            if not pred_seq:
                scores.append(0.0)
                continue

            # Score based on logical transitions
            score = self._score_progression(pred_seq, ref_seq)
            scores.append(score)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )

    def _score_progression(
        self,
        sequence: list[str],
        reference: list[str],
    ) -> float:
        """Score the logical progression of argument types."""
        # Define valid transitions
        valid_transitions = {
            "claim": {"support", "counter", "clarification"},
            "support": {"counter", "concession", "synthesis", "claim"},
            "counter": {"support", "concession", "synthesis", "counter"},
            "concession": {"synthesis", "claim", "support"},
            "synthesis": {"claim", "concession"},
            "clarification": {"support", "counter", "claim"},
        }

        if len(sequence) < 2:
            return 1.0 if sequence else 0.0

        valid_count = 0
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_arg = sequence[i + 1]

            valid_next = valid_transitions.get(current, set())
            if next_arg in valid_next:
                valid_count += 1

        return valid_count / (len(sequence) - 1)


@register_metric
class DebateOutcomeAccuracyMetric(Metric):
    """Measures accuracy of debate outcome prediction.

    Compares predicted winner/outcome to reference.
    """

    name = "debate_outcome_accuracy"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute debate outcome accuracy.

        Args:
            predictions: Predicted outcomes (winner IDs or verdicts).
            references: Reference outcomes.

        Returns:
            MetricResult with accuracy scores.
        """
        self.validate_inputs(predictions, references)

        scores = []
        for pred, ref in zip(predictions, references):
            pred_outcome = pred.lower().strip() if pred else ""
            ref_outcome = ref.lower().strip() if ref else ""

            if pred_outcome == ref_outcome:
                scores.append(1.0)
            elif self._is_partial_match(pred_outcome, ref_outcome):
                scores.append(0.5)
            else:
                scores.append(0.0)

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
        )

    def _is_partial_match(self, pred: str, ref: str) -> bool:
        """Check for partial match (e.g., both indicate same direction)."""
        positive_indicators = {"win", "correct", "true", "yes", "agree"}
        negative_indicators = {"lose", "incorrect", "false", "no", "disagree"}

        pred_positive = any(ind in pred for ind in positive_indicators)
        pred_negative = any(ind in pred for ind in negative_indicators)
        ref_positive = any(ind in ref for ind in positive_indicators)
        ref_negative = any(ind in ref for ind in negative_indicators)

        return (pred_positive and ref_positive) or (pred_negative and ref_negative)


@register_metric
class ArgumentQualityMetric(Metric):
    """Measures overall quality of arguments.

    Combines multiple factors:
    - Relevance to topic
    - Logical coherence
    - Evidence citation
    - Response to others' points
    """

    name = "argument_quality"

    def __init__(
        self,
        relevance_weight: float = 0.3,
        coherence_weight: float = 0.3,
        evidence_weight: float = 0.2,
        response_weight: float = 0.2,
        **kwargs,
    ):
        """Initialize with component weights.

        Args:
            relevance_weight: Weight for relevance score.
            coherence_weight: Weight for coherence score.
            evidence_weight: Weight for evidence citation score.
            response_weight: Weight for response to others score.
        """
        super().__init__(**kwargs)
        self.weights = {
            "relevance": relevance_weight,
            "coherence": coherence_weight,
            "evidence": evidence_weight,
            "response": response_weight,
        }

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> MetricResult:
        """Compute argument quality scores.

        Args:
            predictions: Quality scores as JSON (relevance,coherence,evidence,response).
            references: Reference quality scores.

        Returns:
            MetricResult with weighted quality scores.
        """
        self.validate_inputs(predictions, references)

        import json

        scores = []
        component_scores: dict[str, list[float]] = {k: [] for k in self.weights}

        for pred, ref in zip(predictions, references):
            try:
                # Parse component scores
                if pred.startswith("{"):
                    pred_scores = json.loads(pred)
                else:
                    # Assume comma-separated: relevance,coherence,evidence,response
                    parts = [float(p.strip()) for p in pred.split(",")]
                    pred_scores = dict(zip(self.weights.keys(), parts))

                # Compute weighted average
                total = 0.0
                for component, weight in self.weights.items():
                    comp_score = pred_scores.get(component, 0.0)
                    total += weight * comp_score
                    component_scores[component].append(comp_score)

                scores.append(min(1.0, max(0.0, total)))
            except (json.JSONDecodeError, ValueError):
                scores.append(0.0)
                for comp in component_scores:
                    component_scores[comp].append(0.0)

        # Compute average component scores for metadata
        avg_components = {k: sum(v) / len(v) if v else 0.0 for k, v in component_scores.items()}

        return MetricResult(
            name=self.name,
            value=sum(scores) / len(scores) if scores else 0.0,
            per_example_scores=scores,
            metadata={"component_scores": avg_components},
        )


def parse_debate_from_messages(
    messages: list[dict[str, Any]],
    session_id: str = "unknown",
    topic: str = "",
) -> DebateSession:
    """Parse a debate session from a list of messages.

    Args:
        messages: List of message dicts with agent/role info.
        session_id: ID for the debate session.
        topic: Debate topic.

    Returns:
        DebateSession object.
    """
    agents = set()
    rounds: list[DebateRound] = []
    current_round_args: list[Argument] = []
    round_num = 0

    for msg in messages:
        agent_id = msg.get("agent_id", msg.get("name", "unknown"))
        agents.add(agent_id)

        role_str = msg.get("debate_role", "proponent").lower()
        role_map = {
            "proponent": DebateRole.PROPONENT,
            "opponent": DebateRole.OPPONENT,
            "mediator": DebateRole.MEDIATOR,
            "judge": DebateRole.JUDGE,
        }
        role = role_map.get(role_str, DebateRole.PROPONENT)

        arg_type_str = msg.get("argument_type", "claim").lower()
        arg_type_map = {
            "claim": ArgumentType.CLAIM,
            "support": ArgumentType.SUPPORT,
            "counter": ArgumentType.COUNTER,
            "concession": ArgumentType.CONCESSION,
            "synthesis": ArgumentType.SYNTHESIS,
            "clarification": ArgumentType.CLARIFICATION,
        }
        arg_type = arg_type_map.get(arg_type_str, ArgumentType.CLAIM)

        content = msg.get("content", "")
        references = msg.get("references", [])

        arg = Argument(
            agent_id=agent_id,
            role=role,
            argument_type=arg_type,
            content=content,
            references=references,
            metadata=msg.get("metadata", {}),
        )
        current_round_args.append(arg)

        # Check for round boundary
        if msg.get("end_round", False) or (
            arg_type == ArgumentType.SYNTHESIS and len(current_round_args) >= 2
        ):
            rounds.append(
                DebateRound(
                    round_number=round_num,
                    arguments=current_round_args,
                    topic=topic,
                )
            )
            current_round_args = []
            round_num += 1

    # Add remaining arguments as final round
    if current_round_args:
        rounds.append(
            DebateRound(
                round_number=round_num,
                arguments=current_round_args,
                topic=topic,
            )
        )

    return DebateSession(
        session_id=session_id,
        topic=topic,
        agents=list(agents),
        rounds=rounds,
    )
