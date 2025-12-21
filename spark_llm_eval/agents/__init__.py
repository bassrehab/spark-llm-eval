"""Agent and multi-turn evaluation support.

This module provides tools for evaluating agent-based LLM systems,
including:
- Multi-turn trajectory evaluation
- Tool use assessment
- Multi-agent debate metrics
"""

from spark_llm_eval.agents.debate import (
    Argument,
    ArgumentDiversityMetric,
    ArgumentQualityMetric,
    # Core types
    ArgumentType,
    # Metrics
    ConsensusReachedMetric,
    ContributionBalanceMetric,
    DebateOutcomeAccuracyMetric,
    DebateProgressionMetric,
    DebateRole,
    DebateRound,
    DebateSession,
    # Utilities
    parse_debate_from_messages,
)
from spark_llm_eval.agents.tool_use import (
    # Core types
    ToolCall,
    ToolCallEfficiencyMetric,
    ToolCallPrecisionRecallMetric,
    ToolCallSequence,
    ToolErrorRecoveryMetric,
    ToolOrderAccuracyMetric,
    ToolParameterAccuracyMetric,
    # Metrics
    ToolSelectionAccuracyMetric,
    # Utilities
    parse_tool_calls_from_messages,
)
from spark_llm_eval.agents.trajectory import (
    Action,
    ActionSequenceF1Metric,
    ActionType,
    # Metrics
    GoalCompletionMetric,
    Observation,
    ToolCallAccuracyMetric,
    Trajectory,
    TrajectoryEfficiencyMetric,
    TrajectoryMetric,
    TrajectoryPair,
    Turn,
    # Core types
    TurnRole,
    # Utilities
    parse_trajectory_from_messages,
)

__all__ = [
    # Trajectory types
    "TurnRole",
    "ActionType",
    "Action",
    "Observation",
    "Turn",
    "Trajectory",
    "TrajectoryPair",
    "TrajectoryMetric",
    # Trajectory metrics
    "GoalCompletionMetric",
    "TrajectoryEfficiencyMetric",
    "ToolCallAccuracyMetric",
    "ActionSequenceF1Metric",
    "parse_trajectory_from_messages",
    # Tool use types
    "ToolCall",
    "ToolCallSequence",
    # Tool use metrics
    "ToolSelectionAccuracyMetric",
    "ToolOrderAccuracyMetric",
    "ToolParameterAccuracyMetric",
    "ToolCallEfficiencyMetric",
    "ToolErrorRecoveryMetric",
    "ToolCallPrecisionRecallMetric",
    "parse_tool_calls_from_messages",
    # Debate types
    "ArgumentType",
    "DebateRole",
    "Argument",
    "DebateRound",
    "DebateSession",
    # Debate metrics
    "ConsensusReachedMetric",
    "ArgumentDiversityMetric",
    "ContributionBalanceMetric",
    "DebateProgressionMetric",
    "DebateOutcomeAccuracyMetric",
    "ArgumentQualityMetric",
    "parse_debate_from_messages",
]
