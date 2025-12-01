"""Unit tests for agent evaluation metrics.

Run with: pytest tests/unit/test_agents.py -v
"""

import pytest
from spark_llm_eval.agents import (
    # Trajectory
    TurnRole,
    ActionType,
    Action,
    Observation,
    Turn,
    Trajectory,
    TrajectoryPair,
    GoalCompletionMetric,
    TrajectoryEfficiencyMetric,
    ToolCallAccuracyMetric,
    ActionSequenceF1Metric,
    parse_trajectory_from_messages,
    # Tool use
    ToolCall,
    ToolCallSequence,
    ToolSelectionAccuracyMetric,
    ToolOrderAccuracyMetric,
    ToolParameterAccuracyMetric,
    ToolCallEfficiencyMetric,
    ToolCallPrecisionRecallMetric,
    parse_tool_calls_from_messages,
    # Debate
    ArgumentType,
    DebateRole,
    Argument,
    DebateRound,
    DebateSession,
    ConsensusReachedMetric,
    ArgumentDiversityMetric,
    ContributionBalanceMetric,
    DebateProgressionMetric,
    DebateOutcomeAccuracyMetric,
    ArgumentQualityMetric,
    parse_debate_from_messages,
)


class TestTrajectoryTypes:
    """Test trajectory data structures."""

    def test_action_creation(self):
        """Test Action dataclass."""
        action = Action(
            action_type=ActionType.TOOL_CALL,
            content="search",
            parameters={"query": "weather"},
        )
        assert action.action_type == ActionType.TOOL_CALL
        assert action.content == "search"
        assert action.parameters["query"] == "weather"

    def test_observation_creation(self):
        """Test Observation dataclass."""
        obs = Observation(content="sunny, 72F", success=True)
        assert obs.content == "sunny, 72F"
        assert obs.success is True
        assert obs.error is None

    def test_turn_creation(self):
        """Test Turn dataclass."""
        turn = Turn(
            role=TurnRole.ASSISTANT,
            content="Let me check the weather",
            action=Action(action_type=ActionType.TOOL_CALL, content="weather"),
            turn_index=0,
        )
        assert turn.role == TurnRole.ASSISTANT
        assert turn.action is not None
        assert turn.action.content == "weather"

    def test_trajectory_properties(self):
        """Test Trajectory computed properties."""
        turns = [
            Turn(role=TurnRole.USER, content="Check weather", turn_index=0),
            Turn(
                role=TurnRole.ASSISTANT,
                content="Checking...",
                action=Action(action_type=ActionType.TOOL_CALL, content="get_weather"),
                turn_index=1,
            ),
            Turn(
                role=TurnRole.TOOL,
                content="72F sunny",
                observation=Observation(content="72F sunny"),
                turn_index=2,
            ),
            Turn(
                role=TurnRole.ASSISTANT,
                content="It's 72F and sunny.",
                action=Action(action_type=ActionType.RESPONSE, content="It's 72F and sunny."),
                turn_index=3,
            ),
        ]
        traj = Trajectory(
            trajectory_id="test-1",
            turns=turns,
            initial_goal="Check weather",
            goal_achieved=True,
        )

        assert traj.num_turns == 4
        assert traj.num_assistant_turns == 2
        assert traj.num_tool_calls == 1
        assert len(traj.actions) == 2
        assert len(traj.tool_calls) == 1

    def test_parse_trajectory_from_messages(self):
        """Test parsing trajectory from message format."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "72F sunny"},
            {"role": "assistant", "content": "It's 72F and sunny in NYC."},
        ]

        traj = parse_trajectory_from_messages(messages, "test-traj")

        assert traj.trajectory_id == "test-traj"
        assert len(traj.turns) == 4
        assert traj.initial_goal == "What's the weather?"
        assert traj.num_tool_calls == 1


class TestTrajectoryMetrics:
    """Test trajectory evaluation metrics."""

    def test_goal_completion_with_trajectory_pairs(self):
        """Test goal completion metric with TrajectoryPair input."""
        metric = GoalCompletionMetric()

        # Create trajectories
        traj1 = Trajectory(
            trajectory_id="1",
            turns=[],
            initial_goal="goal1",
            goal_achieved=True,
        )
        traj2 = Trajectory(
            trajectory_id="2",
            turns=[],
            initial_goal="goal2",
            goal_achieved=False,
        )

        pairs = [
            TrajectoryPair(predicted=traj1),
            TrajectoryPair(predicted=traj2),
        ]

        result = metric.compute(pairs, None)
        assert result.value == 0.5  # 1 out of 2 achieved
        assert result.per_example_scores == [1.0, 0.0]

    def test_goal_completion_with_strings(self):
        """Test goal completion metric with string input."""
        metric = GoalCompletionMetric()

        predictions = ["true", "false", "yes", "no"]
        references = ["", "", "", ""]

        result = metric.compute(predictions, references)
        assert result.value == 0.5
        assert result.per_example_scores == [1.0, 0.0, 1.0, 0.0]

    def test_trajectory_efficiency(self):
        """Test trajectory efficiency metric."""
        metric = TrajectoryEfficiencyMetric(max_turns=10)

        # Test with string inputs (turn counts)
        predictions = ["5", "10", "2"]
        references = ["5", "5", "5"]

        result = metric.compute(predictions, references)

        # 5/5=1.0, 5/10=0.5, 5/2=2.5 (capped at ratio)
        assert result.per_example_scores[0] == 1.0  # Equal turns
        assert result.per_example_scores[1] == 0.5  # Twice as many
        assert result.per_example_scores[2] == 2.5  # Half as many

    def test_action_sequence_f1(self):
        """Test action sequence F1 metric."""
        metric = ActionSequenceF1Metric()

        predictions = [
            "tool_call:search, response:answer",
            "tool_call:calc",
        ]
        references = [
            "tool_call:search, response:answer",
            "tool_call:search, tool_call:calc",
        ]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # Perfect match
        # Second: pred={calc}, ref={search, calc} -> P=1, R=0.5, F1=0.67
        assert 0.6 < result.per_example_scores[1] < 0.7


class TestToolUseTypes:
    """Test tool use data structures."""

    def test_tool_call_creation(self):
        """Test ToolCall dataclass."""
        tc = ToolCall(
            name="search",
            parameters={"query": "python"},
            result="Found 10 results",
            success=True,
        )
        assert tc.name == "search"
        assert tc.parameters["query"] == "python"
        assert tc.success is True

    def test_tool_call_equality(self):
        """Test ToolCall equality and hashing."""
        tc1 = ToolCall(name="search", parameters={"q": "a"})
        tc2 = ToolCall(name="search", parameters={"q": "a"})
        tc3 = ToolCall(name="search", parameters={"q": "b"})

        assert tc1 == tc2
        assert tc1 != tc3
        assert hash(tc1) == hash(tc2)

    def test_tool_call_sequence_properties(self):
        """Test ToolCallSequence computed properties."""
        seq = ToolCallSequence(
            calls=[
                ToolCall(name="search", success=True),
                ToolCall(name="calc", success=False, error="div by zero"),
                ToolCall(name="search", success=True),
            ],
            task_description="Find and calculate",
        )

        assert seq.tool_names == ["search", "calc", "search"]
        assert seq.unique_tools == {"search", "calc"}
        assert seq.num_calls == 3
        assert seq.num_failures == 1

    def test_parse_tool_calls_from_messages(self):
        """Test parsing tool calls from message format."""
        messages = [
            {"role": "user", "content": "Search for python tutorials"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "python tutorials"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Found 10 results"},
            {"role": "assistant", "content": "Here are the tutorials..."},
        ]

        seq = parse_tool_calls_from_messages(messages)

        assert seq.task_description == "Search for python tutorials"
        assert len(seq.calls) == 1
        assert seq.calls[0].name == "web_search"
        assert seq.final_answer == "Here are the tutorials..."


class TestToolUseMetrics:
    """Test tool use evaluation metrics."""

    def test_tool_selection_accuracy(self):
        """Test tool selection accuracy metric."""
        metric = ToolSelectionAccuracyMetric()

        predictions = ["search, calc", "search", ""]
        references = ["search, calc", "search, calc", ""]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # Exact match
        # 1/2 = 0.5 Jaccard
        assert result.per_example_scores[1] == 0.5  # Missing calc
        assert result.per_example_scores[2] == 1.0  # Both empty

    def test_tool_order_accuracy(self):
        """Test tool order accuracy with LCS."""
        metric = ToolOrderAccuracyMetric()

        predictions = ["a, b, c", "a, c, b", "d, e, f"]
        references = ["a, b, c", "a, b, c", "a, b, c"]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # Perfect order
        # LCS(a,c,b vs a,b,c) = 2, score = 2/3
        assert 0.6 < result.per_example_scores[1] < 0.7
        assert result.per_example_scores[2] == 0.0  # No common elements

    def test_tool_parameter_accuracy(self):
        """Test tool parameter accuracy metric."""
        metric = ToolParameterAccuracyMetric(check_values=True)

        predictions = [
            '{"query": "python", "limit": 10}',
            '{"query": "java"}',
            '{}',
        ]
        references = [
            '{"query": "python", "limit": 10}',
            '{"query": "python", "limit": 10}',
            '{"query": "test"}',
        ]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # All match
        assert result.per_example_scores[1] == 0.0  # None match (wrong values)
        assert result.per_example_scores[2] == 0.0  # Missing params

    def test_tool_efficiency(self):
        """Test tool efficiency metric."""
        metric = ToolCallEfficiencyMetric()

        predictions = ["5", "10", "3"]
        references = ["5", "5", "6"]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # 5/5
        assert result.per_example_scores[1] == 0.5  # 5/10
        assert result.per_example_scores[2] == 1.0  # min(1, 6/3)

    def test_tool_call_f1(self):
        """Test tool call precision/recall/F1."""
        metric = ToolCallPrecisionRecallMetric()

        predictions = ["search, calc", "search", ""]
        references = ["search, calc, db", "search, calc", ""]

        result = metric.compute(predictions, references)

        # pred=2, ref=3, TP=2 -> P=1, R=2/3, F1=0.8
        assert 0.79 < result.per_example_scores[0] < 0.81
        # pred=1, ref=2, TP=1 -> P=1, R=0.5, F1=0.67
        assert 0.66 < result.per_example_scores[1] < 0.68
        assert result.per_example_scores[2] == 1.0  # Both empty

        # Check metadata
        assert "avg_precision" in result.metadata
        assert "avg_recall" in result.metadata


class TestDebateTypes:
    """Test debate data structures."""

    def test_argument_creation(self):
        """Test Argument dataclass."""
        arg = Argument(
            agent_id="agent1",
            role=DebateRole.PROPONENT,
            argument_type=ArgumentType.CLAIM,
            content="AI will transform education",
        )
        assert arg.agent_id == "agent1"
        assert arg.role == DebateRole.PROPONENT
        assert arg.argument_type == ArgumentType.CLAIM

    def test_debate_session_properties(self):
        """Test DebateSession computed properties."""
        args1 = [
            Argument("a1", DebateRole.PROPONENT, ArgumentType.CLAIM, "claim"),
            Argument("a2", DebateRole.OPPONENT, ArgumentType.COUNTER, "counter"),
        ]
        args2 = [
            Argument("a1", DebateRole.PROPONENT, ArgumentType.SUPPORT, "support"),
            Argument("a2", DebateRole.OPPONENT, ArgumentType.CONCESSION, "concede"),
        ]

        session = DebateSession(
            session_id="debate1",
            topic="AI in education",
            agents=["a1", "a2"],
            rounds=[
                DebateRound(0, args1),
                DebateRound(1, args2),
            ],
            consensus_reached=True,
        )

        assert session.num_rounds == 2
        assert session.num_arguments == 4
        assert len(session.all_arguments) == 4
        assert len(session.arguments_by_agent("a1")) == 2

        type_counts = session.argument_type_counts()
        assert type_counts[ArgumentType.CLAIM] == 1
        assert type_counts[ArgumentType.COUNTER] == 1

    def test_parse_debate_from_messages(self):
        """Test parsing debate from message format."""
        messages = [
            {
                "agent_id": "agent1",
                "debate_role": "proponent",
                "argument_type": "claim",
                "content": "AI improves learning",
            },
            {
                "agent_id": "agent2",
                "debate_role": "opponent",
                "argument_type": "counter",
                "content": "But reduces critical thinking",
            },
        ]

        session = parse_debate_from_messages(messages, "test-debate", "AI in education")

        assert session.session_id == "test-debate"
        assert session.topic == "AI in education"
        assert len(session.agents) == 2


class TestDebateMetrics:
    """Test debate evaluation metrics."""

    def test_consensus_reached(self):
        """Test consensus reached metric."""
        metric = ConsensusReachedMetric()

        predictions = ["consensus", "no_consensus", "yes", "false"]
        references = ["consensus", "no_consensus", "yes", "true"]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0
        assert result.per_example_scores[1] == 1.0
        assert result.per_example_scores[2] == 1.0
        assert result.per_example_scores[3] == 0.0  # Mismatch

    def test_argument_diversity(self):
        """Test argument diversity metric."""
        metric = ArgumentDiversityMetric()

        predictions = [
            "claim, support, counter",
            "claim",
            "claim, support, counter, concession, synthesis, clarification",
        ]
        references = [
            "claim, support, counter, concession, synthesis, clarification",
            "claim, support, counter, concession, synthesis, clarification",
            "claim, support, counter, concession, synthesis, clarification",
        ]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 0.5  # 3/6
        assert abs(result.per_example_scores[1] - 1 / 6) < 0.01  # 1/6
        assert result.per_example_scores[2] == 1.0  # 6/6

    def test_contribution_balance(self):
        """Test contribution balance using entropy."""
        metric = ContributionBalanceMetric()

        predictions = [
            "5, 5",  # Perfectly balanced
            "10, 0",  # Completely imbalanced
            "3, 3, 3",  # Three agents, balanced
        ]
        references = ["2", "2", "3"]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # Perfect balance
        assert result.per_example_scores[1] == 0.0  # All from one agent
        assert result.per_example_scores[2] == 1.0  # Perfect balance

    def test_debate_progression(self):
        """Test debate progression metric."""
        metric = DebateProgressionMetric()

        predictions = [
            "claim, support, counter",  # Valid progression
            "counter, claim, support",  # counter->claim invalid
            "claim",  # Single argument
        ]
        references = ["", "", ""]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # All valid transitions
        assert result.per_example_scores[1] < 1.0  # Some invalid
        assert result.per_example_scores[2] == 1.0  # Single = valid

    def test_debate_outcome_accuracy(self):
        """Test debate outcome accuracy metric."""
        metric = DebateOutcomeAccuracyMetric()

        predictions = ["agent1 wins", "tie", "correct", "totally wrong"]
        references = ["agent1 wins", "agent2 wins", "also correct", "correct"]

        result = metric.compute(predictions, references)

        assert result.per_example_scores[0] == 1.0  # Exact match
        assert result.per_example_scores[1] == 0.0  # Different outcomes
        assert result.per_example_scores[2] == 0.5  # Partial match (both have "correct")
        assert result.per_example_scores[3] == 0.0  # No match

    def test_argument_quality(self):
        """Test argument quality metric with component scores."""
        metric = ArgumentQualityMetric(
            relevance_weight=0.25,
            coherence_weight=0.25,
            evidence_weight=0.25,
            response_weight=0.25,
        )

        predictions = [
            "0.8, 0.9, 0.7, 0.8",  # Good scores
            '{"relevance": 1.0, "coherence": 1.0, "evidence": 1.0, "response": 1.0}',
            "0.5, 0.5, 0.5, 0.5",  # Average scores
        ]
        references = ["", "", ""]

        result = metric.compute(predictions, references)

        # (0.8+0.9+0.7+0.8)*0.25 = 0.8
        assert 0.79 < result.per_example_scores[0] < 0.81
        assert result.per_example_scores[1] == 1.0  # Perfect scores
        assert result.per_example_scores[2] == 0.5  # Average

        # Check component scores in metadata
        assert "component_scores" in result.metadata


class TestMetricRegistration:
    """Test that metrics are properly registered."""

    def test_trajectory_metrics_registered(self):
        """Test trajectory metrics are registered."""
        from spark_llm_eval.evaluation.base import get_metric, list_metrics

        available = list_metrics()

        assert "goal_completion" in available
        assert "trajectory_efficiency" in available
        assert "tool_call_accuracy" in available
        assert "action_sequence_f1" in available

    def test_tool_use_metrics_registered(self):
        """Test tool use metrics are registered."""
        from spark_llm_eval.evaluation.base import list_metrics

        available = list_metrics()

        assert "tool_selection_accuracy" in available
        assert "tool_order_accuracy" in available
        assert "tool_param_accuracy" in available
        assert "tool_efficiency" in available
        assert "tool_call_f1" in available

    def test_debate_metrics_registered(self):
        """Test debate metrics are registered."""
        from spark_llm_eval.evaluation.base import list_metrics

        available = list_metrics()

        assert "consensus_reached" in available
        assert "argument_diversity" in available
        assert "contribution_balance" in available
        assert "debate_progression" in available
        assert "debate_outcome_accuracy" in available
        assert "argument_quality" in available

    def test_get_metric_by_name(self):
        """Test getting metrics by name."""
        from spark_llm_eval.evaluation.base import get_metric

        goal_metric = get_metric("goal_completion")
        assert isinstance(goal_metric, GoalCompletionMetric)

        tool_metric = get_metric("tool_selection_accuracy")
        assert isinstance(tool_metric, ToolSelectionAccuracyMetric)

        debate_metric = get_metric("consensus_reached")
        assert isinstance(debate_metric, ConsensusReachedMetric)
