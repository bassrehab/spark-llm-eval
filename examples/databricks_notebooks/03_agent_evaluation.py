# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Evaluation
# MAGIC
# MAGIC This notebook demonstrates how to evaluate agent behaviors including:
# MAGIC - Multi-turn conversation trajectories
# MAGIC - Tool use accuracy
# MAGIC - Goal completion rates
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks Runtime 14.0+ with ML
# MAGIC - spark-llm-eval installed

# COMMAND ----------

# MAGIC %pip install spark-llm-eval
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Agent Evaluation Components

# COMMAND ----------

from spark_llm_eval.agents import (
    # Trajectory types
    Trajectory,
    Turn,
    TurnRole,
    Action,
    ActionType,
    Observation,
    TrajectoryPair,
    # Trajectory metrics
    GoalCompletionMetric,
    TrajectoryEfficiencyMetric,
    ActionSequenceF1Metric,
    # Tool use types
    ToolCall,
    ToolCallSequence,
    # Tool use metrics
    ToolSelectionAccuracyMetric,
    ToolOrderAccuracyMetric,
    ToolCallPrecisionRecallMetric,
    # Utilities
    parse_trajectory_from_messages,
    parse_tool_calls_from_messages,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Sample Agent Trajectories
# MAGIC
# MAGIC We'll create sample trajectories representing agent interactions with tools.

# COMMAND ----------

# Example: Agent helping user find weather information
trajectory_1 = Trajectory(
    trajectory_id="traj-001",
    turns=[
        Turn(
            role=TurnRole.USER,
            content="What's the weather like in New York?",
            turn_index=0,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="I'll check the weather for you.",
            action=Action(
                action_type=ActionType.TOOL_CALL,
                content="get_weather",
                parameters={"location": "New York"},
            ),
            turn_index=1,
        ),
        Turn(
            role=TurnRole.TOOL,
            content="72F, sunny, humidity 45%",
            observation=Observation(content="72F, sunny, humidity 45%", success=True),
            turn_index=2,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="The weather in New York is 72F and sunny with 45% humidity.",
            action=Action(action_type=ActionType.RESPONSE, content="The weather in New York is 72F and sunny with 45% humidity."),
            turn_index=3,
        ),
    ],
    initial_goal="Get weather information for New York",
    goal_achieved=True,
)

# Example: Agent with multiple tool calls
trajectory_2 = Trajectory(
    trajectory_id="traj-002",
    turns=[
        Turn(
            role=TurnRole.USER,
            content="Book me a flight from NYC to LA for tomorrow and find a hotel.",
            turn_index=0,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="I'll search for flights first.",
            action=Action(
                action_type=ActionType.TOOL_CALL,
                content="search_flights",
                parameters={"from": "NYC", "to": "LA", "date": "tomorrow"},
            ),
            turn_index=1,
        ),
        Turn(
            role=TurnRole.TOOL,
            content="Found 5 flights. Cheapest: $299 departing 8am",
            observation=Observation(content="Found 5 flights", success=True),
            turn_index=2,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="Now searching for hotels.",
            action=Action(
                action_type=ActionType.TOOL_CALL,
                content="search_hotels",
                parameters={"location": "LA", "checkin": "tomorrow"},
            ),
            turn_index=3,
        ),
        Turn(
            role=TurnRole.TOOL,
            content="Found 10 hotels. Best rated: Hilton Downtown $189/night",
            observation=Observation(content="Found 10 hotels", success=True),
            turn_index=4,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="I found flights starting at $299 and hotels from $189/night.",
            action=Action(action_type=ActionType.RESPONSE, content="Flight and hotel options found."),
            turn_index=5,
        ),
    ],
    initial_goal="Book flight and find hotel",
    goal_achieved=True,
)

# Example: Failed trajectory
trajectory_3 = Trajectory(
    trajectory_id="traj-003",
    turns=[
        Turn(
            role=TurnRole.USER,
            content="What's the stock price of INVALID_TICKER?",
            turn_index=0,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="Looking up stock price.",
            action=Action(
                action_type=ActionType.TOOL_CALL,
                content="get_stock_price",
                parameters={"ticker": "INVALID_TICKER"},
            ),
            turn_index=1,
        ),
        Turn(
            role=TurnRole.TOOL,
            content="Error: Ticker not found",
            observation=Observation(content="Error: Ticker not found", success=False, error="Ticker not found"),
            turn_index=2,
        ),
        Turn(
            role=TurnRole.ASSISTANT,
            content="I couldn't find that stock ticker. Please check the symbol.",
            action=Action(action_type=ActionType.RESPONSE, content="Error response"),
            turn_index=3,
        ),
    ],
    initial_goal="Get stock price",
    goal_achieved=False,
)

trajectories = [trajectory_1, trajectory_2, trajectory_3]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluate Goal Completion

# COMMAND ----------

goal_metric = GoalCompletionMetric()

# Create trajectory pairs (no reference trajectories in this example)
trajectory_pairs = [TrajectoryPair(predicted=t) for t in trajectories]

result = goal_metric.compute(trajectory_pairs, None)

print("Goal Completion Results:")
print(f"  Overall rate: {result.value:.2%}")
print(f"  Per-trajectory: {result.per_example_scores}")

for i, traj in enumerate(trajectories):
    status = "Achieved" if result.per_example_scores[i] == 1.0 else "Not achieved"
    print(f"  {traj.trajectory_id}: {status} - {traj.initial_goal}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate Trajectory Efficiency

# COMMAND ----------

efficiency_metric = TrajectoryEfficiencyMetric(max_turns=10)

result = efficiency_metric.compute(trajectory_pairs, None)

print("Trajectory Efficiency Results:")
print(f"  Average efficiency: {result.value:.2%}")
print()
for i, traj in enumerate(trajectories):
    print(f"  {traj.trajectory_id}:")
    print(f"    Turns: {traj.num_turns}")
    print(f"    Tool calls: {traj.num_tool_calls}")
    print(f"    Efficiency score: {result.per_example_scores[i]:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate Tool Selection

# COMMAND ----------

# For tool metrics, we compare predicted vs expected tool sequences
# Format: comma-separated tool names

predicted_tools = [
    "get_weather",
    "search_flights, search_hotels",
    "get_stock_price",
]

expected_tools = [
    "get_weather",
    "search_flights, search_hotels",
    "get_stock_price",
]

tool_selection_metric = ToolSelectionAccuracyMetric()
result = tool_selection_metric.compute(predicted_tools, expected_tools)

print("Tool Selection Accuracy:")
print(f"  Overall accuracy: {result.value:.2%}")
for i, score in enumerate(result.per_example_scores):
    print(f"  Trajectory {i+1}: {score:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate Tool Order

# COMMAND ----------

# Check if tools were called in the correct order
tool_order_metric = ToolOrderAccuracyMetric()

predicted_order = [
    "get_weather",
    "search_flights, search_hotels",
    "get_stock_price",
]

expected_order = [
    "get_weather",
    "search_hotels, search_flights",  # Different order
    "get_stock_price",
]

result = tool_order_metric.compute(predicted_order, expected_order)

print("Tool Order Accuracy:")
print(f"  Overall accuracy: {result.value:.2%}")
for i, score in enumerate(result.per_example_scores):
    print(f"  Trajectory {i+1}: {score:.2%}")
    print(f"    Predicted: {predicted_order[i]}")
    print(f"    Expected:  {expected_order[i]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tool Call Precision/Recall/F1

# COMMAND ----------

tool_f1_metric = ToolCallPrecisionRecallMetric()

predicted = [
    "get_weather",
    "search_flights, search_hotels, book_flight",  # Extra: book_flight
    "get_stock_price",
]

expected = [
    "get_weather",
    "search_flights, search_hotels",
    "get_stock_price, get_company_info",  # Missing: get_company_info
]

result = tool_f1_metric.compute(predicted, expected)

print("Tool Call F1 Analysis:")
print(f"  Average F1: {result.value:.2%}")
print(f"  Average Precision: {result.metadata['avg_precision']:.2%}")
print(f"  Average Recall: {result.metadata['avg_recall']:.2%}")
print()
for i, score in enumerate(result.per_example_scores):
    print(f"  Trajectory {i+1}: F1={score:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Parse Trajectories from Chat Messages
# MAGIC
# MAGIC If you have agent logs in standard chat format, you can parse them.

# COMMAND ----------

# Example: Parse from OpenAI-style messages
messages = [
    {"role": "user", "content": "Search for Python tutorials"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "Python tutorials"}',
                }
            }
        ]
    },
    {"role": "tool", "tool_call_id": "call_123", "content": "Found 100 results"},
    {"role": "assistant", "content": "I found 100 Python tutorials for you."},
]

parsed_trajectory = parse_trajectory_from_messages(messages, trajectory_id="parsed-001")

print("Parsed Trajectory:")
print(f"  ID: {parsed_trajectory.trajectory_id}")
print(f"  Initial goal: {parsed_trajectory.initial_goal}")
print(f"  Turns: {parsed_trajectory.num_turns}")
print(f"  Tool calls: {parsed_trajectory.num_tool_calls}")

for turn in parsed_trajectory.turns:
    print(f"\n  Turn {turn.turn_index} ({turn.role.value}):")
    print(f"    Content: {turn.content[:50]}..." if len(turn.content) > 50 else f"    Content: {turn.content}")
    if turn.action:
        print(f"    Action: {turn.action.action_type.value} - {turn.action.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Scale with Spark
# MAGIC
# MAGIC For large-scale evaluation, convert trajectories to DataFrames.

# COMMAND ----------

# Create a DataFrame of trajectory metrics for analysis
trajectory_data = []

for traj in trajectories:
    trajectory_data.append({
        "trajectory_id": traj.trajectory_id,
        "initial_goal": traj.initial_goal,
        "num_turns": traj.num_turns,
        "num_tool_calls": traj.num_tool_calls,
        "goal_achieved": traj.goal_achieved,
        "tools_used": ",".join([tc.content for tc in traj.tool_calls]),
    })

traj_df = spark.createDataFrame(trajectory_data)
display(traj_df)

# COMMAND ----------

# Aggregate statistics
print("Aggregate Statistics:")
traj_df.groupBy("goal_achieved").agg(
    F.count("*").alias("count"),
    F.avg("num_turns").alias("avg_turns"),
    F.avg("num_tool_calls").alias("avg_tool_calls"),
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC - Creating and evaluating agent trajectories
# MAGIC - Measuring goal completion rates
# MAGIC - Evaluating trajectory efficiency
# MAGIC - Assessing tool selection and ordering accuracy
# MAGIC - Parsing trajectories from chat message formats
# MAGIC - Scaling agent evaluation with Spark DataFrames
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Integrate with your agent logging system
# MAGIC - Create reference trajectories for comparison
# MAGIC - Build dashboards for monitoring agent performance

# COMMAND ----------

# Import for Spark SQL functions used above
from pyspark.sql import functions as F
