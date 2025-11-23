"""Pytest configuration and fixtures for integration tests.

These tests require:
- Running Spark cluster (Docker or local)
- Valid API keys for LLM providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
"""

import os
import pytest
from typing import Generator

# Test configuration
TEST_SAMPLE_SIZE = 10  # Number of examples per test
SKIP_EXPENSIVE_TESTS = os.environ.get("SKIP_EXPENSIVE_TESTS", "false").lower() == "true"

# Models to use for testing (cheapest options)
TEST_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "google": "gemini-2.0-flash",
}

# Sample QA dataset for testing
SAMPLE_QA_DATA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What color is the sky on a clear day?", "answer": "Blue"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "How many days are in a week?", "answer": "7"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What is the speed of light in a vacuum?", "answer": "299,792,458 meters per second"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the smallest prime number?", "answer": "2"},
]


def get_spark_master() -> str:
    """Get Spark master URL from environment or default."""
    return os.environ.get("SPARK_MASTER", "local[2]")


def has_api_key(provider: str) -> bool:
    """Check if API key is available for provider."""
    key_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    key_name = key_names.get(provider)
    return key_name and os.environ.get(key_name) is not None


@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for integration tests.

    Uses Docker cluster if SPARK_MASTER is set, otherwise local mode.
    """
    from pyspark.sql import SparkSession

    master = get_spark_master()

    spark = (
        SparkSession.builder
        .appName("spark-llm-eval-integration-tests")
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    yield spark

    spark.stop()


@pytest.fixture(scope="session")
def sample_qa_df(spark_session):
    """Create a sample QA DataFrame for testing."""
    return spark_session.createDataFrame(SAMPLE_QA_DATA)


@pytest.fixture(scope="module")
def temp_delta_path(tmp_path_factory) -> str:
    """Create a temporary directory for Delta tables."""
    return str(tmp_path_factory.mktemp("delta_tables"))


@pytest.fixture
def openai_config():
    """Create OpenAI model configuration."""
    from spark_llm_eval.core.config import ModelConfig, ModelProvider

    if not has_api_key("openai"):
        pytest.skip("OPENAI_API_KEY not set")

    return ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name=TEST_MODELS["openai"],
        api_key_secret="OPENAI_API_KEY",
    )


@pytest.fixture
def anthropic_config():
    """Create Anthropic model configuration."""
    from spark_llm_eval.core.config import ModelConfig, ModelProvider

    if not has_api_key("anthropic"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    return ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name=TEST_MODELS["anthropic"],
        api_key_secret="ANTHROPIC_API_KEY",
    )


@pytest.fixture
def google_config():
    """Create Google Gemini model configuration."""
    from spark_llm_eval.core.config import ModelConfig, ModelProvider

    if not has_api_key("google"):
        pytest.skip("GOOGLE_API_KEY not set")

    return ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name=TEST_MODELS["google"],
        api_key_secret="GOOGLE_API_KEY",
    )


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "expensive: marks tests that incur API costs")
    config.addinivalue_line("markers", "openai: marks tests requiring OpenAI API")
    config.addinivalue_line("markers", "anthropic: marks tests requiring Anthropic API")
    config.addinivalue_line("markers", "google: marks tests requiring Google API")
    config.addinivalue_line("markers", "delta: marks tests requiring Delta Lake")
