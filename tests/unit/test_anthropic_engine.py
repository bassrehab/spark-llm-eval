"""Tests for Anthropic engine."""

import pytest
from unittest.mock import MagicMock, patch

from spark_llm_eval.core.config import ModelConfig, ModelProvider
from spark_llm_eval.core.exceptions import InferenceError, RateLimitError
from spark_llm_eval.inference.base import InferenceRequest


@pytest.fixture
def model_config():
    return ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
    )


@pytest.fixture
def mock_anthropic():
    mock = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Test response")]
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 5
    mock_message.stop_reason = "end_turn"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    mock.Anthropic.return_value = mock_client
    mock.RateLimitError = type("RateLimitError", (Exception,), {})
    mock.APIError = type("APIError", (Exception,), {})
    return mock


class TestAnthropicEngine:

    @patch("spark_llm_eval.inference.anthropic_engine._load_anthropic")
    def test_init_and_infer(self, mock_load, model_config, mock_anthropic):
        mock_load.return_value = mock_anthropic

        from spark_llm_eval.inference.anthropic_engine import AnthropicEngine
        engine = AnthropicEngine(model_config, api_key="test-key")
        engine.initialize()

        resp = engine.infer(InferenceRequest(prompt="test", max_tokens=100))
        assert resp.text == "Test response"
        assert resp.input_tokens == 10

    @patch("spark_llm_eval.inference.anthropic_engine._load_anthropic")
    def test_rate_limit(self, mock_load, model_config, mock_anthropic):
        mock_load.return_value = mock_anthropic
        mock_anthropic.Anthropic().messages.create.side_effect = mock_anthropic.RateLimitError("limited")

        from spark_llm_eval.inference.anthropic_engine import AnthropicEngine
        engine = AnthropicEngine(model_config, api_key="test-key")
        engine.initialize()

        with pytest.raises(RateLimitError):
            engine.infer(InferenceRequest(prompt="test"))

    @patch("spark_llm_eval.inference.anthropic_engine._load_anthropic")
    def test_batch(self, mock_load, model_config, mock_anthropic):
        mock_load.return_value = mock_anthropic

        from spark_llm_eval.inference.anthropic_engine import AnthropicEngine
        engine = AnthropicEngine(model_config, api_key="test-key")
        engine.initialize()

        reqs = [InferenceRequest(prompt=f"q{i}", request_id=f"r{i}") for i in range(3)]
        resps = engine.infer_batch(reqs)
        assert len(resps) == 3

    @patch("spark_llm_eval.inference.anthropic_engine._load_anthropic")
    def test_token_tracking(self, mock_load, model_config, mock_anthropic):
        mock_load.return_value = mock_anthropic

        from spark_llm_eval.inference.anthropic_engine import AnthropicEngine
        engine = AnthropicEngine(model_config, api_key="test-key")
        engine.initialize()

        for _ in range(2):
            engine.infer(InferenceRequest(prompt="x"))

        inp, out = engine.total_tokens
        assert inp == 20
        assert out == 10
