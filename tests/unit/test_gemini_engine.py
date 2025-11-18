"""Gemini engine tests."""

import pytest
from unittest.mock import MagicMock, patch

from spark_llm_eval.core.config import ModelConfig, ModelProvider
from spark_llm_eval.core.exceptions import InferenceError, RateLimitError
from spark_llm_eval.inference.base import InferenceRequest


@pytest.fixture
def model_config():
    return ModelConfig(provider=ModelProvider.GOOGLE, model_name="gemini-2.0-flash")


@pytest.fixture
def mock_genai():
    mock = MagicMock()

    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="STOP"))]

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock.GenerativeModel.return_value = mock_model
    return mock


class TestGeminiEngine:

    @patch("spark_llm_eval.inference.gemini_engine._import_genai")
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})
    def test_inference(self, mock_import, model_config, mock_genai):
        mock_import.return_value = mock_genai

        from spark_llm_eval.inference.gemini_engine import GeminiEngine
        engine = GeminiEngine(model_config)
        engine.initialize()

        resp = engine.infer(InferenceRequest(prompt="hi"))
        assert resp.text == "Test response"
        assert resp.input_tokens == 10

    @patch("spark_llm_eval.inference.gemini_engine._import_genai")
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})
    def test_batch(self, mock_import, model_config, mock_genai):
        mock_import.return_value = mock_genai

        from spark_llm_eval.inference.gemini_engine import GeminiEngine
        engine = GeminiEngine(model_config)
        engine.initialize()

        resps = engine.infer_batch([InferenceRequest(prompt=f"q{i}") for i in range(2)])
        assert len(resps) == 2
