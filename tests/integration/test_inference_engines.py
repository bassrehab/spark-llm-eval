"""Integration tests for inference engines with real API calls.

These tests make actual API calls and incur costs.
Run with: pytest tests/integration/test_inference_engines.py -v
"""

import pytest
import os

from spark_llm_eval.inference.base import InferenceRequest, InferenceResponse


class TestOpenAIEngine:
    """Integration tests for OpenAI inference engine."""

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_openai_single_inference(self, openai_config):
        """Test single inference with OpenAI."""
        from spark_llm_eval.inference import OpenAIInferenceEngine

        engine = OpenAIInferenceEngine(openai_config)
        engine.initialize()

        try:
            request = InferenceRequest(
                prompt="What is 2 + 2? Answer with just the number.",
                max_tokens=10,
                temperature=0.0,
            )

            response = engine.infer(request)

            assert isinstance(response, InferenceResponse)
            assert response.text is not None
            assert "4" in response.text
            assert response.input_tokens > 0
            assert response.output_tokens > 0
            assert response.latency_ms > 0
            assert openai_config.model_name in response.model  # OpenAI returns full model name with date
            # finish_reason may be None or "stop" depending on implementation
        finally:
            engine.shutdown()

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_openai_batch_inference(self, openai_config):
        """Test batch inference with OpenAI."""
        from spark_llm_eval.inference import OpenAIInferenceEngine

        engine = OpenAIInferenceEngine(openai_config)
        engine.initialize()

        try:
            requests = [
                InferenceRequest(
                    prompt="What is the capital of France? Answer in one word.",
                    max_tokens=10,
                    temperature=0.0,
                ),
                InferenceRequest(
                    prompt="What is the capital of Germany? Answer in one word.",
                    max_tokens=10,
                    temperature=0.0,
                ),
            ]

            responses = engine.infer_batch(requests)

            assert len(responses) == 2
            assert "Paris" in responses[0].text
            assert "Berlin" in responses[1].text
        finally:
            engine.shutdown()

    @pytest.mark.openai
    @pytest.mark.expensive
    def test_openai_token_tracking(self, openai_config):
        """Test that token usage is tracked correctly."""
        from spark_llm_eval.inference import OpenAIInferenceEngine

        engine = OpenAIInferenceEngine(openai_config)
        engine.initialize()

        try:
            # Check if total_tokens property exists
            if not hasattr(engine, 'total_tokens'):
                pytest.skip("OpenAIInferenceEngine does not have total_tokens property")

            initial_tokens = engine.total_tokens

            request = InferenceRequest(
                prompt="Say 'hello'",
                max_tokens=10,
                temperature=0.0,
            )
            engine.infer(request)

            final_tokens = engine.total_tokens
            assert final_tokens[0] > initial_tokens[0]  # input tokens increased
            assert final_tokens[1] > initial_tokens[1]  # output tokens increased
        finally:
            engine.shutdown()


class TestAnthropicEngine:
    """Integration tests for Anthropic inference engine."""

    @pytest.mark.anthropic
    @pytest.mark.expensive
    def test_anthropic_single_inference(self, anthropic_config):
        """Test single inference with Anthropic Claude."""
        from spark_llm_eval.inference import AnthropicEngine

        engine = AnthropicEngine(anthropic_config)
        engine.initialize()

        try:
            request = InferenceRequest(
                prompt="What is 2 + 2? Answer with just the number.",
                max_tokens=10,
                temperature=0.0,
            )

            response = engine.infer(request)

            assert isinstance(response, InferenceResponse)
            assert response.text is not None
            assert "4" in response.text
            assert response.input_tokens > 0
            assert response.output_tokens > 0
            assert response.latency_ms > 0
            assert response.finish_reason == "end_turn"
        finally:
            engine.shutdown()

    @pytest.mark.anthropic
    @pytest.mark.expensive
    def test_anthropic_batch_inference(self, anthropic_config):
        """Test batch inference with Anthropic."""
        from spark_llm_eval.inference import AnthropicEngine

        engine = AnthropicEngine(anthropic_config)
        engine.initialize()

        try:
            requests = [
                InferenceRequest(
                    prompt="What is the capital of France? Answer in one word.",
                    max_tokens=10,
                    temperature=0.0,
                ),
                InferenceRequest(
                    prompt="What is the capital of Germany? Answer in one word.",
                    max_tokens=10,
                    temperature=0.0,
                ),
            ]

            responses = engine.infer_batch(requests)

            assert len(responses) == 2
            assert "Paris" in responses[0].text
            assert "Berlin" in responses[1].text
        finally:
            engine.shutdown()

    @pytest.mark.anthropic
    @pytest.mark.expensive
    def test_anthropic_cost_tracking(self, anthropic_config):
        """Test that cost is tracked correctly."""
        from spark_llm_eval.inference import AnthropicEngine

        engine = AnthropicEngine(anthropic_config)
        engine.initialize()

        try:
            request = InferenceRequest(
                prompt="Say 'hello'",
                max_tokens=10,
                temperature=0.0,
            )
            engine.infer(request)

            assert engine.total_cost > 0
        finally:
            engine.shutdown()


class TestGeminiEngine:
    """Integration tests for Google Gemini inference engine."""

    @pytest.mark.google
    @pytest.mark.expensive
    def test_gemini_single_inference(self, google_config):
        """Test single inference with Gemini."""
        from spark_llm_eval.inference import GeminiEngine

        engine = GeminiEngine(google_config)
        engine.initialize()

        try:
            request = InferenceRequest(
                prompt="What is 2 + 2? Answer with just the number.",
                max_tokens=10,
                temperature=0.0,
            )

            response = engine.infer(request)

            assert isinstance(response, InferenceResponse)
            assert response.text is not None
            assert "4" in response.text
            assert response.latency_ms > 0
            assert response.model == google_config.model_name
        finally:
            engine.shutdown()

    @pytest.mark.google
    @pytest.mark.expensive
    def test_gemini_batch_inference(self, google_config):
        """Test batch inference with Gemini."""
        from spark_llm_eval.inference import GeminiEngine

        engine = GeminiEngine(google_config)
        engine.initialize()

        try:
            requests = [
                InferenceRequest(
                    prompt="What is the capital of France? Answer in one word.",
                    max_tokens=10,
                    temperature=0.0,
                ),
                InferenceRequest(
                    prompt="What is the capital of Germany? Answer in one word.",
                    max_tokens=10,
                    temperature=0.0,
                ),
            ]

            responses = engine.infer_batch(requests)

            assert len(responses) == 2
            assert "Paris" in responses[0].text
            assert "Berlin" in responses[1].text
        finally:
            engine.shutdown()

    @pytest.mark.google
    @pytest.mark.expensive
    def test_gemini_provider_name(self, google_config):
        """Test provider name is correct."""
        from spark_llm_eval.inference import GeminiEngine

        engine = GeminiEngine(google_config)
        assert engine.provider_name == "google"


class TestEngineFactory:
    """Test the engine factory function."""

    @pytest.mark.openai
    def test_create_openai_engine(self, openai_config):
        """Test creating OpenAI engine via factory."""
        from spark_llm_eval.inference import create_engine

        engine = create_engine(openai_config)
        assert engine is not None
        assert engine.provider_name == "openai"

    @pytest.mark.anthropic
    def test_create_anthropic_engine(self, anthropic_config):
        """Test creating Anthropic engine via factory."""
        from spark_llm_eval.inference import create_engine

        engine = create_engine(anthropic_config)
        assert engine is not None
        assert engine.provider_name == "anthropic"

    @pytest.mark.google
    def test_create_google_engine(self, google_config):
        """Test creating Google engine via factory."""
        from spark_llm_eval.inference import create_engine

        engine = create_engine(google_config)
        assert engine is not None
        assert engine.provider_name == "google"


class TestCrossProviderComparison:
    """Test comparing outputs across providers."""

    @pytest.mark.openai
    @pytest.mark.anthropic
    @pytest.mark.google
    @pytest.mark.expensive
    @pytest.mark.slow
    def test_all_providers_same_question(self, openai_config, anthropic_config, google_config):
        """Test that all providers can answer the same question."""
        from spark_llm_eval.inference import create_engine

        engines = []
        responses = {}

        # Create and initialize all engines
        for name, config in [
            ("openai", openai_config),
            ("anthropic", anthropic_config),
            ("google", google_config),
        ]:
            try:
                engine = create_engine(config)
                engine.initialize()
                engines.append((name, engine))
            except Exception as e:
                pytest.skip(f"Could not initialize {name}: {e}")

        try:
            request = InferenceRequest(
                prompt="What is 2 + 2? Answer with just the number.",
                max_tokens=10,
                temperature=0.0,
            )

            for name, engine in engines:
                response = engine.infer(request)
                responses[name] = response

            # All providers should return "4"
            for name, response in responses.items():
                assert "4" in response.text, f"{name} did not return '4': {response.text}"
        finally:
            for _, engine in engines:
                engine.shutdown()
