"""Tests for inference base classes."""

import pytest
from spark_llm_eval.inference.base import (
    InferenceRequest,
    InferenceResponse,
    InferenceEngine,
)


class TestInferenceRequest:
    """Tests for InferenceRequest."""

    def test_create_basic(self):
        req = InferenceRequest(
            prompt="What is 2+2?",
            request_id="req-001",
        )
        assert req.prompt == "What is 2+2?"
        assert req.request_id == "req-001"
        assert req.metadata == {}

    def test_create_with_metadata(self):
        req = InferenceRequest(
            prompt="Hello",
            request_id="req-002",
            metadata={"row_idx": 42, "category": "greeting"},
        )
        assert req.metadata["row_idx"] == 42


class TestInferenceResponse:
    """Tests for InferenceResponse."""

    def test_create_success_response(self):
        resp = InferenceResponse(
            request_id="req-001",
            text="4",
            input_tokens=10,
            output_tokens=1,
            latency_ms=150.0,
            cost_usd=0.0001,
        )
        assert resp.success is True
        assert resp.error is None
        assert resp.total_tokens == 11

    def test_create_error_response(self):
        resp = InferenceResponse(
            request_id="req-002",
            text=None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            cost_usd=0,
            error="Rate limit exceeded",
        )
        assert resp.success is False
        assert resp.error == "Rate limit exceeded"

    def test_total_tokens(self):
        resp = InferenceResponse(
            request_id="req-003",
            text="response",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200.0,
            cost_usd=0.001,
        )
        assert resp.total_tokens == 150


class ConcreteEngine(InferenceEngine):
    """Concrete implementation for testing."""

    def __init__(self):
        self._initialized = False
        self._shutdown = False

    def initialize(self) -> None:
        self._initialized = True

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            text=f"Response to: {request.prompt}",
            input_tokens=len(request.prompt) // 4,
            output_tokens=10,
            latency_ms=100.0,
            cost_usd=0.001,
            metadata=request.metadata,
        )

    def shutdown(self) -> None:
        self._shutdown = True

    @property
    def provider_name(self) -> str:
        return "test"


class TestInferenceEngine:
    """Tests for InferenceEngine abstract class."""

    def test_concrete_implementation(self):
        engine = ConcreteEngine()
        engine.initialize()

        req = InferenceRequest(prompt="Hello", request_id="test-1")
        resp = engine.infer(req)

        assert resp.request_id == "test-1"
        assert "Hello" in resp.text
        assert resp.success

        engine.shutdown()
        assert engine._shutdown

    def test_infer_batch_default(self):
        """Default batch implementation processes sequentially."""
        engine = ConcreteEngine()
        engine.initialize()

        requests = [
            InferenceRequest(prompt=f"Question {i}", request_id=f"req-{i}")
            for i in range(5)
        ]

        responses = engine.infer_batch(requests)

        assert len(responses) == 5
        for i, resp in enumerate(responses):
            assert resp.request_id == f"req-{i}"
            assert resp.success

    def test_estimate_tokens_default(self):
        """Default token estimation is roughly 4 chars per token."""
        engine = ConcreteEngine()

        # 100 chars should be roughly 25 tokens
        estimate = engine.estimate_tokens("a" * 100)
        assert 20 <= estimate <= 30

    def test_provider_name(self):
        engine = ConcreteEngine()
        assert engine.provider_name == "test"


class FailingEngine(InferenceEngine):
    """Engine that fails on some requests."""

    def initialize(self) -> None:
        pass

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if "fail" in request.prompt.lower():
            raise ValueError("Intentional failure")
        return InferenceResponse(
            request_id=request.request_id,
            text="OK",
            input_tokens=5,
            output_tokens=1,
            latency_ms=50.0,
            cost_usd=0.0001,
        )

    def shutdown(self) -> None:
        pass

    @property
    def provider_name(self) -> str:
        return "failing"


class TestBatchErrorHandling:
    """Test error handling in batch processing."""

    def test_batch_handles_individual_failures(self):
        """Batch doesn't fail entirely if one request fails."""
        engine = FailingEngine()
        engine.initialize()

        requests = [
            InferenceRequest(prompt="Good request 1", request_id="ok-1"),
            InferenceRequest(prompt="Please fail", request_id="fail-1"),
            InferenceRequest(prompt="Good request 2", request_id="ok-2"),
        ]

        responses = engine.infer_batch(requests)

        assert len(responses) == 3
        assert responses[0].success
        assert not responses[1].success
        assert "Intentional failure" in responses[1].error
        assert responses[2].success
