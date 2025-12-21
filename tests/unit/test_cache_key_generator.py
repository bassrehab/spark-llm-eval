"""Tests for cache key generation."""

import pytest

from spark_llm_eval.cache.key_generator import generate_cache_key, generate_prompt_hash


class TestGenerateCacheKey:
    """Tests for generate_cache_key function."""

    def test_deterministic(self):
        """Same inputs should produce same key."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        assert key1 == key2

    def test_key_length(self):
        """Key should be 64-character SHA-256 hex digest."""
        key = generate_cache_key(
            prompt="Test prompt",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_prompts_different_keys(self):
        """Different prompts should produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="What is 3+3?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        assert key1 != key2

    def test_different_models_different_keys(self):
        """Different models should produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o-mini",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        assert key1 != key2

    def test_different_providers_different_keys(self):
        """Different providers should produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4",
            provider="azure",
            temperature=0.0,
            max_tokens=1024,
        )
        assert key1 != key2

    def test_different_temperature_different_keys(self):
        """Different temperatures should produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.7,
            max_tokens=1024,
        )
        assert key1 != key2

    def test_different_max_tokens_different_keys(self):
        """Different max_tokens should produce different keys."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=2048,
        )
        assert key1 != key2

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should be normalized."""
        key1 = generate_cache_key(
            prompt="What is 2+2?",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="  What is 2+2?  ",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        assert key1 == key2

    def test_line_ending_normalization(self):
        """Line endings should be normalized."""
        key1 = generate_cache_key(
            prompt="Line 1\nLine 2",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="Line 1\r\nLine 2",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
        )
        assert key1 == key2

    def test_float_precision_handling(self):
        """Temperature with different precision should produce same key."""
        key1 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.7,
            max_tokens=1024,
        )
        key2 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.7000,
            max_tokens=1024,
        )
        assert key1 == key2

    def test_extra_params_included(self):
        """Extra params should affect the key."""
        key1 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
            extra_params=None,
        )
        key2 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
            extra_params={"top_p": 0.9},
        )
        assert key1 != key2

    def test_extra_params_order_independent(self):
        """Extra params order should not affect the key."""
        key1 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
            extra_params={"top_p": 0.9, "seed": 42},
        )
        key2 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
            extra_params={"seed": 42, "top_p": 0.9},
        )
        assert key1 == key2

    def test_excluded_params_not_affect_key(self):
        """Excluded params (stream, user) should not affect key."""
        key1 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
            extra_params={"top_p": 0.9},
        )
        key2 = generate_cache_key(
            prompt="Test",
            model_name="gpt-4o",
            provider="openai",
            temperature=0.0,
            max_tokens=1024,
            extra_params={"top_p": 0.9, "stream": True, "user": "test-user"},
        )
        assert key1 == key2


class TestGeneratePromptHash:
    """Tests for generate_prompt_hash function."""

    def test_deterministic(self):
        """Same prompt should produce same hash."""
        hash1 = generate_prompt_hash("What is 2+2?")
        hash2 = generate_prompt_hash("What is 2+2?")
        assert hash1 == hash2

    def test_hash_length(self):
        """Hash should be 64-character SHA-256 hex digest."""
        hash_ = generate_prompt_hash("Test prompt")
        assert len(hash_) == 64
        assert all(c in "0123456789abcdef" for c in hash_)

    def test_different_prompts_different_hashes(self):
        """Different prompts should produce different hashes."""
        hash1 = generate_prompt_hash("What is 2+2?")
        hash2 = generate_prompt_hash("What is 3+3?")
        assert hash1 != hash2

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should be normalized."""
        hash1 = generate_prompt_hash("Test")
        hash2 = generate_prompt_hash("  Test  ")
        assert hash1 == hash2

    def test_line_ending_normalization(self):
        """Line endings should be normalized."""
        hash1 = generate_prompt_hash("Line 1\nLine 2")
        hash2 = generate_prompt_hash("Line 1\r\nLine 2")
        assert hash1 == hash2
