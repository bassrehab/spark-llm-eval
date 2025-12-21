"""Content-addressable cache key generation."""

import hashlib
import json
from typing import Any


def generate_cache_key(
    prompt: str,
    model_name: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Generate a content-addressable cache key.

    Creates a deterministic SHA-256 hash from the prompt and model configuration.
    Same inputs always produce the same key, enabling cache lookups.

    Args:
        prompt: The input prompt text
        model_name: Model identifier (e.g., "gpt-4o", "claude-3-opus")
        provider: Provider name (e.g., "openai", "anthropic")
        temperature: Sampling temperature (affects randomness)
        max_tokens: Maximum output tokens
        extra_params: Additional model parameters affecting output

    Returns:
        64-character hex string (SHA-256 digest)

    Example:
        >>> key = generate_cache_key(
        ...     prompt="What is 2+2?",
        ...     model_name="gpt-4o",
        ...     provider="openai",
        ...     temperature=0.0,
        ...     max_tokens=1024,
        ... )
        >>> len(key)
        64
    """
    # Normalize prompt: strip whitespace and normalize line endings
    normalized_prompt = prompt.strip().replace("\r\n", "\n")

    # Build canonical representation
    key_data: dict[str, Any] = {
        "prompt": normalized_prompt,
        "model": model_name,
        "provider": provider,
        "temperature": round(temperature, 4),  # Avoid float precision issues
        "max_tokens": max_tokens,
    }

    # Include extra_params if they affect output
    if extra_params:
        # Sort keys for determinism, exclude non-deterministic params
        filtered = {
            k: v
            for k, v in sorted(extra_params.items())
            if k not in ("stream", "user", "request_id")
        }
        if filtered:
            key_data["extra"] = filtered

    # Generate hash from canonical JSON
    canonical = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_prompt_hash(prompt: str) -> str:
    """Generate a hash of just the prompt text.

    Useful for grouping responses by prompt across different models.

    Args:
        prompt: The input prompt text

    Returns:
        64-character hex string (SHA-256 digest)
    """
    normalized = prompt.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
