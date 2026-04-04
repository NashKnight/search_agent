"""
Abstract base class for LLM backends.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """All LLM backends must implement this interface."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple[list, str, str]:
        """Generate a response from a prompt.

        Returns:
            (token_ids, raw_text, clean_text)
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """Release any per-request cache (KV cache, etc.)."""
