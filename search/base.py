"""
Abstract base class for search backends.
"""

from abc import ABC, abstractmethod


class BaseSearch(ABC):
    """All search backends must implement this interface."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> dict:
        """Execute a search query and return structured results.

        Returns:
            {
                "sources": {
                    "source1": {"url": str, "title": str, "description": str},
                    ...
                },
                "error": str | None
            }
        """
