"""
Memory manager — maintains the Dynamic Memory across search rounds.

Dynamic Memory replaces the original Chinese "全局信息板" (global info board)
and tracks: Global Query, Task Plan, History Information, and Pending Queue.
"""

import re

from models.base import BaseLLM
from utils import load_config
from agent.prompts import (
    MEMORY_INIT_PROMPT,
    MEMORY_UPDATE_PROMPT,
    MEMORY_UPDATE_QUEUE_ONLY_PROMPT,
)


class MemoryManager:
    """Maintains and updates the Dynamic Memory throughout a search session."""

    def __init__(self, llm: BaseLLM, config: dict | None = None):
        if config is None:
            config = load_config()
        self._llm = llm
        self._max_tokens: int = config["limits"].get("max_memory_tokens", 1500)
        self.memory: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, user_query: str, pending_queries: list[str]) -> str:
        """[Req 2] Bootstrap — create an empty Dynamic Memory for a new search session."""
        prompt = MEMORY_INIT_PROMPT.format(
            user_query=user_query,
            pending_queries=", ".join(pending_queries) if pending_queries else "none",
        )
        _, _, raw = self._llm.generate(prompt, max_new_tokens=self._max_tokens)
        self.memory = self._clean(raw)
        return self.memory

    def update(
        self,
        search_history: list[dict],
        pending_queue: list[str],
        last_search_relevant: bool = True,
    ) -> str:
        """[Req 2] Memory Update — update the Dynamic Memory after a search round.

        Args:
            search_history: All search records so far, each ``{"query": str, "results": str}``.
            pending_queue:  Queries still waiting to be searched.
            last_search_relevant: If False, only refresh the pending queue section.
        """
        queue_str = ", ".join(pending_queue[:10]) if pending_queue else "empty"

        if not last_search_relevant or not search_history:
            prompt = MEMORY_UPDATE_QUEUE_ONLY_PROMPT.format(
                memory=self.memory,
                pending_queue=queue_str,
            )
        else:
            last = search_history[-1]
            prompt = MEMORY_UPDATE_PROMPT.format(
                memory=self.memory,
                search_query=last.get("query", ""),
                search_results=last.get("results", "")[:800],
                pending_queue=queue_str,
            )

        _, _, raw = self._llm.generate(prompt, max_new_tokens=self._max_tokens)
        self.memory = self._clean(raw)
        return self.memory

    def get(self) -> str:
        return self.memory

    def reset(self) -> None:
        self.memory = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        return text.strip()
