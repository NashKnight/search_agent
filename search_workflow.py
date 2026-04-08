"""
Search Agent workflow — the main entry point for running a single query
through the multi-round search + memory loop.

Architecture
------------
Round 1  : Initial analysis → extract queries → init memory → filter queue
Round 2+ : Pop query → search → analyse results → update memory → filter queue
Final    : Generate answer from memory (triggered when queue empties or max_rounds hit)
"""

import re
from collections import deque

from agent.memory import MemoryManager
from agent.prompts import (
    BASE_PROMPT,
    ANALYSIS_PROMPT,
    FILTER_QUERIES_PROMPT,
    FINAL_ANSWER_PROMPT,
    DIRECT_ANSWER_PROMPT,
)
from models.base import BaseLLM
from search.base import BaseSearch
from utils import load_config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _extract_search_queries(text: str) -> list[str]:
    """Return all <search>…</search> contents from model output."""
    pattern = r"<search>\s*([^<\n]{1,200}?)\s*</search>"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.IGNORECASE)]


def _clean_final(text: str) -> str:
    """Strip <think> and <search> tags from the final answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<search>.*?</search>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(?:think|search)>", "", text, flags=re.IGNORECASE)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _format_sources(sources: dict, used_sources: dict) -> str:
    """Format search result sources for injection into a prompt."""
    if not sources:
        return "[No results available]"
    lines = ["**Search results:**"]
    for key, val in sources.items():
        desc = val.get("description", "")[:300]
        url = val.get("url", "")
        title = val.get("title", "untitled")
        lines.append(f"{key} ({title}): {desc}")
        lines.append(f"  URL: {url}")
        if url and url not in used_sources:
            used_sources[url] = title
    return "\n".join(lines)


def _filter_queries(llm: BaseLLM, memory: str, candidates: list[str]) -> list[str]:
    """Ask the LLM to keep only queries relevant to the memory's [User Goal]."""
    if not candidates:
        return []
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(candidates))
    prompt = FILTER_QUERIES_PROMPT.format(memory=memory[:800], query_list=numbered)
    _, _, clean = llm.generate(prompt, max_new_tokens=1024)

    kept: list[str] = []
    for line in clean.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("KEEP"):
            q = re.sub(r"^KEEP\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
            # fuzzy-match back to original candidates
            for orig in candidates:
                if (q in orig or orig in q) and orig not in kept:
                    kept.append(orig)
                    break
    # Safety: if filter returns nothing, keep all
    return kept if kept else list(candidates)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

class SearchWorkflow:
    """Orchestrates the full multi-round search agent loop for one query."""

    def __init__(
        self,
        llm: BaseLLM,
        searcher: BaseSearch,
        config: dict | None = None,
    ):
        self._llm = llm
        self._searcher = searcher
        if config is None:
            config = load_config()
        limits = config["limits"]
        self._max_rounds: int = limits.get("max_rounds", 15)
        self._max_new_tokens: int = limits.get("max_new_tokens_default", 1536)
        self._max_final_tokens: int = limits.get("max_final_tokens", 8192)
        self._max_formatted_len: int = limits.get("max_formatted_sources_len", 4500)
        self._max_sources: int = limits.get("max_sources_per_search", 5)

    def run(self, user_query: str, max_rounds: int | None = None, log=print) -> dict:
        """Run the full search agent for *user_query*.

        Returns a result dict:
        {
            "answer": str,          # final clean answer
            "memory": str,          # final memory board state
            "used_sources": dict,   # {url: title}
            "rounds": list[dict],   # per-round trace
        }
        """
        max_rounds = max_rounds or self._max_rounds
        self._llm.clear_cache()

        # Extract <root_url> tag if present; strip it from the prompt query
        root_url: str | None = None
        m = re.search(r"<root_url>(https?://[^<\s]+)</root_url>", user_query)
        if m:
            root_url = m.group(1).strip()
        clean_query = re.sub(r"\s*<root_url>.*?</root_url>", "", user_query, flags=re.DOTALL).strip()
        root_domain: str | None = root_url.split("//")[-1].split("/")[0] if root_url else None

        memory_mgr = MemoryManager(self._llm)
        search_queue: deque[str] = deque()
        searched_queries: set[str] = set()
        search_history: list[dict] = []
        used_sources: dict[str, str] = {}
        rounds: list[dict] = []

        # ── Round 1: initial analysis ─────────────────────────────────────
        log("=" * 70)
        log("=== Round 1: Initial analysis ===")
        if root_url:
            log(f"[root_url detected] {root_url}")

        root_url_hint = (
            f"Reference website: {root_url} — include the site name or domain in your search queries.\n\n"
            if root_url else ""
        )
        init_prompt = (
            f"{BASE_PROMPT}\n\n"
            f"User question: {clean_query}\n\n"
            f"{root_url_hint}"
            "Quick decision:\n"
            "- Uncertain concept/term → <search> immediately\n"
            "- Multiple pieces of info needed → output ALL <search> tags at once\n"
            "- Can answer directly → no <search>, just answer\n"
            "- Each <search> on its own line, 3-10 words\n\n"
            "Begin:"
        )
        _, raw, clean = self._llm.generate(init_prompt, max_new_tokens=self._max_new_tokens)
        log(raw)

        buffer = _extract_search_queries(raw)
        log(f"\n[Extracted queries] {buffer}")

        entry: dict = {
            "round": 1,
            "raw": raw,
            "clean": clean,
            "extracted_queries": list(buffer),
        }

        if not buffer:
            log("[No search needed — generating direct answer]")
            direct_prompt = DIRECT_ANSWER_PROMPT.format(user_query=clean_query)
            _, _, direct_clean = self._llm.generate(direct_prompt, max_new_tokens=self._max_new_tokens)
            answer = _clean_final(direct_clean)
            entry["clean"] = answer
            entry["is_final"] = True
            entry["current_queue"] = []
            rounds.append(entry)
            return {
                "answer": answer,
                "memory": "",
                "used_sources": used_sources,
                "rounds": rounds,
            }
        memory = memory_mgr.initialize(clean_query, buffer)
        log(f"\n[Memory initialized]\n{memory}\n")

        filtered = _filter_queries(self._llm, memory, buffer)
        log(f"[Filtered queries] {filtered}")

        for q in filtered:
            if q not in searched_queries:
                search_queue.append(q)

        # Move root_url-related queries to the front of the queue
        if root_domain:
            priority = [q for q in search_queue if root_domain in q]
            for q in priority:
                search_queue.remove(q)
                search_queue.appendleft(q)
            if priority:
                log(f"[root_url queries prioritized] {priority}")

        entry["filtered_queries"] = filtered
        entry["current_queue"] = list(search_queue)
        entry["memory"] = memory
        rounds.append(entry)

        if not search_queue:
            return {"answer": _clean_final(clean), "memory": memory, "used_sources": used_sources, "rounds": rounds}

        # ── Round 2+: search loop ─────────────────────────────────────────
        round_num = 2
        while search_queue and round_num <= max_rounds:
            log(f"\n{'=' * 70}")
            log(f"=== Round {round_num} ===")
            log(f"[Memory]\n{memory[:800]}{'...' if len(memory) > 800 else ''}\n")

            current_query = search_queue.popleft()
            searched_queries.add(current_query)
            site_query = f"{current_query} site:{root_domain}" if root_domain else current_query
            log(f"[Searching] {site_query}")

            result = self._searcher.search(site_query, max_results=self._max_sources)
            round_entry: dict = {
                "round": round_num,
                "search_query": current_query,
                "search_result": result,
            }

            if result.get("error"):
                log(f"[Search error] {result['error']}")
                round_entry["error"] = result["error"]
                round_entry["current_queue"] = list(search_queue)
                round_entry["memory"] = memory
                rounds.append(round_entry)
                round_num += 1
                continue

            sources = result.get("sources", {})
            formatted = _format_sources(sources, used_sources)
            log(f"[Sources found] {len(sources)}")

            # URL queries are always considered relevant (forced visit — skip filter)
            relevance = _filter_queries(self._llm, memory, [current_query])
            is_relevant = bool(relevance)

            if not is_relevant:
                log("[Skipped — query no longer relevant]")
                round_entry.update({
                    "raw": "[Skipped: irrelevant]",
                    "clean": "",
                    "extracted_queries": [],
                    "filtered_queries": [],
                    "current_queue": list(search_queue),
                    "memory": memory,
                    "search_relevant": False,
                })
                rounds.append(round_entry)
                round_num += 1
                continue

            search_history.append({"query": current_query, "results": formatted})
            log("[Relevant — analysing results]")

            # Ask LLM to analyse new results against memory
            analysis_prompt = ANALYSIS_PROMPT.format(
                base_prompt=BASE_PROMPT,
                memory=memory,
                search_query=current_query,
                num_sources=len(sources),
                formatted_sources=formatted[: self._max_formatted_len],
                searched_count=len(searched_queries),
                queue_len=len(search_queue),
            )
            _, raw, clean = self._llm.generate(analysis_prompt, max_new_tokens=self._max_new_tokens)
            log(f"\n[Model response]\n{raw}")

            round_entry["raw"] = raw
            round_entry["clean"] = clean
            round_entry["formatted_sources"] = formatted[: self._max_formatted_len]

            new_queries = _extract_search_queries(raw)
            log(f"[New queries extracted] {new_queries}")

            # Merge queue + new queries, re-filter everything
            all_pending = list(search_queue) + new_queries
            # Temporarily update memory to help the filter judge new queries
            temp_memory = memory_mgr.update(
                search_history, all_pending, last_search_relevant=True
            )
            filtered_all = _filter_queries(self._llm, temp_memory, all_pending)
            log(f"[Filtered pending] {filtered_all}")

            search_queue.clear()
            for q in filtered_all:
                if q not in searched_queries:
                    search_queue.append(q)

            # Final memory update with the definitive queue
            memory = memory_mgr.update(
                search_history, list(search_queue), last_search_relevant=True
            )

            round_entry.update({
                "extracted_queries": new_queries,
                "filtered_queries": filtered_all,
                "current_queue": list(search_queue),
                "memory": memory,
                "search_relevant": True,
            })
            rounds.append(round_entry)

            # If queue is now empty and this round produced a complete answer
            if not search_queue and not new_queries:
                round_entry["is_final"] = True
                round_entry["clean"] = _clean_final(clean)
                log("\n[Queue empty — marking as final answer]")
                log(round_entry["clean"])
                return {
                    "answer": round_entry["clean"],
                    "memory": memory,
                    "used_sources": used_sources,
                    "rounds": rounds,
                }

            round_num += 1

        # ── Forced final answer generation ────────────────────────────────
        log(f"\n{'=' * 70}")
        log("=== Final answer generation ===")
        memory = memory_mgr.get()
        final_prompt = FINAL_ANSWER_PROMPT.format(
            memory=memory, user_query=clean_query
        )
        _, raw_final, clean_final = self._llm.generate(
            final_prompt, max_new_tokens=self._max_final_tokens
        )
        answer = _clean_final(clean_final)
        log(answer)

        rounds.append({
            "round": round_num,
            "raw": raw_final,
            "clean": answer,
            "is_final": True,
            "memory": memory,
        })

        return {
            "answer": answer,
            "memory": memory,
            "used_sources": used_sources,
            "rounds": rounds,
        }
