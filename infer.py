"""
MA-HReAct: Memory-Augmented Hierarchical ReAct
===============================================
Multi-call architecture for multi-hop reasoning over a JSONL benchmark.

Each round decomposes into distinct LLM sub-requests with single responsibilities:

  Phase 1  Init
           Decompose the question into goal, plan, and initial tool_call queue.

  Phase 2  Execution loop (per round):
    2.1    Analysis + Compression
           Input:  search observation + current memory
           Output: analysis, <new_compressed_history> (mandatory), optional
                   <tool_call> queries or <finish>
    2.2    Queue Filter
           Input:  updated memory (with new compressed_history and pending queue)
           Output: pruned pending queue — removes duplicates, already-searched
                   items, and queries irrelevant to the goal/plan

  Phase 3  Synthesis
           Triggered when queue empties or max_rounds is reached.
           Generates the final answer from full memory.

Termination: model outputs <finish> in 2.1, OR queue empties after 2.2,
             OR max_rounds is exceeded.

Usage
-----
    python infer_6.0.py [--config config.yaml] [--port 6001] [--workers 4] \\
                        [--benchmark data.jsonl] [--output results.jsonl]
                        [--rollouts 3]
                        [--onetime]
                        [--max-rounds N]
"""

import argparse
import json
import re
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Allow execution from processing_scripts/ as well as search_agent/
_agent_dir = Path(__file__).resolve().parent.parent / "search_agent"
if str(_agent_dir) not in sys.path:
    sys.path.insert(0, str(_agent_dir))

from models.vllm_server_model import VLLMServerModel
from search.jina_search import JinaSearch
from utils import load_config


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INIT_PROMPT = """\
You are a research assistant. Analyze the question and produce a structured research plan.

Question: {question}

Output the following tags in order:
<goal>one-sentence research objective</goal>
<plan>
1. first step
2. second step
...
</plan>
<tool_call>first search query (3-10 words)</tool_call>
<tool_call>second search query (3-10 words)</tool_call>

If the question requires no search at all, replace <tool_call> with:
<finish>complete answer here</finish>
"""

ANALYSIS_PROMPT = """\
{memory}

---
Search result for query: "{current_query}"

{observation}
---

Analyze the result in the context of your research plan and memory.

Output:
1. Your analysis (free text).
2. A <new_compressed_history> tag (required):
   <new_compressed_history>Q: {current_query} | A: key facts in 1-2 sentences</new_compressed_history>
3. Either additional queries OR a finish signal:
   <tool_call>new query if more information is needed</tool_call>
   ...or...
   <finish>complete answer to the original question</finish>
"""

FILTER_PROMPT = """\
{memory}

---
Review the Pending Queue listed above. For each query, decide whether to keep or remove it.

Remove a query if it:
- Is already answered in Search History
- Is irrelevant to the Goal or Plan
- Duplicates another query in the queue (keep only the more specific one)

Output one line per pending query, using exact original text:
<keep>query text</keep>
<remove>query text</remove>

If all queries should be removed, output: <queue_empty/>
"""

SYNTHESIS_PROMPT = """\
{memory}

---
Original question: {question}

Based on all information in the Search History above, provide a complete and accurate answer.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return slug[:max_len].strip("_")


def _build_query(record: dict) -> str:
    question = record["question"]
    root_url = record.get("root_url", "")
    lang     = record.get("info", {}).get("lang", "")
    query    = question
    if root_url:
        query += f"\n<root_url>{root_url}</root_url>"
    if lang == "en":
        query += "\n[Language requirement: Your final answer MUST be written in English.]"
    elif lang == "zh":
        query += "\n[Language requirement: Your final answer MUST be written in Chinese.]"
    return query


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _format_observation(sources: dict, used_sources: dict) -> str:
    """Format Jina search results into a plain-text observation string."""
    if not sources:
        return "[No results found]"
    lines: list[str] = []
    for val in sources.values():
        desc  = val.get("description", "")
        url   = val.get("url", "")
        title = val.get("title", "untitled")
        lines.append(f"- {title}: {desc}")
        lines.append(f"  URL: {url}")
        if url and url not in used_sources:
            used_sources[url] = title
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _new_memory() -> dict:
    return {
        "global_query":       "",
        "plan":               "",
        "compressed_history": [],   # list of {"query": str, "key_fact": str}
        "pending_queue":      deque(),
        "searched_queries":   set(),
    }


def _format_memory(memory: dict) -> str:
    lines = [
        "[Goal]",
        memory["global_query"] or "(not set)",
        "",
        "[Research Plan]",
        memory["plan"] or "(not set)",
        "",
        "[Search History]",
    ]
    if memory["compressed_history"]:
        for item in memory["compressed_history"]:
            lines.append(f"- Q: {item['query']} | A: {item['key_fact']}")
    else:
        lines.append("(empty)")
    lines += ["", "[Pending Queue]"]
    if memory["pending_queue"]:
        for q in memory["pending_queue"]:
            lines.append(f"- {q}")
    else:
        lines.append("(empty)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _parse_all_tags(text: str, tag: str) -> list[str]:
    return [
        m.group(1).strip()
        for m in re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    ]


def _parse_tool_calls(text: str) -> list[str]:
    # Prefer closed tags; fall back to unclosed
    results = _parse_all_tags(text, "tool_call")
    if not results:
        results = re.findall(r"<tool_call>([^<\n]{3,80})", text, re.IGNORECASE)
    return [q.strip() for q in results if q.strip()]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _llm_call(llm: VLLMServerModel, prompt: str, max_tokens: int) -> tuple[str, str]:
    """Single-turn call (no conversation history). Returns (raw, clean)."""
    resp = llm._client.chat.completions.create(
        model=llm.model_path,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm._temperature,
        top_p=llm._top_p,
        max_tokens=max_tokens,
    )
    raw   = resp.choices[0].message.content or ""
    clean = _strip_think(raw)
    return raw, clean


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------

def _run_init(
    llm: VLLMServerModel,
    question_text: str,
    max_new_tokens: int,
    log,
) -> dict:
    """Phase 1: decompose question into goal + plan + initial tool_calls."""
    prompt    = INIT_PROMPT.replace("{question}", question_text)
    raw, clean = _llm_call(llm, prompt, max_new_tokens)
    log(f"[Init] Model output:\n{raw}")

    goal       = _parse_tag(clean, "goal") or question_text
    plan       = _parse_tag(clean, "plan") or "Search and synthesize."
    finish     = _parse_tag(clean, "finish")
    tool_calls = _parse_tool_calls(clean)

    log(f"[Init] Goal       : {goal}")
    log(f"[Init] Plan       : {plan}")
    log(f"[Init] Tool calls : {tool_calls}")
    if finish:
        log(f"[Init] Direct finish detected")

    return {"goal": goal, "plan": plan, "tool_calls": tool_calls, "finish": finish}


def _run_analysis(
    llm: VLLMServerModel,
    memory: dict,
    current_query: str,
    observation: str,
    max_new_tokens: int,
    log,
) -> dict:
    """Request 2.1: analyze observation, compress into history, extract new queries."""
    prompt = (
        ANALYSIS_PROMPT
        .replace("{memory}", _format_memory(memory))
        .replace("{current_query}", current_query)
        .replace("{observation}", observation)
    )
    raw, clean = _llm_call(llm, prompt, max_new_tokens)
    log(f"[Analysis 2.1] Model output:\n{raw}")

    new_history = _parse_tag(clean, "new_compressed_history")
    finish      = _parse_tag(clean, "finish")
    tool_calls  = _parse_tool_calls(clean)

    # Retry once if mandatory tag is missing
    if new_history is None:
        log("[Analysis 2.1] <new_compressed_history> missing — retrying")
        retry_prompt = (
            prompt
            + "\n\nYour response must include <new_compressed_history>. "
              "Output it now."
        )
        raw2, clean2 = _llm_call(llm, retry_prompt, max_new_tokens // 2)
        log(f"[Analysis 2.1 retry] Model output:\n{raw2}")
        new_history = _parse_tag(clean2, "new_compressed_history")
        tool_calls += _parse_tool_calls(clean2)
        if not finish:
            finish = _parse_tag(clean2, "finish")

    # Fallback: extract from raw observation if still missing
    if new_history is None:
        new_history = f"Q: {current_query} | A: {observation[:150]}"
        log("[Analysis 2.1] Using fallback compressed history")

    # Dedup tool_calls preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in tool_calls:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return {"new_history": new_history, "tool_calls": deduped, "finish": finish}


def _run_filter(
    llm: VLLMServerModel,
    memory: dict,
    max_filter_tokens: int,
    log,
) -> list[str]:
    """Request 2.2: prune the pending queue.

    Returns the filtered list. Returns [] (empty) when filter signals
    <queue_empty/> or removes all items — caller should proceed to synthesis.
    Safety fallback only activates on parse failure (no tags at all).
    """
    if not memory["pending_queue"]:
        return []

    prompt = FILTER_PROMPT.replace("{memory}", _format_memory(memory))
    _, clean = _llm_call(llm, prompt, max_filter_tokens)
    log(f"[Filter 2.2] Model output:\n{clean}")

    # Model explicitly says queue is empty / done
    if re.search(r"<queue_empty\s*/?>", clean, re.IGNORECASE):
        log("[Filter 2.2] <queue_empty/> — proceeding to synthesis")
        return []

    kept_raw  = _parse_all_tags(clean, "keep")
    original  = list(memory["pending_queue"])

    # No tags at all → parse failure, keep all (don't silently drop queue)
    if not kept_raw and not _parse_all_tags(clean, "remove"):
        log("[Filter 2.2] No tags found — keeping all (parse failure)")
        return original

    # Fuzzy-match kept_raw back to original items
    kept_lower = [k.lower() for k in kept_raw]
    result = [
        q for q in original
        if any(q.lower() in k or k in q.lower() for k in kept_lower)
    ]

    # Model chose to remove all → respect it, proceed to synthesis
    if not result:
        log("[Filter 2.2] All queries removed — proceeding to synthesis")
        return []

    log(f"[Filter 2.2] Kept {len(result)}/{len(original)}: {result}")
    return result


def _run_synthesis(
    llm: VLLMServerModel,
    memory: dict,
    question: str,
    max_final_tokens: int,
    log,
) -> str:
    """Phase 3: generate final answer from full memory."""
    prompt = (
        SYNTHESIS_PROMPT
        .replace("{memory}", _format_memory(memory))
        .replace("{question}", question)
    )
    raw, clean = _llm_call(llm, prompt, max_final_tokens)
    log(f"[Synthesis] Model output:\n{raw}")

    # Strip any residual structural tags
    answer = re.sub(r"<[^>]+>.*?</[^>]+>", "", clean, flags=re.DOTALL).strip()
    answer = re.sub(r"<[^>]+>", "", answer).strip()
    return answer or clean


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_single_rollout(
    llm: VLLMServerModel,
    config: dict,
    record: dict,
    rollout_idx: int,
    trace_path: Path,
    searcher: JinaSearch,
) -> dict:
    """Run one MA-HReAct rollout, write trace, return rollout-level result dict."""
    question    = record["question"]
    gold_answer = record.get("answer", "")
    root_url    = record.get("root_url", "")

    limits           = config.get("limits", {})
    max_new_tokens   = limits.get("max_new_tokens_default", 1536)
    max_final_tokens = limits.get("max_final_tokens", 4096)
    max_filter_tokens = limits.get("max_filter_tokens", 512)
    max_rounds       = limits.get("max_rounds", 15)
    max_sources      = limits.get("max_sources_per_search", 5)

    trace_lines: list[str] = []

    def log(msg: str = "") -> None:
        trace_lines.append(str(msg))

    log("=" * 70)
    log(f"Rollout  : {rollout_idx}")
    log(f"Question : {question}")
    if gold_answer:
        log(f"Answer   : {gold_answer}")
    if root_url:
        log(f"Root URL : {root_url}")
    log("Mode     : MA-HReAct (init + analysis/compress + filter + synthesis)")
    log("=" * 70)
    log()

    used_sources: dict[str, str] = {}
    question_text = _build_query(record)
    memory        = _new_memory()
    answer        = ""
    num_rounds    = 0

    try:
        # ── Phase 1: Decomposition ────────────────────────────────────────────
        log("=== Phase 1: Decomposition ===")
        init = _run_init(llm, question_text, max_new_tokens, log)

        if init["finish"]:
            answer = init["finish"]
            log(f"[Direct answer] {answer}")
        else:
            memory["global_query"] = init["goal"]
            memory["plan"]         = init["plan"]
            for q in init["tool_calls"]:
                if q not in memory["searched_queries"] and q not in memory["pending_queue"]:
                    memory["pending_queue"].append(q)

            log(f"[Queue initialized] {list(memory['pending_queue'])}")

            # ── Phase 2: Execution loop ───────────────────────────────────────
            round_num = 2
            while memory["pending_queue"] and round_num <= max_rounds:
                log(f"\n{'=' * 70}")
                log(f"=== Round {round_num} ===")

                query = memory["pending_queue"].popleft()
                memory["searched_queries"].add(query)
                num_rounds += 1
                log(f"[Search] {query}")

                search_result = searcher.search(query, max_results=max_sources)
                if search_result.get("error"):
                    log(f"[Search error] {search_result['error']}")

                sources  = search_result.get("sources", {})
                obs_text = _format_observation(sources, used_sources)
                log(f"[Sources found] {len(sources)}")
                log(obs_text)

                # ── Request 2.1: Analysis + Compression ──────────────────────
                log(f"\n--- Request {round_num}.1: Analysis ---")
                analysis = _run_analysis(
                    llm, memory, query, obs_text, max_new_tokens, log
                )

                # Commit compressed history entry
                memory["compressed_history"].append({
                    "query":    query,
                    "key_fact": analysis["new_history"],
                })

                if analysis["finish"]:
                    answer = analysis["finish"]
                    log(f"[Finish] {answer}")
                    break

                # Enqueue new tool_calls (with exact dedup)
                for q in analysis["tool_calls"]:
                    if q not in memory["searched_queries"] and q not in memory["pending_queue"]:
                        memory["pending_queue"].append(q)

                # ── Request 2.2: Queue Filter ─────────────────────────────────
                if memory["pending_queue"]:
                    log(f"\n--- Request {round_num}.2: Filter ---")
                    log(f"[Pre-filter]  {list(memory['pending_queue'])}")
                    filtered = _run_filter(llm, memory, max_filter_tokens, log)
                    memory["pending_queue"] = deque(filtered)
                    log(f"[Post-filter] {list(memory['pending_queue'])}")

                round_num += 1

            # ── Phase 3: Synthesis ────────────────────────────────────────────
            if not answer:
                if round_num > max_rounds:
                    log(f"\n[Max rounds ({max_rounds}) reached]")
                else:
                    log("\n[Queue empty]")
                log(f"\n{'=' * 70}")
                log("=== Phase 3: Synthesis ===")
                answer = _run_synthesis(
                    llm, memory, question, max_final_tokens, log
                )

        log()
        log("=" * 70)
        log(f"[Final answer]  {answer}")
        log(f"[Search rounds] {num_rounds}")
        log(f"[Sources used]  {len(used_sources)}")
        for url, title in used_sources.items():
            log(f"  {title}: {url}")
        log(f"[History entries] {len(memory['compressed_history'])}")

        rollout = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": answer,
            "used_sources":     used_sources,
            "final_memory":     _format_memory(memory),
            "num_rounds":       num_rounds,
            "error":            None,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log()
        log(f"[ERROR] {exc}")
        log(tb)
        rollout = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": "",
            "used_sources":     {},
            "final_memory":     "",
            "num_rounds":       0,
            "error":            str(exc),
        }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines))

    return rollout


# ---------------------------------------------------------------------------
# Per-record entry (N rollouts)
# ---------------------------------------------------------------------------

def run_record(
    llm: VLLMServerModel,
    config: dict,
    record: dict,
    traces_dir: Path,
    record_idx: int,
    rollout_count: int,
    searcher: JinaSearch,
) -> dict:
    slug     = _make_slug(record["question"])
    rollouts = []
    for r in range(1, rollout_count + 1):
        trace_path = traces_dir / f"sample_{record_idx:04d}_{slug}_r{r}.txt"
        rollout    = run_single_rollout(llm, config, record, r, trace_path, searcher)
        rollouts.append(rollout)
    return {
        "question":    record["question"],
        "gold_answer": record.get("answer", ""),
        "root_url":    record.get("root_url", ""),
        "info":        record.get("info", {}),
        "rollouts":    rollouts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MA-HReAct inference (init + analysis + filter + synthesis) on a JSONL benchmark."
    )
    parser.add_argument("--config",     default=None, help="Path to config.yaml")
    parser.add_argument("--benchmark",  default=None,
                        help="'hotpot', 'webwalker', or a file path (default: webwalker from config)")
    parser.add_argument("--port",       type=int, default=None,
                        help="vLLM server port (overrides config)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel inference threads")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max number of questions")
    parser.add_argument("--offset",     type=int, default=0,
                        help="Skip first N questions")
    parser.add_argument("--output",     default=None,
                        help="Output JSONL path (default: tests/run_mahreact_<ts>.jsonl)")
    parser.add_argument("--rollouts",   type=int, default=3,
                        help="Rollouts per question for Pass@N evaluation (default: 3)")
    parser.add_argument("--onetime",    action="store_true",
                        help="Single rollout mode -- equivalent to --rollouts 1")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Override max search rounds per rollout (default: from config)")
    args = parser.parse_args()

    rollout_count = 1 if args.onetime else args.rollouts

    config = load_config(args.config)
    if args.max_rounds is not None:
        config.setdefault("limits", {})["max_rounds"] = args.max_rounds

    project_root = Path(__file__).parent
    if args.benchmark == "hotpot":
        benchmark_path = Path(config["eval"]["hotpot_benchmark_path"])
    elif args.benchmark and args.benchmark != "webwalker":
        benchmark_path = Path(args.benchmark)
    else:
        benchmark_path = project_root / config["eval"]["benchmark_path"]
    output_dir = project_root / config["eval"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"run_mahreact_{ts}.jsonl"

    traces_dir = output_path.parent / (output_path.stem + "_traces")

    tqdm.write(f"Loading benchmark : {benchmark_path}")
    records = load_benchmark(benchmark_path)
    if args.benchmark == "hotpot":
        _drop = {"supporting_facts", "context"}
        records = [{k: v for k, v in r.items() if k not in _drop} for r in records]
    records = records[args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    tqdm.write(f"Questions  : {len(records)}")
    tqdm.write(f"Rollouts   : {rollout_count} per question"
               + (" (onetime mode)" if args.onetime else ""))

    port = args.port
    tqdm.write(f"Connecting to vLLM on port "
               f"{port or config.get('vllm_server', {}).get('port', 6001)} ...")
    llm      = VLLMServerModel(config=config, port=port)
    searcher = JinaSearch(config=config)

    max_rounds = config.get("limits", {}).get("max_rounds", 15)
    tqdm.write(f"Workers    : {args.workers}")
    tqdm.write(f"Max rounds : {max_rounds} per rollout")
    tqdm.write(f"Output     : {output_path}")
    tqdm.write(f"Traces     : {traces_dir}/")
    tqdm.write("Mode       : MA-HReAct (init + analysis + filter + synthesis)\n")

    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                run_record, llm, config, record, traces_dir, idx, rollout_count, searcher
            ): idx
            for idx, record in enumerate(records)
        }

        with tqdm(total=len(records), desc="Inference", unit="q",
                  dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    record = records[idx]
                    tqdm.write(f"[FATAL] idx={idx} q={record['question'][:60]!r}: {exc}")
                    result = {
                        "question":    record["question"],
                        "gold_answer": record.get("answer", ""),
                        "root_url":    record.get("root_url", ""),
                        "info":        record.get("info", {}),
                        "rollouts": [{
                            "rollout_idx":      r,
                            "predicted_answer": "",
                            "used_sources":     {},
                            "final_memory":     "",
                            "num_rounds":       0,
                            "error":            str(exc),
                        } for r in range(1, rollout_count + 1)],
                    }

                results_by_idx[idx] = result
                n_rollouts = len(result.get("rollouts", []))
                errors     = sum(1 for ro in result.get("rollouts", []) if ro.get("error"))
                tqdm.write(f"[idx={idx}] {n_rollouts} rollouts"
                           + (f", {errors} err" if errors else "")
                           + f"  q={result['question'][:55]!r}")
                pbar.update(1)

    tqdm.write(f"\nWriting {len(results_by_idx)} records ...")
    total_errors = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx in range(len(records)):
            entry = results_by_idx[idx]
            total_errors += sum(1 for ro in entry.get("rollouts", []) if ro.get("error"))
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    tqdm.write(f"\nDone. {len(records)} questions x {rollout_count} rollouts"
               f", {total_errors} rollout errors.")
    tqdm.write(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
