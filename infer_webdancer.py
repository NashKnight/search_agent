"""
infer_webdancer.py — ReAct-style inference for WebDancer (or any instruction-tuned model)
                     with output format compatible with eval.py.

Uses the same two-tool setup as DeepResearch (search via Serper, visit via Jina Reader)
and extracts used_sources from the conversation history so eval.py can include source
URLs in the judge prompt — the same way it evaluates search_agent outputs.

Requires:
  - vLLM server running with WebDancer (bash start_vllm.sh -d)
  - config.yaml: webdancer.serper_api_key and search.jina_api_key

Usage
-----
    python infer_webdancer.py [--config config.yaml] [--port 6001] [--workers 4] \\
                              [--benchmark data.jsonl] [--output tests/run_wd.jsonl]
    python infer_webdancer.py --limit 20 --workers 8
"""

import argparse
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from openai import OpenAI
from tqdm import tqdm

from utils import load_config


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = (
    "You are a deep research assistant. Your core function is to conduct thorough, "
    "multi-source investigations into any topic. You must handle both broad, open-domain "
    "inquiries and queries within specialized academic fields. For every request, synthesize "
    "information from credible, diverse sources to deliver a comprehensive, accurate, and "
    "objective response. When you have gathered sufficient information and are ready to "
    "provide the definitive response, you must enclose the entire final answer within "
    "<answer></answer> tags.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "search", "description": '
    '"Perform web searches and return top results. Accepts multiple queries.", '
    '"parameters": {"type": "object", "properties": {"query": {"type": "array", '
    '"items": {"type": "string"}, "description": "List of search queries."}}, '
    '"required": ["query"]}}}\n'
    '{"type": "function", "function": {"name": "visit", "description": '
    '"Visit webpage(s) and return their content.", '
    '"parameters": {"type": "object", "properties": {'
    '"url": {"type": "array", "items": {"type": "string"}, "description": "URL(s) to visit."}, '
    '"goal": {"type": "string", "description": "What information to look for on the page(s)."}}, '
    '"required": ["url", "goal"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Current date: {date}"
)

_TOKEN_LIMIT_PROMPT = (
    "You have now reached the maximum context length you can handle. "
    "Stop making tool calls and, based on all the information above, provide "
    "your best answer in the following format:\n"
    "<think>your final reasoning</think>\n"
    "<answer>your answer</answer>"
)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _search_jina(queries: list[str], api_key: str, proxies: dict | None) -> str:
    """Search via Jina Search API (s.jina.ai), return combined markdown."""
    if not api_key:
        return "[search] No jina_api_key configured. Set search.jina_api_key in config.yaml."

    parts = []
    for query in queries:
        try:
            url = f"https://s.jina.ai/?q={requests.utils.quote(query)}"
            resp = requests.get(
                url,
                headers={"Accept": "application/json",
                         "Authorization": f"Bearer {api_key}",
                         "X-Respond-With": "no-content"},
                proxies=proxies,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                parts.append(f"No results for '{query}'.")
                continue

            snippets = []
            for i, item in enumerate(data[:10], 1):
                title = (item.get("title") or "").strip()
                link  = (item.get("url") or "").strip()
                snip  = (item.get("description") or "").strip()
                snippets.append(f"{i}. [{title}]({link})\n{snip}")

            parts.append(
                f"Search results for '{query}' ({len(snippets)} results):\n\n"
                + "\n\n".join(snippets)
            )
        except Exception as exc:
            parts.append(f"[search] Error for '{query}': {exc}")

    return "\n\n=======\n\n".join(parts)


def _visit_jina(urls: list[str], goal: str, api_key: str, max_chars: int,
                proxies: dict | None = None) -> str:
    """Visit page(s) by searching the URL via Jina Search (s.jina.ai).

    Uses the search API instead of the reader API to avoid large token consumption —
    consistent with how search_workflow.py handles URL-based queries.
    """
    if not api_key:
        return "[visit] No JINA_API_KEY configured. Set search.jina_api_key in config.yaml."

    parts = []
    for url in urls:
        # Build a targeted query: the URL itself, optionally narrowed by the goal
        query = url if not goal else f"{url} {goal}"
        try:
            resp = requests.get(
                f"https://s.jina.ai/?q={requests.utils.quote(query)}",
                headers={"Accept": "application/json",
                         "Authorization": f"Bearer {api_key}",
                         "X-Respond-With": "no-content"},
                proxies=proxies,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                parts.append(f"[visit] No results found for {url}")
                continue

            snippets = []
            for i, item in enumerate(data[:5], 1):
                title = (item.get("title") or "").strip()
                link  = (item.get("url") or "").strip()
                snip  = (item.get("description") or "").strip()
                snippets.append(f"{i}. [{title}]({link})\n{snip}")

            parts.append(
                f"## Search results for {url}\n\n" + "\n\n".join(snippets)
            )
        except Exception as exc:
            parts.append(f"[visit] Error for {url}: {exc}")

    return "\n\n=======\n\n".join(parts)


# ---------------------------------------------------------------------------
# used_sources extraction
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')


def _extract_sources(messages: list[dict]) -> dict[str, str]:
    """
    Parse conversation history and build {url: title} from:
      - visit tool_call arguments (url list)
      - search tool_response markdown links  [Title](URL)
    """
    used: dict[str, str] = {}

    for i, msg in enumerate(messages):
        content = msg.get("content", "") or ""
        role    = msg.get("role", "")

        if role == "assistant" and "<tool_call>" in content:
            # extract all tool_call blocks
            for block in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
                try:
                    call = json.loads(block.strip())
                except Exception:
                    continue
                name = call.get("name", "")
                args = call.get("arguments", {})

                if name == "visit":
                    raw_urls = args.get("url", [])
                    if isinstance(raw_urls, str):
                        raw_urls = [raw_urls]
                    for url in raw_urls:
                        if url not in used:
                            # use domain as fallback title
                            try:
                                domain = urlparse(url).netloc
                            except Exception:
                                domain = url
                            used[url] = domain

        elif role == "user" and "<tool_response>" in content:
            # search responses contain markdown links
            for title, url in _MD_LINK_RE.findall(content):
                if url not in used:
                    used[url] = title

    return used


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def _build_query(record: dict) -> str:
    """Build user message with optional root_url and lang hints (mirrors infer.py logic)."""
    question = record["question"]
    root_url = record.get("root_url", "")
    lang     = record.get("info", {}).get("lang", "")

    query = question
    if root_url:
        query += (
            f"\n[Search instruction: Start your search from {root_url} as the first entry "
            "point. Visit or search that URL first, then expand to other pages as needed.]"
        )
    if lang == "en":
        query += "\n[Language requirement: Your final answer MUST be written in English.]"
    elif lang == "zh":
        query += "\n[语言要求：最终回答必须使用中文。]"

    return query


def _last_think(messages: list[dict]) -> str:
    """Return the content of the last <think>…</think> block across all assistant turns."""
    last = ""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in re.findall(r"<think>(.*?)</think>", msg.get("content", ""), re.DOTALL):
            last = block.strip()
    return last


def run_single_rollout(record: dict, client: OpenAI, model: str, cfg: dict,
                       rollout_idx: int, trace_path: Path) -> dict:
    """Run ONE rollout for a benchmark record. Returns a rollout-level dict."""
    question  = record["question"]
    gold      = record.get("answer", "")
    root_url  = record.get("root_url", "")

    jina_key   = cfg.get("search", {}).get("jina_api_key", "")
    search_cfg = cfg.get("search", {})
    proxies    = search_cfg.get("proxies") if search_cfg.get("use_proxy") else None
    max_rounds = int(cfg.get("webdancer", {}).get("max_rounds", 50))
    max_tokens = int(cfg.get("webdancer", {}).get("max_new_tokens", 10000))
    max_ctx    = int(cfg.get("webdancer", {}).get("max_context_chars", 400000))
    visit_max  = int(cfg.get("webdancer", {}).get("visit_max_chars", 15000))
    temperature       = float(cfg.get("webdancer", {}).get("temperature", 0.6))
    presence_penalty  = float(cfg.get("webdancer", {}).get("presence_penalty", 1.1))

    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = _SYSTEM_PROMPT_BASE.replace("{date}", today)
    user_query    = _build_query(record)

    messages: list[dict] = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_query},
    ]

    trace_lines: list[str] = []

    def log(msg: str = "") -> None:
        trace_lines.append(str(msg))

    log("=" * 70)
    log(f"Rollout  : {rollout_idx}")
    log(f"Question : {question}")
    if gold:
        log(f"Answer   : {gold}")
    if root_url:
        log(f"Root URL : {root_url}")
    log("=" * 70)

    prediction  = ""
    termination = "max_rounds_reached"
    num_rounds  = 0

    try:
        for _ in range(max_rounds):
            num_rounds += 1

            # ── Call vLLM ──────────────────────────────────────────────────
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=presence_penalty,
                stop=["<tool_response>", "\n<tool_response>"],
            )
            content = resp.choices[0].message.content or ""

            # strip dangling tool_response open tag if model included it
            if "<tool_response>" in content:
                content = content[:content.find("<tool_response>")]

            content = content.strip()
            log(f"\n--- Round {num_rounds} ---")
            log(content)

            messages.append({"role": "assistant", "content": content})

            # ── Final answer ───────────────────────────────────────────────
            if "<answer>" in content and "</answer>" in content:
                prediction  = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                break

            # ── Context limit ──────────────────────────────────────────────
            total_chars = sum(len(m.get("content", "") or "") for m in messages)
            if total_chars > max_ctx:
                log("\n[Context limit reached — requesting forced answer]")
                messages.append({"role": "user", "content": _TOKEN_LIMIT_PROMPT})
                forced = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                forced_content = (forced.choices[0].message.content or "").strip()
                messages.append({"role": "assistant", "content": forced_content})
                log(forced_content)
                if "<answer>" in forced_content and "</answer>" in forced_content:
                    prediction  = forced_content.split("<answer>")[1].split("</answer>")[0].strip()
                    termination = "context_limit"
                else:
                    prediction  = forced_content
                    termination = "context_limit_no_tag"
                break

            # ── Tool call ──────────────────────────────────────────────────
            if "<tool_call>" not in content or "</tool_call>" not in content:
                # model stopped without a tool call or answer — treat as final
                prediction  = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                termination = "no_tool_call"
                break

            tool_block = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            try:
                call      = json.loads(tool_block)
                tool_name = call.get("name", "")
                tool_args = call.get("arguments", {})
            except Exception:
                tool_result = "[tool] Could not parse tool_call JSON."
                messages.append({"role": "user",
                                  "content": f"<tool_response>\n{tool_result}\n</tool_response>"})
                continue

            if tool_name == "search":
                queries     = tool_args.get("query", [])
                if isinstance(queries, str):
                    queries = [queries]
                log(f"[search] {queries}")
                tool_result = _search_jina(queries, jina_key, proxies)

            elif tool_name == "visit":
                raw_urls = tool_args.get("url", [])
                if isinstance(raw_urls, str):
                    raw_urls = [raw_urls]
                goal        = tool_args.get("goal", "")
                log(f"[visit] {raw_urls}")
                tool_result = _visit_jina(raw_urls, goal, jina_key, visit_max, proxies)

            else:
                tool_result = f"[tool] Unknown tool: {tool_name}"

            messages.append({
                "role":    "user",
                "content": f"<tool_response>\n{tool_result}\n</tool_response>",
            })

        # ── Build result ───────────────────────────────────────────────────
        used_sources = _extract_sources(messages)

        if not prediction:
            # last assistant message might contain the answer without tags
            last_assistant = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
            )
            prediction = re.sub(r"<think>.*?</think>", "", last_assistant or "",
                                 flags=re.DOTALL).strip()
            termination = "no_answer_tag"

        log("\n" + "=" * 70)
        log(f"[Prediction]  {prediction[:200]}")
        log(f"[Rounds]      {num_rounds}")
        log(f"[Termination] {termination}")
        log(f"[Sources]     {len(used_sources)}")

        entry = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": prediction,
            "used_sources":     used_sources,
            "final_memory":     _last_think(messages),
            "num_rounds":       num_rounds,
            "error":            None,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log(f"\n[ERROR] {exc}\n{tb}")
        entry = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": "",
            "used_sources":     {},
            "final_memory":     "",
            "num_rounds":       num_rounds,
            "error":            str(exc),
        }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("\n".join(trace_lines), encoding="utf-8")

    return entry


def run_record(record: dict, client: OpenAI, model: str, cfg: dict,
               traces_dir: Path, record_idx: int, rollout_count: int) -> dict:
    """Run N rollouts for one record, return combined entry (matches infer.py format)."""
    slug     = _make_slug(record["question"])
    rollouts = []
    for r in range(1, rollout_count + 1):
        trace_path = traces_dir / f"sample_{record_idx:04d}_{slug}_r{r}.txt"
        rollout    = run_single_rollout(record, client, model, cfg, r, trace_path)
        rollouts.append(rollout)
    return {
        "question":   record["question"],
        "gold_answer": record.get("answer", ""),
        "root_url":   record.get("root_url", ""),
        "info":       record.get("info", {}),
        "rollouts":   rollouts,
    }


# ---------------------------------------------------------------------------
# Benchmark loader  (identical to infer.py)
# ---------------------------------------------------------------------------

def load_benchmark(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return slug[:max_len].strip("_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WebDancer-style ReAct inference; outputs infer.py-compatible JSONL."
    )
    parser.add_argument("--config",    default=None, help="Path to config.yaml")
    parser.add_argument("--benchmark", default=None, help="Override benchmark JSONL path")
    parser.add_argument("--port",      type=int, default=None,
                        help="vLLM server port (overrides config webdancer.vllm_port)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel inference threads (default: 4)")
    parser.add_argument("--rollouts",  type=int, default=3,
                        help="Rollouts per question for Pass@N evaluation (default: 3)")
    parser.add_argument("--onetime",   action="store_true",
                        help="Single rollout mode — equivalent to --rollouts 1")
    parser.add_argument("--limit",     type=int, default=None, help="Max questions to run")
    parser.add_argument("--offset",    type=int, default=0,    help="Skip first N questions")
    parser.add_argument("--output",    default=None,
                        help="Output JSONL path (default: tests/run_wd_<ts>.jsonl)")
    args = parser.parse_args()
    if args.onetime:
        args.rollouts = 1

    cfg = load_config(args.config)

    project_root   = Path(__file__).parent
    benchmark_path = Path(args.benchmark) if args.benchmark else (
        project_root / cfg["eval"]["benchmark_path"]
    )
    output_dir = project_root / cfg["eval"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"run_wd_{ts}.jsonl"

    traces_dir = output_path.parent / (output_path.stem + "_traces")

    # ── vLLM client ───────────────────────────────────────────────────────
    port = args.port or int(cfg.get("webdancer", {}).get("vllm_port", 6001))
    api_base = f"http://127.0.0.1:{port}/v1"
    tqdm.write(f"Connecting to vLLM on port {port} ...")

    client = OpenAI(api_key="EMPTY", base_url=api_base, timeout=300.0)
    try:
        model = client.models.list().data[0].id
        tqdm.write(f"Model: {model}")
    except Exception as e:
        tqdm.write(f"Error: cannot reach vLLM at {api_base}: {e}")
        return

    # ── Benchmark ─────────────────────────────────────────────────────────
    tqdm.write(f"Loading benchmark: {benchmark_path}")
    records = load_benchmark(benchmark_path)[args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    tqdm.write(f"Questions to evaluate: {len(records)}")
    tqdm.write(f"Workers:  {args.workers}")
    tqdm.write(f"Rollouts: {args.rollouts} per question")
    tqdm.write(f"Output:   {output_path}")
    tqdm.write(f"Traces:   {traces_dir}/\n")

    # ── Parallel inference ────────────────────────────────────────────────
    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                run_record,
                record,
                client,
                model,
                cfg,
                traces_dir,
                idx,
                args.rollouts,
            ): idx
            for idx, record in enumerate(records)
        }

        with tqdm(total=len(records), desc="Inference", unit="sample",
                  dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    record = records[idx]
                    tqdm.write(f"[FATAL] idx={idx}: {exc}")
                    result = {
                        "question":   record["question"],
                        "gold_answer": record.get("answer", ""),
                        "root_url":   record.get("root_url", ""),
                        "info":       record.get("info", {}),
                        "rollouts":   [{
                            "rollout_idx":      1,
                            "predicted_answer": "",
                            "used_sources":     {},
                            "final_memory":     "",
                            "num_rounds":       0,
                            "error":            str(exc),
                        }],
                    }

                results_by_idx[idx] = result
                rollouts  = result.get("rollouts", [])
                n_errors  = sum(1 for ro in rollouts if ro.get("error"))
                status    = "ERR" if n_errors else "OK"
                avg_rounds = (sum(ro.get("num_rounds", 0) for ro in rollouts)
                              / len(rollouts)) if rollouts else 0
                tqdm.write(
                    f"[{status}] idx={idx} rollouts={len(rollouts)} "
                    f"avg_rounds={avg_rounds:.1f} q={result['question'][:55]!r}"
                )
                pbar.update(1)

    # ── Write results in original order ──────────────────────────────────
    errors = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx in range(len(records)):
            result = results_by_idx[idx]
            if any(ro.get("error") for ro in result.get("rollouts", [])):
                errors += 1
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    tqdm.write(f"\nDone. {len(records)} questions, {errors} with errors.")
    tqdm.write(f"Results: {output_path}")


if __name__ == "__main__":
    main()
