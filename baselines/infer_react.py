"""Vanilla ReAct baseline — Thought / Action / Observation loop, Jina search only.

Only Search[query] and Finish[answer] actions are available.
No visit/read to keep Jina token usage bounded.
Output format mirrors infer.py for direct comparison.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)

Usage
-----
    python infer_react.py [--config config.yaml] [--port 6001] [--workers 4] \\
                          [--benchmark data.jsonl] [--output results.jsonl]
                          [--rollouts 3]     # default: 3 rollouts per question
                          [--onetime]        # shorthand for --rollouts 1
                          [--max-rounds N]   # override config max_rounds
"""

import argparse
import json
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from models.vllm_server_model import VLLMServerModel
from search.jina_search import JinaSearch
from utils import load_config


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an intelligent research assistant. Answer the user's question step by step.

Available actions:
  Search[query]  — search the web with a concise query (3-10 words)
  Finish[answer] — return the final answer and end the task

Format your response as interleaved Thought / Action steps:

  Thought 1: <reason about what to do next>
  Action 1: Search[your search query]

After each Search the system will provide an Observation. Then continue:

  Thought 2: <reason about the search results>
  Action 2: Finish[your final answer]

Rules:
- Do NOT write Observation lines yourself — the system fills them in.
- Output exactly ONE action per response.
- Use Finish[answer] as soon as you have enough information.
- Keep search queries concise and specific.\
"""

FORCE_FINISH_MSG = (
    "You have used all available search rounds. "
    "Based on what you have gathered, provide your final answer now in the format:\n"
    "Action: Finish[your answer here]"
)


# ---------------------------------------------------------------------------
# Benchmark loader & helpers
# ---------------------------------------------------------------------------

def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_slug(text: str, max_len: int = 40) -> str:
    """Turn a question into a safe filename fragment."""
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return slug[:max_len].strip("_")


def _build_query(record: dict) -> str:
    """Build query string with optional root_url and lang hints."""
    question = record["question"]
    root_url = record.get("root_url", "")
    lang     = record.get("info", {}).get("lang", "")

    query = question
    if root_url:
        query += f"\n<root_url>{root_url}</root_url>"
    if lang == "en":
        query += "\n[Language requirement: Your final answer MUST be written in English.]"
    elif lang == "zh":
        query += "\n[语言要求：最终回答必须使用中文。]"
    return query


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 thinking mode output)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _format_observation(sources: dict, used_sources: dict) -> str:
    """Format Jina search results as a ReAct observation string."""
    if not sources:
        return "[No results found]"
    lines = []
    for val in sources.values():
        desc  = val.get("description", "")
        url   = val.get("url", "")
        title = val.get("title", "untitled")
        lines.append(f"- {title}: {desc}")
        lines.append(f"  URL: {url}")
        if url and url not in used_sources:
            used_sources[url] = title
    return "\n".join(lines)


def _parse_action(text: str) -> tuple[str, str] | None:
    """Extract the first Search or Finish action from model output.

    Returns ('search', query) | ('finish', answer) | None.
    Prefers explicit 'Action N:' lines; falls back to bare tags.
    """
    m = re.search(
        r"(?:^|\n)\s*Action\s*(?:\d+\s*)?:\s*(Search|Finish)\[(.+?)\]",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        m = re.search(r"\b(Search|Finish)\[(.+?)\]", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return None


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
    """Run ONE ReAct rollout, write trace, return rollout-level result dict."""
    question    = record["question"]
    gold_answer = record.get("answer", "")
    root_url    = record.get("root_url", "")

    limits         = config.get("limits", {})
    max_new_tokens = limits.get("max_new_tokens_default", 1536)
    max_rounds     = limits.get("max_rounds", 15)
    max_sources    = limits.get("max_sources_per_search", 5)

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
    log("Mode     : vanilla ReAct (search-only)")
    log("=" * 70)
    log()

    used_sources: dict[str, str] = {}
    question_text = _build_query(record)

    # Multi-turn conversation: [system, user(Q+Thought1:), assistant(T+A), user(Obs+Thought2:), ...]
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Question: {question_text}\nThought 1:"},
    ]

    answer     = ""
    num_rounds = 0  # counts actual Search calls made
    step       = 1
    done       = False

    try:
        while step <= max_rounds:
            log(f"--- Step {step} ---")

            # stop=["\nObservation"] prevents the model from hallucinating search results
            resp = llm._client.chat.completions.create(
                model=llm.model_path,
                messages=messages,
                temperature=llm._temperature,
                top_p=llm._top_p,
                max_tokens=max_new_tokens,
                stop=["\nObservation"],
            )
            raw   = resp.choices[0].message.content or ""
            clean = _strip_think(raw)
            log(f"[Model output]\n{raw}")

            action = _parse_action(clean)

            if action is None:
                log("[No valid action found — treating output as final answer]")
                answer = clean or raw
                messages.append({"role": "assistant", "content": raw})
                done = True
                break

            action_type, action_arg = action

            if action_type == "finish":
                answer = action_arg
                log(f"[Finish] {answer}")
                messages.append({"role": "assistant", "content": raw})
                done = True
                break

            # action_type == "search"
            num_rounds += 1
            log(f"[Search] {action_arg}")

            search_result = searcher.search(action_arg, max_results=max_sources)
            if search_result.get("error"):
                log(f"[Search error] {search_result['error']}")

            sources  = search_result.get("sources", {})
            obs_text = _format_observation(sources, used_sources)
            log(f"[Sources found] {len(sources)}")
            log(obs_text)
            log()

            # Append assistant turn (Thought + Action, cut at stop sequence)
            messages.append({"role": "assistant", "content": raw})
            # Append user turn: Observation + prompt for next Thought
            messages.append({
                "role": "user",
                "content": f"Observation {step}: {obs_text}\nThought {step + 1}:",
            })

            step += 1

        # ── Forced final answer if loop exhausted without Finish ──────────────
        if not done:
            log("[Max rounds reached — forcing final answer]")
            messages.append({"role": "user", "content": FORCE_FINISH_MSG})
            resp = llm._client.chat.completions.create(
                model=llm.model_path,
                messages=messages,
                temperature=llm._temperature,
                top_p=llm._top_p,
                max_tokens=512,
            )
            raw    = resp.choices[0].message.content or ""
            clean  = _strip_think(raw)
            forced = _parse_action(clean)
            answer = forced[1] if (forced and forced[0] == "finish") else clean
            log(f"[Forced answer] {answer}")

        log()
        log("=" * 70)
        log(f"[Final answer] {answer}")
        log(f"[Search rounds] {num_rounds}")
        log(f"[Sources used] {len(used_sources)}")
        for url, title in used_sources.items():
            log(f"  {title}: {url}")

        rollout = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": answer,
            "used_sources":     used_sources,
            "final_memory":     "",
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
    """Run N rollouts for one record, return combined entry."""
    slug     = _make_slug(record["question"])
    rollouts = []

    for r in range(1, rollout_count + 1):
        trace_path = traces_dir / f"sample_{record_idx:04d}_{slug}_r{r}.txt"
        rollout    = run_single_rollout(llm, config, record, r, trace_path, searcher)
        rollouts.append(rollout)

    return {
        "id":          record.get("id", ""),
        "question":    record["question"],
        "gold_answer": record.get("answer", ""),
        "type":        record.get("type", ""),
        "level":       record.get("level", ""),
        "root_url":    record.get("root_url", ""),
        "info":        record.get("info", {}),
        "rollouts":    rollouts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vanilla ReAct inference (search-only) on a JSONL benchmark."
    )
    parser.add_argument("--config",     default=None, help="Path to config.yaml")
    parser.add_argument("--benchmark",  default=None,
                        help="'hotpot', 'webwalker', or a file path (default: webwalker from config)")
    parser.add_argument("--port",       type=int, default=None, help="vLLM server port (overrides config)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel inference threads")
    parser.add_argument("--limit",      type=int, default=None, help="Max number of questions")
    parser.add_argument("--offset",     type=int, default=0,    help="Skip first N questions")
    parser.add_argument("--output",     default=None,
                        help="Output JSONL path (default: tests/run_react_<ts>.jsonl)")
    parser.add_argument("--rollouts",   type=int, default=3,
                        help="Rollouts per question for Pass@N evaluation (default: 3)")
    parser.add_argument("--onetime",    action="store_true",
                        help="Single rollout mode — equivalent to --rollouts 1")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Override max search rounds per rollout (default: from config)")
    args = parser.parse_args()

    rollout_count = 1 if args.onetime else args.rollouts

    # ── Config & paths ────────────────────────────────────────────────────────
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
        output_path = output_dir / f"run_react_{ts}.jsonl"

    traces_dir = output_path.parent / (output_path.stem + "_traces")

    # ── Load benchmark ────────────────────────────────────────────────────────
    tqdm.write(f"Loading benchmark: {benchmark_path}")
    records = load_benchmark(benchmark_path)
    if args.benchmark == "hotpot":
        _drop = {"supporting_facts", "context"}
        records = [{k: v for k, v in r.items() if k not in _drop} for r in records]
    records = records[args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    tqdm.write(f"Questions : {len(records)}")
    tqdm.write(f"Rollouts  : {rollout_count} per question"
               + (" (onetime mode)" if args.onetime else ""))

    # ── Build shared objects ──────────────────────────────────────────────────
    port = args.port
    tqdm.write(f"Connecting to vLLM on port "
               f"{port or config.get('vllm_server', {}).get('port', 6001)} ...")
    llm      = VLLMServerModel(config=config, port=port)
    searcher = JinaSearch(config=config)

    max_rounds = config.get("limits", {}).get("max_rounds", 15)
    tqdm.write(f"Workers   : {args.workers}")
    tqdm.write(f"Max rounds: {max_rounds} per rollout")
    tqdm.write(f"Output    : {output_path}")
    tqdm.write(f"Traces    : {traces_dir}/")
    tqdm.write("Mode      : vanilla ReAct (search-only)\n")

    # ── Parallel inference ────────────────────────────────────────────────────
    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                run_record,
                llm,
                config,
                record,
                traces_dir,
                idx,
                rollout_count,
                searcher,
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
                        "id":          record.get("id", ""),
                        "question":    record["question"],
                        "gold_answer": record.get("answer", ""),
                        "type":        record.get("type", ""),
                        "level":       record.get("level", ""),
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

    # ── Write results in original order ──────────────────────────────────────
    tqdm.write(f"\nWriting {len(results_by_idx)} records ...")
    total_errors = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx in range(len(records)):
            entry = results_by_idx[idx]
            total_errors += sum(1 for ro in entry.get("rollouts", []) if ro.get("error"))
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    tqdm.write(f"\nDone. {len(records)} questions × {rollout_count} rollouts"
               f", {total_errors} rollout errors.")
    tqdm.write(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
