"""Single-pass baseline inference for comparison with infer.py.

Two modes (controlled by --jina flag):
  default  — pure LLM direct answer, no search
  --jina   — single Jina search call → LLM synthesises final answer

Does NOT compute metrics. Pass the output to eval.py for scoring.

Usage
-----
    python infer_base.py [--config config.yaml] [--port 6001] [--workers 4] \\
                         [--benchmark data.jsonl] [--output results.jsonl]
                         [--rollouts 3]   # default: 3 rollouts per question
                         [--onetime]      # shorthand for --rollouts 1
                         [--jina]         # enable single-hop web search
"""

import argparse
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from agent.prompts import BASE_PROMPT
from models.vllm_server_model import VLLMServerModel
from search.jina_search import JinaSearch
from utils import load_config


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DIRECT_PROMPT = """You are a question answering assistant.

Answer the user's question directly using your own internal knowledge.
If you are unsure, answer briefly and avoid making up extra details.
Return the shortest complete answer.

Question:
{question}

Answer:"""

# Mirrors ANALYSIS_PROMPT from agent/prompts.py: same BASE_PROMPT header,
# same formatted_sources format, same decision line — but without memory/queue
# since this is a single-hop baseline (no multi-round context).
JINA_PROMPT = (
    "{base_prompt}\n\n"
    "Search results ({num_sources} sources):\n{formatted_sources}\n\n"
    "User question: {question}\n\n"
    "This is your only search. Give the complete and accurate final answer now.\n"
    "Do NOT output <search> tags — answer directly.\n\n"
    "Begin:"
)


# ---------------------------------------------------------------------------
# Helpers
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


def _format_sources(sources: dict, used_sources: dict) -> str:
    """Format search result sources for injection into a prompt."""
    if not sources:
        return "[No results available]"
    lines = ["**Search results:**"]
    for key, val in sources.items():
        desc  = val.get("description", "")[:300]
        url   = val.get("url", "")
        title = val.get("title", "untitled")
        lines.append(f"{key} ({title}): {desc}")
        lines.append(f"  URL: {url}")
        if url and url not in used_sources:
            used_sources[url] = title
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_single_rollout(
    llm: VLLMServerModel,
    config: dict,
    record: dict,
    rollout_idx: int,
    trace_path: Path,
    searcher: JinaSearch | None = None,
) -> dict:
    """Run ONE rollout, write trace, return rollout-level result dict."""
    question    = record["question"]
    gold_answer = record.get("answer", "")
    root_url    = record.get("root_url", "")
    max_new_tokens = config.get("limits", {}).get("max_new_tokens_default", 1536)
    mode = "single-hop jina search" if searcher is not None else "direct answer"

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
    log(f"Mode     : {mode}")
    log("=" * 70)
    log()

    used_sources: dict[str, str] = {}

    try:
        query = _build_query(record)

        if searcher is not None:
            # ── Single Jina search ────────────────────────────────────────
            log(f"[Searching] {question}")
            result = searcher.search(question)

            if result.get("error"):
                log(f"[Search error] {result['error']}")

            sources   = result.get("sources", {})
            formatted = _format_sources(sources, used_sources)
            log(f"[Sources found] {len(sources)}")
            log(formatted)
            log()

            prompt = JINA_PROMPT.format(
                base_prompt=BASE_PROMPT,
                num_sources=len(sources),
                formatted_sources=formatted,
                question=query,
            )
        else:
            prompt = DIRECT_PROMPT.format(question=query)

        _, _, clean_text = llm.generate(prompt, max_new_tokens=max_new_tokens)

        log()
        log("=" * 70)
        log(f"[Final answer] {clean_text}")
        if used_sources:
            log(f"[Sources used] {len(used_sources)}")
            for url, title in used_sources.items():
                log(f"  {title}: {url}")

        rollout = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": clean_text,
            "used_sources":     used_sources,
            "final_memory":     "",
            "num_rounds":       1,
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
    searcher: JinaSearch | None = None,
) -> dict:
    """Run N rollouts for one record, return combined entry."""
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
        description="Run single-pass baseline inference on a JSONL benchmark."
    )
    parser.add_argument("--config",    default=None,  help="Path to config.yaml")
    parser.add_argument("--benchmark", default=None,  help="Override benchmark JSONL path")
    parser.add_argument("--port",      type=int, default=None, help="vLLM server port (overrides config)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel inference threads")
    parser.add_argument("--limit",     type=int, default=None, help="Max number of questions")
    parser.add_argument("--offset",    type=int, default=0,    help="Skip first N questions")
    parser.add_argument("--output",    default=None,
                        help="Output JSONL path (default: tests/run_base[_jina]_<ts>.jsonl)")
    parser.add_argument("--rollouts",  type=int, default=3,
                        help="Rollouts per question for Pass@N evaluation (default: 3)")
    parser.add_argument("--onetime",   action="store_true",
                        help="Single rollout mode — equivalent to --rollouts 1")
    parser.add_argument("--jina",      action="store_true",
                        help="Enable single-hop Jina web search before answering")
    args = parser.parse_args()

    rollout_count = 1 if args.onetime else args.rollouts
    mode_label = "single-hop jina search" if args.jina else "direct answer"

    # ── Config & paths ────────────────────────────────────────────────────────
    config = load_config(args.config)

    project_root   = Path(__file__).parent
    benchmark_path = Path(args.benchmark) if args.benchmark else (
        project_root / config["eval"]["benchmark_path"]
    )
    output_dir = project_root / config["eval"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem     = "run_base_jina" if args.jina else "run_base"
        output_path = output_dir / f"{stem}_{ts}.jsonl"

    traces_dir = output_path.parent / (output_path.stem + "_traces")

    # ── Load benchmark ────────────────────────────────────────────────────────
    tqdm.write(f"Loading benchmark: {benchmark_path}")
    records = load_benchmark(benchmark_path)
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
    searcher = JinaSearch(config=config) if args.jina else None

    tqdm.write(f"Workers   : {args.workers}")
    tqdm.write(f"Output    : {output_path}")
    tqdm.write(f"Traces    : {traces_dir}/")
    tqdm.write(f"Mode      : {mode_label}\n")

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
