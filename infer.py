"""
Inference script — runs the search agent over a JSONL benchmark and saves raw predictions.

Does NOT compute metrics. Pass the output to eval.py for scoring.

Requires a running vLLM server (start with: bash start_vllm.sh --port PORT)

Usage
-----
    python infer.py [--config config.yaml] [--port 6001] [--workers 8] \\
                    [--benchmark data.jsonl] [--output results.jsonl]
                    [--rollouts 3]   # default: 3 rollouts per question
                    [--onetime]      # shorthand for --rollouts 1
"""

import argparse
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from models.vllm_server_model import VLLMServerModel
from search.jina_search import JinaSearch
from search_workflow import SearchWorkflow
from utils import load_config


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------

def load_benchmark(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_single_rollout(workflow: SearchWorkflow, query: str, record: dict,
                       rollout_idx: int, trace_path: Path) -> dict:
    """Run ONE rollout, write trace, return rollout-level result dict."""
    question    = record["question"]
    gold_answer = record.get("answer", "")
    root_url    = record.get("root_url", "")

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
    log("=" * 70)
    log()

    try:
        result = workflow.run(query, log=log)

        log()
        log("=" * 70)
        log(f"[Final answer] {result['answer']}")
        log(f"[Rounds] {len(result['rounds'])}")
        log(f"[Sources used] {len(result['used_sources'])}")
        for url, title in result["used_sources"].items():
            log(f"  {title}: {url}")

        rollout = {
            "rollout_idx":      rollout_idx,
            "predicted_answer": result["answer"],
            "used_sources":     result["used_sources"],
            "final_memory":     result["memory"],
            "num_rounds":       len(result["rounds"]),
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

def run_record(workflow: SearchWorkflow, record: dict,
               traces_dir: Path, record_idx: int, rollout_count: int) -> dict:
    """Run N rollouts for one record, return combined entry."""
    query    = _build_query(record)
    slug     = _make_slug(record["question"])
    rollouts = []

    for r in range(1, rollout_count + 1):
        trace_path = traces_dir / f"sample_{record_idx:04d}_{slug}_r{r}.txt"
        rollout    = run_single_rollout(workflow, query, record, r, trace_path)
        rollouts.append(rollout)

    return {
        "question":   record["question"],
        "gold_answer": record.get("answer", ""),
        "root_url":   record.get("root_url", ""),
        "info":       record.get("info", {}),
        "rollouts":   rollouts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run search agent on a JSONL benchmark.")
    parser.add_argument("--config",    default=None,  help="Path to config.yaml")
    parser.add_argument("--benchmark", default=None,  help="Override benchmark JSONL path")
    parser.add_argument("--port",      type=int, default=None, help="vLLM server port (overrides config)")
    parser.add_argument("--workers",   type=int, default=4,    help="Parallel inference threads")
    parser.add_argument("--limit",     type=int, default=None, help="Max number of questions")
    parser.add_argument("--offset",    type=int, default=0,    help="Skip first N questions")
    parser.add_argument("--output",    default=None,  help="Output JSONL path (default: tests/run_<ts>.jsonl)")
    parser.add_argument("--rollouts",  type=int, default=3,
                        help="Rollouts per question for Pass@N evaluation (default: 3)")
    parser.add_argument("--onetime",   action="store_true",
                        help="Single rollout mode — equivalent to --rollouts 1")
    args = parser.parse_args()

    rollout_count = 1 if args.onetime else args.rollouts

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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"run_{ts}.jsonl"

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

    # ── Build shared workflow ─────────────────────────────────────────────────
    port = args.port
    tqdm.write(f"Connecting to vLLM on port "
               f"{port or config.get('vllm_server', {}).get('port', 6001)} ...")
    llm      = VLLMServerModel(config=config, port=port)
    searcher = JinaSearch(config=config)
    workflow = SearchWorkflow(llm=llm, searcher=searcher, config=config)

    tqdm.write(f"Workers   : {args.workers}")
    tqdm.write(f"Output    : {output_path}")
    tqdm.write(f"Traces    : {traces_dir}/\n")

    # ── Parallel inference ────────────────────────────────────────────────────
    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                run_record,
                workflow,
                record,
                traces_dir,
                idx,
                rollout_count,
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
                            "rollout_idx": r,
                            "predicted_answer": "",
                            "used_sources": {},
                            "final_memory": "",
                            "num_rounds": 0,
                            "error": str(exc),
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
