"""
Inference script — runs the search agent over a JSONL benchmark and saves raw predictions.

Does NOT compute metrics. Pass the output to eval.py for scoring.

Requires a running vLLM server (start with: bash start_vllm.sh --port PORT)

Usage
-----
    python infer.py [--config config.yaml] [--port 6001] [--workers 8] \\
                    [--benchmark data.jsonl] [--output results.jsonl]
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
# Trace writer helpers
# ---------------------------------------------------------------------------

def _make_slug(text: str, max_len: int = 40) -> str:
    """Turn a question into a safe filename fragment."""
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    return slug[:max_len].strip("_")


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def run_single(workflow: SearchWorkflow, record: dict, trace_path: Path) -> dict:
    """Run one benchmark record, write full trace to trace_path, return result."""
    question = record["question"]
    gold_answer = record.get("answer", "")
    root_url = record.get("root_url", "")

    query = question
    if root_url:
        query = f"{question}\n(Reference site: {root_url})"

    # Per-sample trace buffer — thread-safe because it's a local variable
    trace_lines: list[str] = []

    def log(msg: str = "") -> None:
        trace_lines.append(str(msg))

    # Write header
    log("=" * 70)
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

        entry = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": result["answer"],
            "used_sources": result["used_sources"],
            "final_memory": result["memory"],
            "num_rounds": len(result["rounds"]),
            "root_url": root_url,
            "info": record.get("info", {}),
            "error": None,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        log()
        log(f"[ERROR] {exc}")
        log(tb)
        entry = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": "",
            "used_sources": {},
            "final_memory": "",
            "num_rounds": 0,
            "root_url": root_url,
            "info": record.get("info", {}),
            "error": str(exc),
        }

    # Write trace file
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines))

    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate search agent on a JSONL benchmark.")
    parser.add_argument("--config",    default=None,  help="Path to config.yaml")
    parser.add_argument("--benchmark", default=None,  help="Override benchmark JSONL path")
    parser.add_argument("--port",      type=int, default=None, help="vLLM server port (overrides config)")
    parser.add_argument("--workers",   type=int, default=4,    help="Number of parallel inference threads")
    parser.add_argument("--limit",     type=int, default=None, help="Max number of questions")
    parser.add_argument("--offset",    type=int, default=0,    help="Skip first N questions")
    parser.add_argument("--output",    default=None,  help="Output JSONL path (default: tests/run_<ts>.jsonl)")
    args = parser.parse_args()

    # ── Config & paths ────────────────────────────────────────────────────────
    config = load_config(args.config)

    project_root = Path(__file__).parent
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

    # Trace directory: same stem as output file, with _traces suffix
    traces_dir = output_path.parent / (output_path.stem + "_traces")

    # ── Load benchmark ────────────────────────────────────────────────────────
    tqdm.write(f"Loading benchmark: {benchmark_path}")
    records = load_benchmark(benchmark_path)
    records = records[args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    tqdm.write(f"Questions to evaluate: {len(records)}")

    # ── Build shared workflow (stateless across calls) ────────────────────────
    port = args.port  # None → VLLMServerModel reads from config
    tqdm.write(f"Connecting to vLLM server on port {port or config.get('vllm_server', {}).get('port', 6001)} ...")
    llm = VLLMServerModel(config=config, port=port)
    searcher = JinaSearch(config=config)
    workflow = SearchWorkflow(llm=llm, searcher=searcher, config=config)

    tqdm.write(f"Workers: {args.workers}")
    tqdm.write(f"Output:  {output_path}")
    tqdm.write(f"Traces:  {traces_dir}/\n")

    # ── Parallel inference ────────────────────────────────────────────────────
    # results_by_idx preserves original ordering regardless of completion order
    results_by_idx: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                run_single,
                workflow,
                record,
                traces_dir / f"sample_{idx:04d}_{_make_slug(record['question'])}.txt",
            ): idx
            for idx, record in enumerate(records)
        }

        with tqdm(total=len(records), desc="Inference", unit="sample", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:  # should not happen — run_single catches internally
                    record = records[idx]
                    tqdm.write(f"[FATAL] idx={idx} q={record['question'][:60]!r}: {exc}")
                    result = {
                        "question": record["question"],
                        "gold_answer": record.get("answer", ""),
                        "predicted_answer": "",
                        "used_sources": {},
                        "final_memory": "",
                        "num_rounds": 0,
                        "root_url": record.get("root_url", ""),
                        "info": record.get("info", {}),
                        "error": str(exc),
                    }

                results_by_idx[idx] = result

                status = "ERR" if result.get("error") else "OK"
                tqdm.write(f"[{status}] idx={idx} rounds={result.get('num_rounds', 0)} "
                           f"q={result['question'][:60]!r}")
                pbar.update(1)

    # ── Write results in original input order ─────────────────────────────────
    tqdm.write(f"\nWriting {len(results_by_idx)} results in original order...")
    errors = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx in range(len(records)):
            result = results_by_idx[idx]
            if result.get("error"):
                errors += 1
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    tqdm.write(f"\nDone. {len(records)} questions, {errors} errors.")
    tqdm.write(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
