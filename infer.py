"""
Inference script — runs the search agent over a JSONL benchmark and saves raw predictions.

Does NOT compute metrics. Pass the output to eval.py for scoring.

Usage
-----
    python infer.py [--config config.yaml] [--limit N] [--offset N] [--output tests/run_NAME.jsonl]
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from models.vllm_model import VLLMModel
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
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_single(workflow: SearchWorkflow, record: dict) -> dict:
    """Run one benchmark record through the workflow and return the result entry."""
    question = record["question"]
    gold_answer = record.get("answer", "")
    root_url = record.get("root_url", "")

    # Optionally include root_url as search context
    query = question
    if root_url:
        query = f"{question}\n(Reference site: {root_url})"

    try:
        result = workflow.run(query)
        return {
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
        traceback.print_exc()
        return {
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate search agent on a JSONL benchmark.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--benchmark", default=None, help="Override benchmark JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions")
    parser.add_argument("--output", default=None, help="Output JSONL file path (default: tests/run_<timestamp>.jsonl)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve paths
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

    # Load benchmark
    print(f"Loading benchmark: {benchmark_path}")
    records = load_benchmark(benchmark_path)
    records = records[args.offset:]
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Questions to evaluate: {len(records)}")

    # Build workflow
    print("Loading model...")
    llm = VLLMModel(config=config)
    searcher = JinaSearch(config=config)
    workflow = SearchWorkflow(llm=llm, searcher=searcher, config=config)

    # Run evaluation
    print(f"Output file: {output_path}\n")
    errors = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records, 1):
            print(f"\n{'=' * 70}")
            print(f"[{idx}/{len(records)}] {record['question'][:100]}")
            result = evaluate_single(workflow, record)
            if result["error"]:
                errors += 1
                print(f"[ERROR] {result['error']}")
            else:
                print(f"[Answer] {result['predicted_answer'][:200]}")
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()  # write immediately so partial results are saved

    print(f"\n{'=' * 70}")
    print(f"Done. {len(records)} questions, {errors} errors.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
