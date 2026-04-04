"""
Evaluation script — reads infer.py output and scores predictions with a judge model.

The judge is an OpenAI-compatible chat API (GPT-4o, Qwen, etc.) configured in
config.yaml under the `judge` section. It determines whether each predicted
answer is correct relative to the gold answer, then reports accuracy.

Usage
-----
    python eval.py --input tests/run_20240101_120000.jsonl
    python eval.py --input tests/run_*.jsonl --output tests/eval_result.json
    python eval.py --input tests/run.jsonl --config config.yaml --concurrency 8
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from utils import load_config


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a strict answer evaluator. "
    "Given a question, a reference answer, and a predicted answer, "
    "decide whether the predicted answer is correct.\n\n"
    "Rules:\n"
    "- Minor wording differences are acceptable as long as the core fact is the same.\n"
    "- Partial answers that miss key facts are INCORRECT.\n"
    "- If there is no reference answer, output 'unknown'.\n\n"
    "Reply with exactly one word: 'correct', 'incorrect', or 'unknown'."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Reference answer: {gold_answer}\n\n"
    "Predicted answer: {predicted_answer}\n\n"
    "Is the predicted answer correct?"
)


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------

class JudgeClient:
    """Calls an OpenAI-compatible chat API to judge answer correctness."""

    def __init__(self, config: dict):
        cfg = config["judge"]
        self.api_url: str = cfg["api_url"].rstrip("/")
        self.api_key: str = cfg.get("api_key", "")
        self.model: str = cfg.get("model", "gpt-4o")
        self.temperature: float = cfg.get("temperature", 0.0)
        self.timeout: int = cfg.get("timeout", 30)
        self.max_retries: int = cfg.get("max_retries", 3)

    def judge(self, question: str, gold_answer: str, predicted_answer: str) -> str:
        """Return 'correct', 'incorrect', or 'unknown'."""
        if not gold_answer or not predicted_answer:
            return "unknown"

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=predicted_answer,
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                verdict = (
                    resp.json()["choices"][0]["message"]["content"]
                    .strip()
                    .lower()
                    .split()[0]  # take first word only
                )
                if verdict in ("correct", "incorrect", "unknown"):
                    return verdict
                # unexpected output — treat as incorrect
                return "incorrect"
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"  [Judge error after {attempt} attempts] {exc}")
                    return "unknown"
                time.sleep(2 ** attempt)  # exponential backoff

        return "unknown"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_predictions(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_record(judge: JudgeClient, record: dict) -> dict:
    """Judge a single prediction record and return an enriched result dict."""
    verdict = judge.judge(
        question=record.get("question", ""),
        gold_answer=record.get("gold_answer", ""),
        predicted_answer=record.get("predicted_answer", ""),
    )
    return {**record, "verdict": verdict}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score infer.py predictions using a judge model."
    )
    parser.add_argument("--input", required=True, help="Prediction JSONL file from infer.py")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for scored results (default: same dir as input, suffixed _eval.json)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Number of parallel judge calls (default: 4)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    judge = JudgeClient(config)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_eval.json"
    )

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Judge : {judge.model} @ {judge.api_url}\n")

    records = load_predictions(input_path)
    total = len(records)
    print(f"Records to evaluate: {total}\n")

    scored: list[dict] = [None] * total
    correct = incorrect = unknown = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(evaluate_record, judge, rec): idx
            for idx, rec in enumerate(records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            scored[idx] = result
            v = result["verdict"]
            if v == "correct":
                correct += 1
            elif v == "incorrect":
                incorrect += 1
            else:
                unknown += 1

            done = correct + incorrect + unknown
            print(
                f"  [{done}/{total}] {result['question'][:60]:<60}  → {v}"
            )

    # Summary statistics
    evaluated = correct + incorrect   # excludes unknown
    accuracy = correct / evaluated if evaluated > 0 else 0.0

    summary = {
        "input_file": str(input_path),
        "judge_model": judge.model,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "unknown": unknown,
        "evaluated": evaluated,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy * 100:.2f}%",
    }

    output = {"summary": summary, "results": scored}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Total   : {total}")
    print(f"Correct : {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Unknown : {unknown}  (no gold answer or judge error)")
    print(f"Accuracy: {accuracy * 100:.2f}%  ({correct}/{evaluated})")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
