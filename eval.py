"""
Evaluation script — reads infer.py output and scores predictions with a local judge model.

The judge is a vLLM server on port 6002 (start with: bash start_judge_vllm.sh -d).
Uses the JUDGE_PROMPT_GAIA style from DeepResearch: judge sees question, gold answer,
predicted answer, and the source URLs collected during inference as auxiliary context.

Usage
-----
    python eval.py --input tests/run_20240101_120000.jsonl
    python eval.py --input tests/run.jsonl --output tests/eval_result.json
    python eval.py --input tests/run.jsonl --concurrency 8
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from utils import load_config


# ---------------------------------------------------------------------------
# Judge prompt  (JUDGE_PROMPT_GAIA style, extended with source context)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Reference Sources (URLs the model consulted, for context only):
{sources}

Did the model give an answer **equivalent** to the labeled answer?
Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text."""


def _format_sources(used_sources: dict) -> str:
    """Format used_sources dict {url: title} into a readable list."""
    if not used_sources:
        return "(none)"
    lines = []
    for url, title in list(used_sources.items())[:10]:  # cap at 10 to avoid prompt bloat
        lines.append(f"- {title}: {url}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------

class JudgeClient:
    """Calls the local judge vLLM via OpenAI-compatible API."""

    def __init__(self, config: dict):
        cfg = config["judge"]
        api_url: str = cfg["api_url"].rstrip("/")
        api_key: str = cfg.get("api_key", "EMPTY")
        self.model: str = cfg.get("model", "")
        self.temperature: float = float(cfg.get("temperature", 0.0))
        self.timeout: float = float(cfg.get("timeout", 60))
        self.max_retries: int = int(cfg.get("max_retries", 5))

        self._client = OpenAI(
            api_key=api_key,
            base_url=api_url,
            timeout=self.timeout,
        )

        # Auto-detect model name from server if not set in config
        if not self.model:
            try:
                models = self._client.models.list()
                self.model = models.data[0].id
            except Exception:
                raise RuntimeError(
                    "Judge model name not set in config.yaml and could not be auto-detected. "
                    "Is the judge vLLM running on the configured port?"
                )

    def judge(self, question: str, gold_answer: str, predicted_answer: str,
              used_sources: dict) -> str:
        """Return 'correct' or 'incorrect'."""
        if not gold_answer or not predicted_answer:
            return "unknown"

        prompt = JUDGE_PROMPT.format(
            question=question,
            correct_answer=gold_answer,
            response=predicted_answer,
            sources=_format_sources(used_sources),
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=16,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content.strip()
                # Accept "Correct" / "Incorrect" (case-insensitive, first word)
                first_word = raw.lower().split()[0] if raw else ""
                if first_word.startswith("correct"):
                    return "correct"
                if first_word.startswith("incorrect"):
                    return "incorrect"
                # Unexpected output — treat as incorrect
                print(f"  [Judge] unexpected response: {raw!r}")
                return "incorrect"
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"  [Judge error after {attempt} attempts] {exc}")
                    return "unknown"
                time.sleep(min(2 ** attempt, 30))

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
    verdict = judge.judge(
        question=record.get("question", ""),
        gold_answer=record.get("gold_answer", ""),
        predicted_answer=record.get("predicted_answer", ""),
        used_sources=record.get("used_sources", {}),
    )
    return {**record, "verdict": verdict}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score infer.py predictions using the local judge vLLM on port 6002."
    )
    parser.add_argument("--input",       required=True, help="Prediction JSONL file from infer.py")
    parser.add_argument("--config",      default=None,  help="Path to config.yaml")
    parser.add_argument("--output",      default=None,  help="Output JSON file (default: <input>_eval.json)")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel judge calls (default: 8)")
    args = parser.parse_args()

    config = load_config(args.config)
    judge = JudgeClient(config)

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_eval.json"
    )

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Judge  : {judge.model} @ {config['judge']['api_url']}")
    print(f"Workers: {args.concurrency}\n")

    records = load_predictions(input_path)
    total   = len(records)
    print(f"Records to evaluate: {total}\n")

    scored: list[dict] = [None] * total
    correct = incorrect = unknown = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(evaluate_record, judge, rec): idx
            for idx, rec in enumerate(records)
        }
        for future in as_completed(futures):
            idx    = futures[future]
            result = future.result()
            scored[idx] = result
            v = result["verdict"]
            if v == "correct":
                correct   += 1
            elif v == "incorrect":
                incorrect += 1
            else:
                unknown   += 1

            done = correct + incorrect + unknown
            q_short = result["question"][:55]
            print(f"  [{done:>4}/{total}]  {v.upper():<10}  {q_short}")

    evaluated = correct + incorrect
    accuracy  = correct / evaluated if evaluated > 0 else 0.0

    summary = {
        "input_file":    str(input_path),
        "judge_model":   judge.model,
        "judge_api":     config["judge"]["api_url"],
        "total":         total,
        "correct":       correct,
        "incorrect":     incorrect,
        "unknown":       unknown,
        "evaluated":     evaluated,
        "accuracy":      round(accuracy, 4),
        "accuracy_pct":  f"{accuracy * 100:.2f}%",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": scored}, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Total     : {total}")
    print(f"Correct   : {correct}")
    print(f"Incorrect : {incorrect}")
    print(f"Unknown   : {unknown}  (missing gold answer or judge error)")
    print(f"Accuracy  : {accuracy * 100:.2f}%  ({correct}/{evaluated})")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
