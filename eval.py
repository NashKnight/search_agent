"""
Evaluation script — reads infer.py output and scores predictions with a local judge model.

The judge is a vLLM server on port 6002 (start with: bash commands/start_judge_vllm.sh -d).
Uses JUDGE_PROMPT_GAIA style: judge sees question, gold answer, predicted answer,
and source URLs as auxiliary context.

Supports multi-rollout format (default from infer.py): Pass@N = at least one
rollout correct. Also reports Avg.Pass (average per-rollout accuracy).

Usage
-----
    python eval.py --input tests/run_20240101_120000.jsonl
    python eval.py --input tests/run.jsonl --output tests/eval_result.json
    python eval.py --input tests/run.jsonl --workers 8
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from utils import load_config


# ---------------------------------------------------------------------------
# Judge prompt — JUDGE_PROMPT_GAIA style + source context
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
    if not used_sources:
        return "(none)"
    lines = [f"- {title}: {url}"
             for url, title in list(used_sources.items())[:10]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------

class JudgeClient:
    def __init__(self, config: dict):
        cfg             = config["judge"]
        api_url: str    = cfg["api_url"].rstrip("/")
        api_key: str    = cfg.get("api_key", "EMPTY")
        self.model: str = cfg.get("model", "")
        self.temperature  = float(cfg.get("temperature", 0.0))
        self.timeout      = float(cfg.get("timeout", 60))
        self.max_retries  = int(cfg.get("max_retries", 5))

        self._client = OpenAI(api_key=api_key, base_url=api_url, timeout=self.timeout)

        if not self.model:
            try:
                self.model = self._client.models.list().data[0].id
            except Exception:
                raise RuntimeError(
                    "Judge model not set in config.yaml and server unreachable. "
                    "Start judge vLLM: bash commands/start_judge_vllm.sh -d"
                )

    def judge(self, question: str, gold_answer: str, predicted_answer: str,
              used_sources: dict) -> str:
        """Return 'correct', 'incorrect', or 'unknown'."""
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
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw        = resp.choices[0].message.content.strip()
                first_word = raw.lower().split()[0] if raw else ""
                if first_word.startswith("correct"):
                    return "correct"
                if first_word.startswith("incorrect"):
                    return "incorrect"
                print(f"  [Judge] unexpected: {raw!r}")
                return "incorrect"
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"  [Judge error ×{attempt}] {exc}")
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


def _normalize_rollouts(record: dict) -> list[dict]:
    """Return list of rollout dicts. Handles both multi-rollout and legacy single format."""
    if "rollouts" in record:
        return record["rollouts"]
    # legacy single-rollout format from pre-5.1
    return [{
        "rollout_idx":      1,
        "predicted_answer": record.get("predicted_answer", ""),
        "used_sources":     record.get("used_sources", {}),
        "final_memory":     record.get("final_memory", ""),
        "num_rounds":       record.get("num_rounds", 0),
        "error":            record.get("error"),
    }]


def _normalize_level(record: dict) -> str:
    info = record.get("info") or {}
    level = str(info.get("difficulty_level", "unknown")).strip().lower()
    aliases = {
        "easy": "easy",
        "medium": "medium",
        "midium": "medium",
        "hard": "hard",
    }
    return aliases.get(level, level or "unknown")


def build_level_summary(scored_records: list[dict]) -> dict:
    level_order = ["easy", "medium", "hard"]
    buckets: dict[str, dict[str, int]] = {}

    for record in scored_records:
        level = _normalize_level(record)
        bucket = buckets.setdefault(level, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if record.get("pass_at_n"):
            bucket["correct"] += 1

    summary: dict[str, dict] = {}
    for level in level_order + sorted(k for k in buckets if k not in level_order):
        if level not in buckets:
            continue
        total = buckets[level]["total"]
        correct = buckets[level]["correct"]
        acc = correct / total if total else 0.0
        summary[level] = {
            "total_questions": total,
            "correct_questions": correct,
            "accuracy": round(acc, 4),
            "accuracy_pct": f"{acc * 100:.2f}%",
        }

    return summary


def evaluate_record(judge: JudgeClient, record: dict) -> dict:
    """Judge every rollout, return record enriched with per-rollout verdicts and Pass@N."""
    question    = record.get("question", "")
    gold_answer = record.get("gold_answer", "")
    rollouts    = _normalize_rollouts(record)

    scored_rollouts = []
    for ro in rollouts:
        verdict = judge.judge(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=ro.get("predicted_answer", ""),
            used_sources=ro.get("used_sources", {}),
        )
        scored_rollouts.append({**ro, "verdict": verdict})

    # Pass@N: at least one rollout correct
    pass_at_n = any(ro["verdict"] == "correct" for ro in scored_rollouts)

    return {
        **record,
        "rollouts":   scored_rollouts,
        "pass_at_n":  pass_at_n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score infer.py predictions using the local judge vLLM."
    )
    parser.add_argument("--input",       required=True, help="Prediction JSONL from infer.py")
    parser.add_argument("--config",      default=None)
    parser.add_argument("--output",      default=None,  help="Output JSON (default: <input>_eval.json)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel judge threads")
    args = parser.parse_args()

    config = load_config(args.config)
    judge  = JudgeClient(config)

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_eval.json")

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Judge  : {judge.model} @ {config['judge']['api_url']}")
    print(f"Workers: {args.workers}\n")

    records = load_predictions(input_path)
    total   = len(records)
    # detect rollout count from first record
    n_rollouts = len(_normalize_rollouts(records[0])) if records else 1
    print(f"Records    : {total}")
    print(f"Rollouts   : {n_rollouts} per question  →  judging {total * n_rollouts} predictions\n")

    scored: list[dict] = [None] * total

    # counters
    pass_n = 0          # questions where ≥1 rollout correct  (Pass@N)
    total_correct_rollouts = 0   # for Avg.Pass
    total_judged_rollouts  = 0
    unknown_rollouts       = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_record, judge, rec): idx
                   for idx, rec in enumerate(records)}

        for future in as_completed(futures):
            idx    = futures[future]
            result = future.result()
            scored[idx] = result

            rollouts   = result.get("rollouts", [])
            verdicts   = [ro["verdict"] for ro in rollouts]
            n_correct  = sum(v == "correct"   for v in verdicts)
            n_unknown  = sum(v == "unknown"   for v in verdicts)
            n_judged   = sum(v != "unknown"   for v in verdicts)

            if result["pass_at_n"]:
                pass_n += 1
            total_correct_rollouts += n_correct
            total_judged_rollouts  += n_judged
            unknown_rollouts       += n_unknown

            done      = sum(1 for s in scored if s is not None)
            tag       = "PASS" if result["pass_at_n"] else "FAIL"
            v_summary = "/".join(v[0].upper() for v in verdicts)   # e.g. C/I/C
            print(f"  [{done:>4}/{total}]  {tag}  [{v_summary}]  {result['question'][:50]}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    pass_at_n_pct = pass_n / total if total else 0.0
    avg_pass      = (total_correct_rollouts / total_judged_rollouts
                     if total_judged_rollouts else 0.0)

    summary = {
        "input_file":      str(input_path),
        "judge_model":     judge.model,
        "judge_api":       config["judge"]["api_url"],
        "total_questions": total,
        "rollouts_per_q":  n_rollouts,
        # Pass@N — at least 1 rollout correct
        "pass_at_n":       pass_n,
        "pass_at_n_pct":   f"{pass_at_n_pct * 100:.2f}%",
        # Avg.Pass — average of individual rollout accuracies
        "avg_pass":        round(avg_pass, 4),
        "avg_pass_pct":    f"{avg_pass * 100:.2f}%",
        # raw rollout counts
        "total_correct_rollouts": total_correct_rollouts,
        "total_judged_rollouts":  total_judged_rollouts,
        "unknown_rollouts":       unknown_rollouts,
    }

    level_summary = build_level_summary(scored)
    summary["difficulty_level"] = level_summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": scored}, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Questions  : {total}  ×  {n_rollouts} rollouts")
    print(f"Pass@{n_rollouts}     : {pass_n}/{total}  =  {pass_at_n_pct * 100:.2f}%"
          f"  (≥1 rollout correct)")
    print(f"Avg.Pass   : {avg_pass * 100:.2f}%"
          f"  ({total_correct_rollouts}/{total_judged_rollouts} rollouts correct)")
    print(f"Unknown    : {unknown_rollouts} rollouts  (no gold answer or judge error)")
    if level_summary:
        print("Difficulty:")
        for level, stats in level_summary.items():
            print(
                f"  - {level:<6} : {stats['correct_questions']}/{stats['total_questions']}"
                f"  =  {stats['accuracy_pct']}  (Pass@{n_rollouts})"
            )
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
