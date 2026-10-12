"""
evaluate.py — Automated Evaluation Pipeline with CI Gating

Runs every question in the golden Q/A set through the full RAG pipeline and
scores the system on five quality gates (thresholds in configs/config.yaml):

  1. Citation coverage   — share of answer sentences that cite a chunk   (rule-based)
  2. Faithfulness        — share of answer claims supported by the
                           retrieved chunks                              (LLM-as-judge)
  3. Answer correctness  — agreement with the golden reference answer    (LLM-as-judge)
  4. Latency P95         — end-to-end pipeline latency
  5. Error rate          — share of questions that failed outright

Results are always written to evals/results.json (uploaded as a CI artifact).
In CI mode (--ci) the process exits 1 if any gate fails, which blocks the PR.

Usage:
    python -m src.evaluate
    python -m src.evaluate --ci
    python -m src.evaluate --no-judge      # skip the LLM-as-judge gates (no extra API calls)
"""

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config

console = Console()

JUDGE_SYSTEM = (
    "You are a strict evaluator of retrieval-augmented answers. "
    "You reply with a single JSON object and nothing else."
)

JUDGE_TEMPLATE = """Evaluate the ANSWER to the QUESTION.

QUESTION:
{question}

RETRIEVED CONTEXT (the only evidence the answer may use):
{context}

REFERENCE ANSWER (ground truth):
{reference}

ANSWER:
{answer}

Score two things from 0.0 to 1.0:
- "faithfulness": the fraction of factual claims in the ANSWER that are directly supported by the RETRIEVED CONTEXT. Claims not found in the context count against it. An answer that correctly says the context is insufficient, and makes no other claims, scores 1.0.
- "correctness": how well the ANSWER covers the key facts of the REFERENCE ANSWER (1.0 = all key facts, 0.0 = none or contradicts it). Extra correct detail is fine.

Reply with JSON only: {{"faithfulness": <float>, "correctness": <float>, "reason": "<one short sentence>"}}"""


@dataclass
class Gate:
    name: str
    value: float | None
    threshold: float
    higher_is_better: bool = True

    @property
    def skipped(self) -> bool:
        return self.value is None

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return self.value >= self.threshold if self.higher_is_better else self.value <= self.threshold


def load_golden_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def parse_judge_json(text: str) -> dict:
    """Pull the first JSON object out of a judge reply and clamp scores to [0, 1]."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge reply had no JSON: {text[:200]!r}")
    data = json.loads(match.group(0))
    out = {}
    for key in ("faithfulness", "correctness"):
        out[key] = min(1.0, max(0.0, float(data[key])))
    out["reason"] = str(data.get("reason", ""))
    return out


def judge_answer(llm, model: str, question: str, context: str, reference: str, answer: str) -> dict:
    completion = llm.complete(
        system=JUDGE_SYSTEM,
        user=JUDGE_TEMPLATE.format(question=question, context=context,
                                   reference=reference, answer=answer),
        model=model,
        temperature=0.0,
        max_tokens=300,
    )
    return parse_judge_json(completion.text)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def compute_gates(results: list[dict], eval_cfg: dict, judged: bool) -> list[Gate]:
    ok = [r for r in results if r["status"] == "ok"]
    n_total = len(results)

    def mean(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return [
        Gate("Citation coverage", mean("citation_coverage") if ok else 0.0,
             eval_cfg["citation_coverage_threshold"]),
        Gate("Faithfulness (LLM judge)", (mean("faithfulness") if ok else 0.0) if judged else None,
             eval_cfg["faithfulness_threshold"]),
        Gate("Answer correctness (LLM judge)", (mean("correctness") if ok else 0.0) if judged else None,
             eval_cfg["answer_correctness_threshold"]),
        Gate("Latency P95 (ms)", percentile([r["latency_ms"] for r in ok], 95) if ok else 0.0,
             eval_cfg["latency_p95_ms_threshold"], higher_is_better=False),
        Gate("Error rate", (n_total - len(ok)) / n_total if n_total else 1.0,
             eval_cfg["max_error_rate"], higher_is_better=False),
    ]


def write_results(path: str | Path, results: list[dict], gates: list[Gate], extra: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        **extra,
        "all_gates_passed": all(g.passed for g in gates),
        "gates": [
            {"name": g.name, "value": g.value, "threshold": g.threshold,
             "higher_is_better": g.higher_is_better, "passed": g.passed, "skipped": g.skipped}
            for g in gates
        ],
    }
    with open(output, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    return output


def run_evaluation(config: dict, ci_mode: bool = False, use_judge: bool = True,
                   orchestrator=None) -> bool:
    """Run the full evaluation. Returns True when every gate passes."""
    eval_cfg = config["evaluation"]
    golden_path = eval_cfg["golden_qa_path"]

    if not Path(golden_path).exists():
        console.print(f"[red]✗ Golden dataset not found: {golden_path}[/red]")
        return False

    dataset = load_golden_dataset(golden_path)
    console.print(f"\n[bold]📊 Evaluating on {len(dataset)} questions...[/bold]\n")

    if orchestrator is None:
        from src.orchestrator import RAGOrchestrator
        orchestrator = RAGOrchestrator(config)
    llm = orchestrator.generator.llm
    model = config["generation"]["model"]

    results = []
    for i, item in enumerate(dataset):
        query = item["question"]
        console.print(f"  [{i + 1}/{len(dataset)}] {query[:65]}...")
        try:
            start = time.perf_counter()
            answer = orchestrator.query(query)
            total_ms = (time.perf_counter() - start) * 1000
            row = {
                "question": query,
                "answer": answer.answer,
                "citations": answer.citations,
                "invalid_citations": answer.invalid_citations,
                "citation_coverage": answer.citation_coverage,
                "latency_ms": total_ms,
                "tokens": answer.total_tokens,
                "chunks": len(answer.chunks_used),
                "status": "ok",
            }
            if use_judge:
                from src.generator import Generator
                verdict = judge_answer(
                    llm, model, query, Generator.build_context(answer.chunks_used),
                    item.get("expected_answer", ""), answer.answer,
                )
                row.update(verdict)
            results.append(row)
        except Exception as e:  # one bad question shouldn't hide the rest of the report
            results.append({"question": query, "status": "error", "error": str(e),
                            "citation_coverage": 0.0, "latency_ms": 0})

    gates = compute_gates(results, eval_cfg, judged=use_judge)
    ok = [r for r in results if r["status"] == "ok"]
    latencies = [r["latency_ms"] for r in ok]
    extra = {
        "total_questions": len(results),
        "errors": len(results) - len(ok),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "avg_tokens": sum(r.get("tokens", 0) for r in ok) / len(ok) if ok else 0,
        "model": f"{config['generation'].get('provider')}/{model}",
        "judge": f"{config['generation'].get('provider')}/{model}" if use_judge else None,
    }
    # Write results BEFORE deciding the exit code, so CI always has the artifact.
    output = write_results(eval_cfg.get("results_path", "evals/results.json"), results, gates, extra)

    _print_report(results, gates)
    console.print(f"[dim]Results written to {output}[/dim]\n")

    all_pass = all(g.passed for g in gates)
    if all_pass:
        console.print("[bold green]✅ All quality gates passed.[/bold green]\n")
    else:
        console.print("[bold red]❌ Quality gates FAILED — PR should be blocked.[/bold red]\n")
    return all_pass


def _fmt(gate: Gate, value: float | None) -> str:
    if value is None:
        return "—"
    if "ms" in gate.name:
        return f"{value:.0f}"
    return f"{value:.1%}"


def _print_report(results: list[dict], gates: list[Gate]):
    table = Table(title="Quality Gates", show_lines=True)
    table.add_column("Gate", style="cyan", width=32)
    table.add_column("Value", width=10)
    table.add_column("Threshold", style="yellow", width=12)
    table.add_column("Status", width=8)
    for g in gates:
        op = "≥" if g.higher_is_better else "≤"
        status = "[dim]SKIP[/dim]" if g.skipped else ("[green]PASS[/green]" if g.passed else "[red]FAIL[/red]")
        table.add_row(g.name, _fmt(g, g.value), f"{op} {_fmt(g, g.threshold)}", status)
    console.print()
    console.print(table)

    detail = Table(title="Per-Question Results", show_lines=True)
    for col, width in (("#", 4), ("Question", 40), ("Coverage", 9), ("Faithful", 9),
                       ("Correct", 8), ("Latency", 9), ("Status", 7)):
        detail.add_column(col, width=width)
    for i, r in enumerate(results):
        if r["status"] == "ok":
            pct = lambda k: f"{r[k]:.0%}" if r.get(k) is not None else "—"
            detail.add_row(str(i + 1), r["question"][:40], pct("citation_coverage"),
                           pct("faithfulness"), pct("correctness"),
                           f"{r['latency_ms']:.0f}ms", "[green]OK[/green]")
        else:
            detail.add_row(str(i + 1), r["question"][:40], "—", "—", "—", "—", "[red]ERR[/red]")
    console.print(detail)


def main():
    parser = argparse.ArgumentParser(description="Run DocuQuery evaluation pipeline")
    parser.add_argument("--ci", action="store_true", help="CI mode — exit non-zero on failure")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge gates")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    passed = run_evaluation(config, ci_mode=args.ci, use_judge=not args.no_judge)
    if args.ci and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
