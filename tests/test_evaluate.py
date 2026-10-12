"""
test_evaluate.py — Evaluation gates, judge parsing and CI behaviour.
Run: pytest tests/test_evaluate.py -v
"""

import json

import pytest

from src import evaluate
from src.config import DEFAULTS
from src.evaluate import Gate, compute_gates, parse_judge_json, percentile, run_evaluation
from tests.fakes import FakeLLM, fake_answer

EVAL_CFG = DEFAULTS["evaluation"]


class TestJudgeParsing:
    def test_plain_json(self):
        out = parse_judge_json('{"faithfulness": 0.9, "correctness": 0.8, "reason": "ok"}')
        assert out == {"faithfulness": 0.9, "correctness": 0.8, "reason": "ok"}

    def test_json_wrapped_in_prose_and_clamped(self):
        out = parse_judge_json('Sure! ```json\n{"faithfulness": 1.4, "correctness": -0.2}\n```')
        assert out["faithfulness"] == 1.0 and out["correctness"] == 0.0

    def test_no_json(self):
        with pytest.raises(ValueError):
            parse_judge_json("I refuse")


class TestGates:
    def test_percentile(self):
        assert percentile([10, 20, 30, 40], 50) == 20
        assert percentile(list(range(1, 101)), 95) == 95
        assert percentile([], 95) == 0.0

    def test_gate_directions(self):
        assert Gate("cov", 0.95, 0.9).passed
        assert not Gate("cov", 0.85, 0.9).passed
        assert Gate("lat", 100, 8000, higher_is_better=False).passed
        assert not Gate("lat", 9000, 8000, higher_is_better=False).passed
        assert Gate("skipped", None, 0.9).passed

    def test_compute_gates(self):
        results = [
            {"status": "ok", "citation_coverage": 1.0, "faithfulness": 0.9, "correctness": 0.8, "latency_ms": 900},
            {"status": "error", "citation_coverage": 0.0, "latency_ms": 0},
        ]
        gates = {g.name: g for g in compute_gates(results, EVAL_CFG, judged=True)}
        assert gates["Citation coverage"].passed
        assert gates["Error rate"].value == 0.5 and not gates["Error rate"].passed

    def test_judge_gates_skipped_without_judge(self):
        results = [{"status": "ok", "citation_coverage": 1.0, "latency_ms": 10}]
        gates = compute_gates(results, EVAL_CFG, judged=False)
        assert [g.skipped for g in gates] == [False, True, True, False, False]


class FakeOrchestrator:
    def __init__(self, answer, judge_reply):
        self.answer = answer
        self.generator = type("G", (), {"llm": FakeLLM([judge_reply])})()

    def query(self, q):
        return self.answer


def _config(tmp_path, n_questions=2):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"question": f"Q{i}?", "expected_answer": "A"} for i in range(n_questions)]))
    return {
        "evaluation": {**EVAL_CFG, "golden_qa_path": str(golden), "results_path": str(tmp_path / "out" / "results.json")},
        "generation": {"provider": "fake", "model": "m"},
    }


def test_passing_run_writes_results(tmp_path):
    cfg = _config(tmp_path)
    orch = FakeOrchestrator(fake_answer(), '{"faithfulness": 1.0, "correctness": 0.9, "reason": "good"}')
    assert run_evaluation(cfg, orchestrator=orch) is True
    data = json.loads((tmp_path / "out" / "results.json").read_text())
    assert data["summary"]["all_gates_passed"] is True
    assert len(data["details"]) == 2 and data["details"][0]["faithfulness"] == 1.0
    assert len(data["summary"]["gates"]) == 5


def test_failing_run_still_writes_results(tmp_path):
    cfg = _config(tmp_path)
    orch = FakeOrchestrator(fake_answer(), '{"faithfulness": 0.3, "correctness": 0.9}')
    assert run_evaluation(cfg, orchestrator=orch) is False
    summary = json.loads((tmp_path / "out" / "results.json").read_text())["summary"]
    failed = [g["name"] for g in summary["gates"] if not g["passed"]]
    assert failed == ["Faithfulness (LLM judge)"]


def test_ci_mode_exits_nonzero_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "load_config", lambda path=None: _config(tmp_path))
    monkeypatch.setattr(evaluate, "run_evaluation", lambda *a, **k: False)
    monkeypatch.setattr("sys.argv", ["evaluate", "--ci"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1
