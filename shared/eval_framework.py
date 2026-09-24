# eval_framework.py
"""
Reusable evaluation framework — add to prompt_library.py's family of 
shared modules. Grows alongside your project eval sets.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class EvalCase:
    id: str
    input: Any
    expected: dict
    notes: str = ""
    difficulty: str = "medium"

@dataclass
class EvalResult:
    case_id: str
    passed: bool
    predicted: Any
    expected: Any
    difficulty: str
    error: str = None

@dataclass
class EvalReport:
    total: int
    passed: int
    failed_cases: list = field(default_factory=list)
    by_difficulty: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def print_summary(self, label: str = "Eval Results"):
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        print(f"  Overall: {self.passed}/{self.total} ({self.accuracy:.0%})")
        
        if self.by_difficulty:
            print(f"\n  By difficulty:")
            for diff, (p, t) in self.by_difficulty.items():
                print(f"    {diff:<10} {p}/{t} ({p/t:.0%})")
        
        if self.failed_cases:
            print(f"\n  Failed cases:")
            for f in self.failed_cases:
                print(f"    [{f.case_id}] expected={f.expected}, got={f.predicted}")
        print(f"{'='*55}")


def load_eval_set(path: str) -> list[EvalCase]:
    with open(path) as f:
        raw_cases = json.load(f)
    return [EvalCase(**c) for c in raw_cases]


def run_eval(
    eval_cases: list[EvalCase],
    predict_fn: Callable[[Any], dict],
    match_fn: Callable[[dict, dict], bool] = None,
) -> EvalReport:
    """
    Run predict_fn on every case and compare against expected output.
    match_fn defaults to exact dict equality — override for fuzzier matching.
    """
    if match_fn is None:
        match_fn = lambda predicted, expected: predicted == expected

    results = []
    difficulty_tally = {}

    for case in eval_cases:
        try:
            predicted = predict_fn(case.input)
            passed = match_fn(predicted, case.expected)
        except Exception as e:
            predicted = None
            passed = False

        results.append(EvalResult(
            case_id=case.id, passed=passed, predicted=predicted,
            expected=case.expected, difficulty=case.difficulty
        ))

        d = case.difficulty
        p, t = difficulty_tally.get(d, (0, 0))
        difficulty_tally[d] = (p + passed, t + 1)

    passed_count = sum(r.passed for r in results)
    failed = [r for r in results if not r.passed]

    return EvalReport(
        total=len(results),
        passed=passed_count,
        failed_cases=failed,
        by_difficulty=difficulty_tally
    )
