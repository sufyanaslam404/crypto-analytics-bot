# run_full_eval_suite.py
"""
Run this before shipping any prompt change. Checks accuracy, regression, 
and constraint compliance across every eval-covered task in the project.
"""

from eval_framework import load_eval_set, run_eval
from eval_sentiment import predict_sentiment
from eval_regression import check_regression
from pathlib import Path

EVAL_SUITE = {
    "headline_sentiment": {
        "eval_set": "eval_sets/headline_sentiment.json",
        "predict_fn": predict_sentiment,
    },
    # Add more tasks here as you build them:
    # "trade_setup_grading": {"eval_set": "eval_sets/trade_setups.json", "predict_fn": predict_setup_grade},
    # "whitepaper_chunk_review": {"eval_set": "eval_sets/whitepaper_chunks.json", "predict_fn": predict_chunk_risk},
}

def run_full_suite(tolerance: float = 0.0):
    all_passed = True
    
    for task_name, config in EVAL_SUITE.items():
        if not Path(config["eval_set"]).exists():
            print(f"Skipping '{task_name}' — no eval set found at {config['eval_set']}")
            continue
        
        cases = load_eval_set(config["eval_set"])
        report = run_eval(cases, config["predict_fn"])
        report.print_summary(task_name)
        
        is_safe = check_regression(task_name, report, tolerance=tolerance)
        all_passed = all_passed and is_safe
    
    print(f"\n{'='*55}")
    print(f"  SUITE RESULT: {'PASS — safe to ship' if all_passed else 'FAIL — regressions detected'}")
    print(f"{'='*55}")
    return all_passed


if __name__ == "__main__":
    run_full_suite()
