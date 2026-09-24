# eval_regression.py
"""
Regression checking module for the evaluation framework.
Provides functions to save eval run baselines and check future runs against them 
to ensure prompt tweaks don't cause unintended accuracy drops.
"""

import json
from datetime import datetime
from pathlib import Path
from eval_framework import load_eval_set, run_eval
from eval_sentiment import predict_sentiment

BASELINE_FILE = "eval_baselines.json"

def save_baseline(task_name: str, report, prompt_version: str):
    """Record a passing eval run as the new baseline to compare future changes against."""
    baselines = {}
    if Path(BASELINE_FILE).exists():
        with open(BASELINE_FILE) as f:
            baselines = json.load(f)
    
    baselines[task_name] = {
        "accuracy": report.accuracy,
        "total": report.total,
        "passed": report.passed,
        "prompt_version": prompt_version,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=2)
    
    print(f"Baseline saved for '{task_name}': {report.accuracy:.0%} ({prompt_version})")


def check_regression(task_name: str, new_report, tolerance: float = 0.0) -> bool:
    """
    Compare a new eval run against the saved baseline.
    Returns True if no regression detected (safe to ship).
    tolerance: allowed accuracy drop before flagging as a regression 
               (0.0 = any drop at all is flagged)
    """
    if not Path(BASELINE_FILE).exists():
        print("No baseline found — treating this run as the new baseline.")
        return True
    
    with open(BASELINE_FILE) as f:
        baselines = json.load(f)
    
    if task_name not in baselines:
        print(f"No baseline for '{task_name}' yet — treating this run as the new baseline.")
        return True
    
    old = baselines[task_name]
    old_accuracy = old["accuracy"]
    new_accuracy = new_report.accuracy
    delta = new_accuracy - old_accuracy
    
    print(f"\n{'='*50}")
    print(f"  REGRESSION CHECK: {task_name}")
    print(f"{'='*50}")
    print(f"  Baseline ({old['prompt_version']}): {old_accuracy:.0%}")
    print(f"  Current run:                {new_accuracy:.0%}")
    print(f"  Delta:                      {delta:+.1%}")
    
    if delta < -tolerance:
        print(f"  ⚠ REGRESSION DETECTED — accuracy dropped beyond tolerance")
        print(f"{'='*50}")
        return False
    else:
        print(f"  ✓ No regression — safe to update baseline")
        print(f"{'='*50}")
        return True


if __name__ == "__main__":
    # ── The actual workflow you'll use going forward ───────────

    print("Step 1: Establishing initial baseline...")
    eval_cases = load_eval_set("eval_sets/headline_sentiment.json")
    report_v1 = run_eval(eval_cases, predict_sentiment)
    save_baseline("headline_sentiment", report_v1, prompt_version="v1-original")

    # Step 2: Imagine weeks later, you tweak SENTIMENT_SYSTEM for an unrelated reason
    # (Here we are just re-running it for the demo, which should get the same score)
    
    print("\nStep 3: Checking new run against baseline...")
    report_v2 = run_eval(eval_cases, predict_sentiment)  # using the MODIFIED prompt now
    
    is_safe = check_regression("headline_sentiment", report_v2, tolerance=0.0)

    if is_safe:
        save_baseline("headline_sentiment", report_v2, prompt_version="v2-tweaked")
    else:
        print("Do not ship this prompt change until the regression is understood and fixed.")
