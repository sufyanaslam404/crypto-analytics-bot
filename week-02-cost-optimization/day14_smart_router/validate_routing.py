# smart_router/validate_routing.py

import sys
import os
import json
import re

# Add parent directory to path so we can import eval_framework from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_framework import load_eval_set, run_eval
from router import TASK_MODEL_MAP, client

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text

def make_predict_fn(task_type: str, system_prompt: str):
    def predict(input_text):
        try:
            response = client.chat.completions.create(
                model=TASK_MODEL_MAP[task_type],
                max_tokens=100, 
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return json.loads(extract_json(response.choices[0].message.content))
        except Exception as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                print(f"  [API Quota Exhausted for model {TASK_MODEL_MAP[task_type]}]", end=" ")
                # Return dummy failing result to prevent total crash
                return {}
            raise e
            
    return predict

def validate_all_routes(eval_registry: dict) -> bool:
    """
    Run the eval suite against EACH task type's routed model.
    Only tasks with a passing eval are considered validated for production use.
    """
    all_valid = True
    print("\n=== Validating Router Decisions Against Eval Sets ===\n")

    for task_type, config in eval_registry.items():
        # Evaluate relative to the parent directory where eval_sets lives
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_path = os.path.join(parent_dir, config["eval_set"])
        
        cases = load_eval_set(eval_path)
        predict_fn = make_predict_fn(task_type, config["system_prompt"])
        
        print(f"Testing {task_type} against {len(cases)} cases...")
        report = run_eval(cases, predict_fn)

        status = "✓ VALIDATED" if report.accuracy >= config.get("min_accuracy", 0.9) else "✗ NEEDS REVIEW"
        print(f"  {task_type:<25} {report.accuracy:.0%} accuracy — {status} "
              f"(routed to {TASK_MODEL_MAP[task_type]})")

        if report.accuracy < config.get("min_accuracy", 0.9):
            all_valid = False

    return all_valid


if __name__ == "__main__":
    # Example usage for the single task we have eval data for right now
    REGISTRY = {
        "headline_sentiment": {
            "eval_set": "eval_sets/headline_sentiment.json",
            "system_prompt": "You are a precise crypto market news analyst.\nClassify sentiment using these conventions: exchange inflows=bearish, outflows=bullish, token unlocks=bearish.\nRespond with ONLY valid JSON: {\"sentiment\": \"bullish|bearish|neutral\"}",
            "min_accuracy": 0.8
        }
    }
    
    validate_all_routes(REGISTRY)
