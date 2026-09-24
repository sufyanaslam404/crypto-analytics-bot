# smart_router/run_demo.py

import sys
import os
import uuid

# Add parent directory to path so we can import eval_framework from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router import classify_and_route, execute_task
from dashboard import RouterDashboard
from eval_framework import load_eval_set, run_eval   # Day 13

dashboard = RouterDashboard()

SYSTEM_PROMPTS = {
    "headline_sentiment": """You are a precise crypto market news analyst.
Classify sentiment using these conventions: exchange inflows=bearish, outflows=bullish, 
token unlocks=bearish, institutional adoption=bullish.
Respond with ONLY valid JSON: {"sentiment": "bullish|bearish|neutral"}""",

    "trade_setup_grading": """You are a senior ICT/SMC trading mentor. Grade setup quality 
honestly: A+ (4+ confluences), B (2-3), C (1), skip (0 or conflicting signals).
Respond with ONLY valid JSON: {"setup_quality": "A+|B|C|skip", "reasoning": "one sentence"}""",

    "contract_security_audit": """You are a senior smart contract auditor. Identify the 
single most severe vulnerability in the given code snippet.
Respond with ONLY valid JSON: {"severity": "critical|high|medium|low|none", "finding": "one sentence"}""",
}

# A realistic mixed batch — simulating one operating day for a small product
workload = [
    ("headline_sentiment", "BlackRock ETF sees record inflows this week"),
    ("headline_sentiment", "Whale moves 50,000 ETH to exchange, market watches closely"),
    ("headline_sentiment", "Ethereum devs discuss routine protocol maintenance"),
    ("headline_sentiment", "Token unlocks scheduled, releasing 15% of circulating supply"),
    ("trade_setup_grading", "Gold swept Asian session low, CHoCH on M5, tapped bullish "
     "order block, London open, but H1 bias has been bearish for 2 days"),
    ("trade_setup_grading", "BTC broke previous day high with no liquidity sweep, no CHoCH, "
     "entering purely on a round number"),
    ("contract_security_audit", "function withdraw(uint amount) public { "
     "(bool ok,) = msg.sender.call{value: amount}(''); require(ok); "
     "balances[msg.sender] -= amount; }"),
    ("headline_sentiment", "Major bank announces institutional crypto custody partnership"),
]

if __name__ == "__main__":
    print("Processing workload through Smart Router...\n")

    for task_type, input_text in workload:
        task_id = f"{task_type[:4]}-{uuid.uuid4().hex[:6]}"

        decision = classify_and_route(task_id, task_type, input_text)
        
        try:
            result = execute_task(decision, SYSTEM_PROMPTS[task_type])
            dashboard.log(decision, result)
            print(f"  [{task_id}] {task_type:<25} → {decision.model_chosen:<20} "
                  f"(${result.actual_cost:.6f}, saved ${result.savings:.6f})")
        except Exception as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                print(f"  [{task_id}] {task_type:<25} → {decision.model_chosen:<20} FAILED (OpenAI API Quota Exhausted)")
            else:
                print(f"  [{task_id}] {task_type:<25} → {decision.model_chosen:<20} FAILED ({str(e)})")

    dashboard.print_final_report()
