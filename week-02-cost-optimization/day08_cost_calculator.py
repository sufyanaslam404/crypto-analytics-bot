# day8_cost_calculator.py
"""
Complete cost calculator — estimate, project, and compare LLM costs
before writing production code.
"""

import tiktoken
from dataclasses import dataclass

enc = tiktoken.get_encoding("cl100k_base")

PRICING = {
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
    "claude-sonnet-5":   {"input": 2.00,  "output": 10.00},
    "claude-opus-5":     {"input": 5.00,  "output": 25.00},
}

BATCH_DISCOUNT = 0.50
CACHE_HIT_MULTIPLIER = 0.10


@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    total_cost: float
    calls: int = 1

    @property
    def total_for_all_calls(self) -> float:
        return self.total_cost * self.calls


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def estimate_cost(model, input_text=None, input_tokens=None, output_tokens=500,
                   calls=1, use_batch=False, cached_input_tokens=0):
    if model not in PRICING:
        raise ValueError(f"Unknown model '{model}'")
    
    if input_text is not None:
        input_tokens = count_tokens(input_text)
    
    rates = PRICING[model]
    input_rate = rates["input"] * (BATCH_DISCOUNT if use_batch else 1)
    output_rate = rates["output"] * (BATCH_DISCOUNT if use_batch else 1)
    
    fresh_tokens = input_tokens - cached_input_tokens
    input_cost = (fresh_tokens / 1_000_000 * input_rate + 
                  cached_input_tokens / 1_000_000 * input_rate * CACHE_HIT_MULTIPLIER)
    output_cost = output_tokens / 1_000_000 * output_rate
    
    return CostEstimate(model, input_tokens, output_tokens, input_cost + output_cost, calls)


def project_monthly(estimate, calls_per_day):
    daily = estimate.total_cost * calls_per_day
    return {"daily": daily, "monthly": daily * 30, "yearly": daily * 365}


def compare_models(input_text, output_tokens, models, calls_per_day=100):
    print(f"\n{'Model':<22}{'Per Call':>14}{'Monthly':>15}")
    results = []
    for model in models:
        est = estimate_cost(model, input_text=input_text, output_tokens=output_tokens)
        monthly = est.total_cost * calls_per_day * 30
        results.append((model, est.total_cost, monthly))
        print(f"{model:<22}${est.total_cost:>12.6f}${monthly:>14.2f}")
    return results


if __name__ == "__main__":
    print("="*60)
    print("DAY 8 — COST CALCULATOR DEMO")
    print("="*60)

    # 1. Single call estimate
    est = estimate_cost("claude-sonnet-5", input_tokens=800, output_tokens=300)
    print(f"\nSingle trade analysis call: ${est.total_cost:.6f}")

    # 2. Monthly projection for a 24/7 bot
    proj = project_monthly(est, calls_per_day=96)
    print(f"24/7 bot (96 calls/day) → ${proj['monthly']:.2f}/month")

    # 3. Model comparison at scale
    print("\nComparing models for 500 headlines/day classification:")
    compare_models(
        input_text="Classify sentiment: SEC delays Bitcoin ETF decision",
        output_tokens=60,
        models=["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"],
        calls_per_day=500
    )
