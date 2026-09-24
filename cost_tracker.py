#!/usr/bin/env python3
"""
cost_tracker.py — LLM API Cost Tracker & Session Billing

A decorator-based cost tracker for OpenAI API calls. Wraps any function
that returns an OpenAI ChatCompletion response, extracts actual token
usage from the response, and estimates cost based on per-model pricing.

Supports both OpenAI and Anthropic pricing tables.
"""

import functools
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# PRICING TABLE — USD per 1M tokens (easily updated)
# ═══════════════════════════════════════════════════════════

PRICING = {
    # OpenAI models (per 1M tokens)
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":     {"input": 10.00, "output": 30.00},
    "gpt-4":           {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":   {"input": 0.50,  "output": 1.50},
    "o1":              {"input": 15.00, "output": 60.00},
    "o1-mini":         {"input": 3.00,  "output": 12.00},
    "o3-mini":         {"input": 1.10,  "output": 4.40},

    # Anthropic models (per 1M tokens)
    "claude-sonnet-4-20250514":   {"input": 3.00,  "output": 15.00},
    "claude-sonnet-5":            {"input": 3.00,  "output": 15.00},
    "claude-haiku-3.5":           {"input": 0.80,  "output": 4.00},
    "claude-opus-4-20250514":     {"input": 15.00, "output": 75.00},
}


# ═══════════════════════════════════════════════════════════
# COST ESTIMATION
# ═══════════════════════════════════════════════════════════

@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    """
    Calculate the estimated cost given a model and token counts.
    Falls back to gpt-4o-mini pricing if model is unknown.
    """
    pricing = PRICING.get(model)
    if not pricing:
        # Try partial match (e.g., "gpt-4o-mini-2024-07-18" → "gpt-4o-mini")
        for key in PRICING:
            if key in model or model.startswith(key):
                pricing = PRICING[key]
                break

    if not pricing:
        print(f"  ⚠ Unknown model '{model}', using gpt-4o-mini pricing as fallback")
        pricing = PRICING["gpt-4o-mini"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost
    )


# ═══════════════════════════════════════════════════════════
# SESSION TRACKER (global, accumulates across all calls)
# ═══════════════════════════════════════════════════════════

@dataclass
class CostEntry:
    timestamp: float
    function: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float

_cost_log: list[CostEntry] = []


def _extract_usage_openai(response):
    """Extract token usage from an OpenAI ChatCompletion response."""
    return response.usage.prompt_tokens, response.usage.completion_tokens


def _extract_usage_anthropic(response):
    """Extract token usage from an Anthropic Message response."""
    return response.usage.input_tokens, response.usage.output_tokens


def track_cost(model: str, provider: Literal["openai", "anthropic"] = "openai"):
    """
    Decorator that extracts real token usage from an LLM API response,
    estimates cost, and logs it to the session tracker.

    Usage:
        @track_cost(model="gpt-4o-mini")
        def my_llm_call(...):
            return client.chat.completions.create(...)

        @track_cost(model="claude-sonnet-5", provider="anthropic")
        def my_anthropic_call(...):
            return client.messages.create(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            response = func(*args, **kwargs)

            # Extract actual token usage from the response
            if provider == "anthropic":
                input_tokens, output_tokens = _extract_usage_anthropic(response)
            else:
                input_tokens, output_tokens = _extract_usage_openai(response)

            est = estimate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            _cost_log.append(CostEntry(
                timestamp=time.time(),
                function=func.__name__,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=est.total_cost
            ))

            return response
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════

def get_session_costs() -> list[dict]:
    """Return all cost entries as a list of dicts."""
    return [
        {
            "timestamp": e.timestamp,
            "function": e.function,
            "model": e.model,
            "input_tokens": e.input_tokens,
            "output_tokens": e.output_tokens,
            "cost": e.cost
        }
        for e in _cost_log
    ]


def get_total_cost() -> float:
    """Return the total accumulated cost for this session."""
    return sum(e.cost for e in _cost_log)


def reset_session():
    """Clear all tracked costs."""
    _cost_log.clear()


def print_session_costs():
    """Pretty-print a full cost breakdown for the session."""
    total = get_total_cost()
    total_input = sum(e.input_tokens for e in _cost_log)
    total_output = sum(e.output_tokens for e in _cost_log)

    print(f"\n{'═'*60}")
    print(f"  💰  SESSION COST SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Function':<28} {'Model':<16} {'Tokens':>10}   {'Cost':>10}")
    print(f"  {'─'*28} {'─'*16} {'─'*10}   {'─'*10}")

    for entry in _cost_log:
        tokens = f"{entry.input_tokens + entry.output_tokens:,}"
        print(f"  {entry.function:<28} {entry.model:<16} {tokens:>10}   ${entry.cost:.6f}")

    print(f"  {'─'*66}")
    print(f"  {'TOTAL':<28} {'':<16} {total_input + total_output:>10,}   ${total:.6f}")
    print(f"{'═'*60}")
    print(f"  Input tokens:  {total_input:>10,}")
    print(f"  Output tokens: {total_output:>10,}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════
# STANDALONE DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()

    @track_cost(model="gpt-4o-mini")
    def analyze_headline(headline: str):
        return client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": f"One-word sentiment (bullish/bearish/neutral) of this crypto headline: {headline}"}]
        )

    @track_cost(model="gpt-4o-mini")
    def quick_summary(text: str):
        return client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=80,
            temperature=0,
            messages=[{"role": "user", "content": f"Summarize in one sentence: {text}"}]
        )

    # Run a few calls, then check the running total
    print("Running cost-tracked API calls...\n")

    r1 = analyze_headline("BTC ETF sees record inflows")
    print(f"  → {r1.choices[0].message.content.strip()}")

    r2 = analyze_headline("Exchange reports security incident")
    print(f"  → {r2.choices[0].message.content.strip()}")

    r3 = quick_summary("Ethereum Layer 2 solutions are scaling the network by processing transactions off-chain and posting proofs to mainnet.")
    print(f"  → {r3.choices[0].message.content.strip()}")

    print_session_costs()
