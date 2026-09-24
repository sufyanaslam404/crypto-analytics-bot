# smart_router/router.py

import json
import re
import time
import os
from openai import OpenAI, RateLimitError
from models import RoutingDecision, ExecutionResult
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text

# ─────────────────────────────────────────────────────────
# CURRENT PRICING — verify against provider docs periodically
# OpenAI pricing per 1M tokens
# ─────────────────────────────────────────────────────────

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "o1-preview":  {"input": 15.00, "output": 60.00},
}

BASELINE_MODEL = "o1-preview"  # the "just use the best model always" comparison point

# ─────────────────────────────────────────────────────────
# STATIC ROUTING TABLE — extend as you register new task types
# ─────────────────────────────────────────────────────────

TASK_MODEL_MAP = {
    "headline_sentiment":      "gpt-4o-mini",
    "whitepaper_chunk_review": "gpt-4o-mini",
    "trade_setup_grading":     "gpt-4o",
    "contract_security_audit": "gpt-4o",
    "whitepaper_synthesis":    "gpt-4o",
}

# ─────────────────────────────────────────────────────────
# DYNAMIC ROUTING — used when task_type isn't registered
# ─────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are a task complexity classifier. Classify the reasoning 
capability needed for this request.

- "simple": classification, short extraction, formatting, single-fact lookup
- "moderate": multi-step reasoning, synthesis of multiple inputs, requires judgment
- "complex": high-stakes analysis, ambiguous judgment calls, multi-step planning

Respond with ONLY valid JSON: {"complexity": "simple|moderate|complex", "reasoning": "one short phrase"}"""

COMPLEXITY_MODEL_MAP = {
    "simple":   "gpt-4o-mini",
    "moderate": "gpt-4o",
    "complex":  "o1-preview",
}


def classify_and_route(task_id: str, task_type: str, input_text: str) -> RoutingDecision:
    """
    Routing entry point. Checks the static map first (fast, free, deterministic).
    Falls back to dynamic classification only for unregistered task types.
    """
    if task_type in TASK_MODEL_MAP:
        return RoutingDecision(
            task_id=task_id,
            task_type=task_type,
            input_text=input_text,
            complexity="static",
            model_chosen=TASK_MODEL_MAP[task_type],
            reasoning=f"Static mapping for registered task type '{task_type}'"
        )

    # Unregistered task type — classify dynamically using the cheapest model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=40,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": input_text}
        ]
    )
    result = json.loads(extract_json(response.choices[0].message.content))
    complexity = result.get("complexity", "moderate")

    return RoutingDecision(
        task_id=task_id,
        task_type=task_type,
        input_text=input_text,
        complexity=complexity,
        model_chosen=COMPLEXITY_MODEL_MAP[complexity],
        reasoning=result.get("reasoning", "")
    )


def execute_task(decision: RoutingDecision, system_prompt: str, max_tokens: int = 300) -> ExecutionResult:
    """
    Execute the actual task on the model the router selected, with retry logic,
    then compute real cost using the API's own usage data.
    """
    start = time.time()

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=decision.model_chosen,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": decision.input_text}
                ]
            )
            break
        except RateLimitError as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                raise
            wait = 2 ** attempt
            time.sleep(wait)
    else:
        raise RuntimeError(f"Task {decision.task_id} failed after retries")

    latency = time.time() - start

    usage = response.usage
    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) if hasattr(usage, "prompt_tokens_details") else 0
    fresh_tokens = usage.prompt_tokens - cached_tokens
    
    # OpenAI automatically caches prompts > 1024 tokens. 
    # For this script we map "cache_read" to what OpenAI calls "cached_tokens"
    actual_cost = _compute_cost(decision.model_chosen, fresh_tokens, usage.completion_tokens, cached_tokens)

    # Baseline: what would this SAME call have cost on the biggest model (o1-preview doesn't support system prompts right now, but assuming it costs standard rate)
    baseline_cost = _compute_cost(BASELINE_MODEL, usage.prompt_tokens, usage.completion_tokens, 0)

    return ExecutionResult(
        task_id=decision.task_id,
        output=response.choices[0].message.content,
        model_used=decision.model_chosen,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=cached_tokens,
        cache_write_tokens=0, # OpenAI doesn't explicitly charge for cache writes
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        savings=baseline_cost - actual_cost,
        latency_seconds=latency
    )


def _compute_cost(model: str, fresh_input_tokens: int, output_tokens: int, cached_input_tokens: int) -> float:
    rates = PRICING.get(model, PRICING["gpt-4o-mini"])
    
    # OpenAI gives a 50% discount on cached input tokens
    cost = (fresh_input_tokens / 1_000_000) * rates["input"]
    cost += (cached_input_tokens / 1_000_000) * (rates["input"] * 0.50)
    cost += (output_tokens / 1_000_000) * rates["output"]

    return cost
