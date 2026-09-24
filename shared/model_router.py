# model_router.py
"""
Reusable model routing logic for the rest of the course.
Combines static task-type routing with dynamic confidence-based escalation.
"""

import json
import re
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError
from typing import Literal, Optional

client = Anthropic()

# ─────────────────────────────────────────────────────────
# STATIC ROUTING TABLE — extend this as you build new features
# ─────────────────────────────────────────────────────────

TASK_MODEL_MAP = {
    "headline_sentiment":      "gpt-4o-mini",
    "whitepaper_chunk_review": "gpt-4o-mini",
    "whitepaper_synthesis":    "gpt-4o",
    "trade_setup_grading":     "gpt-4o",
    "contract_security_audit": "gpt-4o",
    "agent_planning":          "gpt-4o",
    "deep_audit_final_pass":   "o1-preview",
}

def get_model_for_task(task_type: str) -> str:
    if task_type not in TASK_MODEL_MAP:
        raise ValueError(
            f"Unknown task type '{task_type}'. "
            f"Register it in TASK_MODEL_MAP or use classify_task_complexity() instead."
        )
    return TASK_MODEL_MAP[task_type]


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


# ─────────────────────────────────────────────────────────
# DYNAMIC ROUTING — for tasks whose difficulty varies per-input
# ─────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a task complexity classifier. Given a user request, 
classify how much reasoning capability is needed to handle it well.

Classify as:
- "simple": classification, short extraction, formatting, single-fact lookup
- "moderate": multi-step reasoning, synthesis of multiple inputs, requires judgment
- "complex": high-stakes analysis, ambiguous judgment calls, multi-step planning

Respond with ONLY valid JSON:
{"complexity": "simple|moderate|complex"}"""

COMPLEXITY_MODEL_MAP = {
    "simple":   "claude-haiku-4-5",
    "moderate": "claude-sonnet-5",
    "complex":  "claude-opus-5",
}

def classify_task_complexity(user_request: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=30,
        temperature=0,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_request}]
    )
    result = json.loads(extract_json(response.content[0].text))
    return result["complexity"]

def route_dynamically(user_request: str) -> str:
    complexity = classify_task_complexity(user_request)
    return COMPLEXITY_MODEL_MAP[complexity]


# ─────────────────────────────────────────────────────────
# ESCALATION PATTERN — cheap model first, escalate on low confidence
# ─────────────────────────────────────────────────────────

def call_with_escalation(
    user_message: str,
    cheap_system: str,
    expensive_system: str,
    response_model: type[BaseModel],
    confidence_field: str = "confidence",
    confidence_threshold: float = 0.7,
    cheap_model: str = "claude-haiku-4-5",
    expensive_model: str = "claude-sonnet-5",
    cheap_max_tokens: int = 150,
    expensive_max_tokens: int = 300,
) -> tuple[BaseModel, str]:
    """
    Generic escalation caller. Requires your response_model (a Pydantic class)
    to include a confidence field. Falls back to the expensive model if the
    cheap model's confidence is below threshold, or if parsing fails.
    """
    try:
        response = client.messages.create(
            model=cheap_model,
            max_tokens=cheap_max_tokens,
            temperature=0,
            system=cheap_system,
            messages=[{"role": "user", "content": user_message}]
        )
        data = json.loads(extract_json(response.content[0].text))
        result = response_model(**data)
        
        if getattr(result, confidence_field) >= confidence_threshold:
            return result, cheap_model
    except (json.JSONDecodeError, ValidationError, KeyError):
        pass  # fall through to escalation
    
    response = client.messages.create(
        model=expensive_model,
        max_tokens=expensive_max_tokens,
        temperature=0,
        system=expensive_system,
        messages=[{"role": "user", "content": user_message}]
    )
    data = json.loads(extract_json(response.content[0].text))
    result = response_model(**data)
    return result, expensive_model
