from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

@dataclass
class RoutingDecision:
    task_id: str
    task_type: str
    input_text: str
    complexity: str          # "simple" | "moderate" | "complex"
    model_chosen: str
    reasoning: str            # why the router picked this model

@dataclass
class ExecutionResult:
    task_id: str
    output: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    actual_cost: float
    baseline_cost: float      # what it WOULD have cost on the biggest model
    savings: float
    latency_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
