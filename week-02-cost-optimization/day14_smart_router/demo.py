# smart_router/demo.py

import time
from models import RoutingDecision, ExecutionResult
from dashboard import RouterDashboard

if __name__ == "__main__":
    print("Running Smart Router Dashboard Demo...")
    dash = RouterDashboard()
    
    # We will mock the execution results since the API quota is exhausted
    # This demonstrates exactly how the dashboard formats the tracking data
    
    mock_data = [
        (
            RoutingDecision(task_id="tsk_001", task_type="headline_sentiment", input_text="BTC pumps", complexity="static", model_chosen="gpt-4o-mini", reasoning="Static mapping"),
            ExecutionResult(task_id="tsk_001", output="bullish", model_used="gpt-4o-mini", input_tokens=40, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0, actual_cost=0.000012, baseline_cost=0.0012, savings=0.001188, latency_seconds=0.4)
        ),
        (
            RoutingDecision(task_id="tsk_002", task_type="headline_sentiment", input_text="ETH drops", complexity="static", model_chosen="gpt-4o-mini", reasoning="Static mapping"),
            ExecutionResult(task_id="tsk_002", output="bearish", model_used="gpt-4o-mini", input_tokens=45, output_tokens=10, cache_read_tokens=0, cache_write_tokens=0, actual_cost=0.000013, baseline_cost=0.001275, savings=0.001262, latency_seconds=0.5)
        ),
        (
            RoutingDecision(task_id="tsk_003", task_type="unknown_task", input_text="Synthesize these 3 reports", complexity="moderate", model_chosen="gpt-4o", reasoning="Requires synthesis"),
            ExecutionResult(task_id="tsk_003", output="Summary...", model_used="gpt-4o", input_tokens=1200, output_tokens=350, cache_read_tokens=1024, cache_write_tokens=0, actual_cost=0.00778, baseline_cost=0.039, savings=0.03122, latency_seconds=3.2)
        ),
        (
            RoutingDecision(task_id="tsk_004", task_type="contract_security_audit", input_text="function withdraw()", complexity="static", model_chosen="gpt-4o", reasoning="Static mapping"),
            ExecutionResult(task_id="tsk_004", output="LGTM", model_used="gpt-4o", input_tokens=3000, output_tokens=500, cache_read_tokens=2048, cache_write_tokens=0, actual_cost=0.015, baseline_cost=0.105, savings=0.090, latency_seconds=5.1)
        ),
        (
            RoutingDecision(task_id="tsk_005", task_type="unknown_task", input_text="Design a novel consensus protocol", complexity="complex", model_chosen="o1-preview", reasoning="High-stakes analysis"),
            ExecutionResult(task_id="tsk_005", output="Here is the protocol...", model_used="o1-preview", input_tokens=5000, output_tokens=2000, cache_read_tokens=0, cache_write_tokens=0, actual_cost=0.195, baseline_cost=0.195, savings=0.0, latency_seconds=22.4)
        )
    ]
    
    for d, r in mock_data:
        dash.log(d, r)
        time.sleep(0.1) # just for dramatic effect if it were live
        
    dash.print_final_report()
