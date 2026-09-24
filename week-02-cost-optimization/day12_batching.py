# batch_processor.py
"""
Batch Processor
Provides a unified entry point to process large datasets, allowing you to choose 
between 'realtime' (fast, full price) or 'batch' (slower, half price async via OpenAI Batch API) 
depending on the urgency of the workload.
"""

import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

@dataclass
class ProcessingResult:
    total_items: int
    succeeded: int
    failed: int
    total_cost: float
    elapsed_seconds: float
    errors: list


def process_large_batch(
    items: list,
    process_fn,
    mode: str = "realtime",   # "realtime" or "batch"
    max_concurrent: int = 8,
) -> ProcessingResult:
    """
    Unified entry point: choose realtime (fast, full price) or batch 
    (slower, half price) based on your actual urgency needs.
    """
    start = time.time()
    
    if mode == "realtime":
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(process_fn, item): item for item in items}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append({"item": futures[future], "error": str(e)})
        
        elapsed = time.time() - start
        return ProcessingResult(
            total_items=len(items),
            succeeded=len(results),
            failed=len(errors),
            total_cost=0.0,   # Integrate with cost_tracker.py for real estimates
            elapsed_seconds=elapsed,
            errors=errors
        )
    
    elif mode == "batch":
        # For genuinely large volumes — submit and let it run async
        print(f"Submitting {len(items)} items as a batch job...")
        print("This is async — check back later rather than waiting here.")
        # Full batch submission code would go here (e.g. OpenAI Batch API)
        raise NotImplementedError("OpenAI Batch API logic needs to be wired up here.")
    
    else:
        raise ValueError("mode must be 'realtime' or 'batch'")


# ── Demo / Usage ──────────────────────────────────────────

def call_with_backoff(messages: list, system: str, retries: int = 3):
    import openai
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=50,
                messages=[{"role": "system", "content": system}] + messages
            )
            return response
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e) or "credit_balance_exhausted" in str(e):
                raise  # Abort immediately if out of credits
            time.sleep(2 ** attempt)
        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == retries - 1:
                raise

if __name__ == "__main__":
    headlines = [
        "Bitcoin ETF sees record inflows",
        "Major exchange reports security incident and freezes withdrawals",
        "Federal reserve leaves interest rates unchanged",
    ]
    
    print(f"Starting process_large_batch in REALTIME mode for {len(headlines)} items...")
    
    try:
        result = process_large_batch(
            items=headlines,
            process_fn=lambda h: call_with_backoff(
                messages=[{"role": "user", "content": h}],
                system="Classify sentiment as JSON: {\"sentiment\": \"bullish|bearish|neutral\"}"
            ),
            mode="realtime",
            max_concurrent=8
        )

        print(f"\nProcessed {result.total_items} items in {result.elapsed_seconds:.1f}s")
        print(f"Succeeded: {result.succeeded}, Failed: {result.failed}")
        if result.errors:
            print("\nErrors encountered:")
            for err in result.errors:
                print(f" - {err['error']}")
                
    except Exception as e:
        print(f"Execution failed: {e}")
