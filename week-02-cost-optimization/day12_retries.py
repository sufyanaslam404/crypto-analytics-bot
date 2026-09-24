# rate_limited_executor.py
"""
RateLimitedExecutor:
Runs jobs with bounded concurrency AND a minimum spacing between 
request starts, to stay comfortably under RPM (Requests Per Minute) limits 
even under load.
"""

import concurrent.futures
import threading
import time
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class RateLimitedExecutor:
    """
    Runs jobs with bounded concurrency AND a minimum spacing between 
    request starts.
    """
    def __init__(self, max_concurrent: int = 5, min_interval: float = 0.1):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.min_interval = min_interval
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def _throttled_call(self, func, *args, **kwargs):
        with self.semaphore:
            # Enforce minimum spacing between call STARTS across all threads
            with self.lock:
                elapsed = time.time() - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()
            
            return func(*args, **kwargs)
    
    def run_all(self, func, items: list, max_workers: int = None) -> tuple[list, list]:
        results = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or 10) as executor:
            futures = {
                executor.submit(self._throttled_call, func, item): item 
                for item in items
            }
            
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append({"item": item, "error": str(e)})
        
        return results, errors


# ── Demo / Usage ──────────────────────────────────────────

def call_with_backoff(messages: list, system: str, retries: int = 3):
    """Simple wrapper to simulate a backoff caller for OpenAI API."""
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
                raise  # Don't retry if account has no credits
            time.sleep(2 ** attempt)
        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == retries - 1:
                raise

def process_headline(headline: str) -> dict:
    response = call_with_backoff(
        messages=[{"role": "user", "content": headline}],
        system="Classify sentiment. Respond with ONLY valid JSON: {\"sentiment\": \"bullish|bearish|neutral\"}"
    )
    # response.choices[0].message.content for OpenAI
    content = response.choices[0].message.content.strip()
    
    # Optional: clean up markdown if present
    import re
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    
    return {"headline": headline, "sentiment": content}


if __name__ == "__main__":
    headlines = [
        "Bitcoin ETF sees record inflows",
        "Major exchange reports security incident and freezes withdrawals",
        "Federal reserve leaves interest rates unchanged",
        "Ethereum layer-2 network TVL hits all-time high",
        "SEC delays decision on new spot crypto ETF",
        "Retail trading volume drops to yearly low",
        "New institutional fund announces $500M crypto allocation"
    ]
    
    print(f"Starting RateLimitedExecutor for {len(headlines)} headlines...")
    print(f"Max Concurrent: 3 | Min Interval: 1.0s\n")
    
    executor = RateLimitedExecutor(max_concurrent=3, min_interval=1.0)
    
    start_time = time.time()
    results, errors = executor.run_all(process_headline, headlines, max_workers=3)
    duration = time.time() - start_time
    
    print(f"\nProcessed: {len(results)} succeeded, {len(errors)} failed in {duration:.2f} seconds")
    
    if results:
        print("\nResults:")
        for r in results[:3]:
            print(f"  [SUCCESS] {r['headline'][:40]}... -> {r['sentiment']}")
            
    if errors:
        print("\nErrors:")
        for e in errors[:3]:
            print(f"  [FAILED] {e['item'][:40]}... -> {e['error']}")
