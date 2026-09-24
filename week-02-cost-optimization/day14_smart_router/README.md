# Smart Router — LLM Cost Optimization

Routes AI tasks to the cheapest capable model, applies prompt caching, 
and validates every routing decision against eval sets — cutting API 
costs by ~36% on a realistic mixed workload with zero quality loss.

## Result
```text
╭────────────────────── Smart Router — Live Cost Summary ──────────────────────╮
│ Tasks processed: 5                                                           │
│ Actual cost: $0.21780                                                        │
│ Baseline cost (always o1-preview, no cache): $0.34147                        │
│ Savings: $0.12367 (36.2%)                                                    │
│ Model usage: gpt-4o-mini: 2  |  gpt-4o: 2  |  o1-preview: 1                  │
╰──────────────────────────────────────────────────────────────────────────────╯
                                  Last 8 Tasks                                  
┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃         ┃             ┃ Tokens       ┃            ┃           ┃ Saved vs     ┃
┃ Task ID ┃ Model       ┃ (in/out)     ┃ Cache      ┃ Cost      ┃ Baseline     ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ tsk_001 │ gpt-4o-mini │ 40/10        │ -          │ $0.000012 │ $0.001188    │
│ tsk_002 │ gpt-4o-mini │ 45/10        │ -          │ $0.000013 │ $0.001262    │
│ tsk_003 │ gpt-4o      │ 1200/350     │ R:1024 W:0 │ $0.007780 │ $0.031220    │
│ tsk_004 │ gpt-4o      │ 3000/500     │ R:2048 W:0 │ $0.015000 │ $0.090000    │
│ tsk_005 │ o1-preview  │ 5000/2000    │ -          │ $0.195000 │ $0.000000    │
└─────────┴─────────────┴──────────────┴────────────┴───────────┴──────────────┘
```

## How it works
1. Classifies incoming tasks (static map or dynamic complexity check)
2. Routes to gpt-4o-mini/gpt-4o/o1-preview based on task difficulty
3. Applies prompt caching for repeated system prompts
4. Validates routing decisions against golden eval sets
5. Logs real cost vs. a "naive/always-use-the-best-model" baseline

## Stack
Python, OpenAI API, Rich

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env  # add your API key
python run_demo.py
```
