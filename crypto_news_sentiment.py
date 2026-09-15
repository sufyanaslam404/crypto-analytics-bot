#!/usr/bin/env python3
"""
crypto_news_sentiment.py — Crypto News Sentiment & Daily Digest Pipeline
Fetches real-time crypto news from CryptoCompare, analyzes sentiment and asset impact
using OpenAI (gpt-4o-mini), generates a daily market digest with Pydantic structured output,
and saves the report to JSON.
"""

import os
import json
import re
import time
import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from typing import Literal, List
import tiktoken

load_dotenv()
client = OpenAI()
enc = tiktoken.get_encoding("cl100k_base")

# ─────────────────────────────────────────────────────────
# STEP 1: Fetch REAL news (grounding)
# ─────────────────────────────────────────────────────────
def fetch_crypto_news(limit=8):
    """
    Fetch crypto news headlines. Uses CryptoCompare with automatic fallback
    to CoinTelegraph's RSS feed (no API key required).
    """
    # Try CryptoCompare API
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {"lang": "EN"}
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        articles = data.get("Data", [])
        if isinstance(articles, list) and articles:
            return [
                {
                    "title": a["title"],
                    "source": a["source"],
                    "published": datetime.fromtimestamp(a["published_on"]).strftime("%Y-%m-%d %H:%M"),
                    "url": a["url"]
                }
                for a in articles[:limit]
            ]
    except Exception:
        pass

    # Fallback to CoinTelegraph public RSS feed
    try:
        import xml.etree.ElementTree as ET
        rss_url = "https://cointelegraph.com/rss"
        res = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        root = ET.fromstring(res.content)
        items = root.findall(".//item")[:limit]
        return [
            {
                "title": (item.find("title").text if item.find("title") is not None else "").strip(),
                "source": "CoinTelegraph",
                "published": (item.find("pubDate").text if item.find("pubDate") is not None else "")[:25],
                "url": (item.find("link").text if item.find("link") is not None else "").strip()
            }
            for item in items
        ]
    except Exception as e:
        print(f"Warning: could not fetch news: {e}")
        return []

# ─────────────────────────────────────────────────────────
# STEP 2: Structured output schema
# ─────────────────────────────────────────────────────────
class NewsSentiment(BaseModel):
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0, le=1)
    affected_asset: str
    reasoning: str

class DailyDigest(BaseModel):
    date: str
    overall_market_mood: Literal["bullish", "bearish", "neutral", "mixed"]
    summary: str
    key_headlines: List[NewsSentiment]

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

# ─────────────────────────────────────────────────────────
# STEP 3: Analyze each headline (OpenAI API, temp=0)
# ─────────────────────────────────────────────────────────
def analyze_headline(headline_text, retries=3):
    system_prompt = """You are a precise crypto market news analyst.
Analyze ONLY the headline given to you. Do not assume information not present in the headline.
Respond with ONLY valid JSON, no markdown fences, no extra text:
{"sentiment": "bullish|bearish|neutral", "confidence": 0.0-1.0, "affected_asset": "TICKER or GENERAL", "reasoning": "one concise sentence"}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=150,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": headline_text}
                ]
            )
            raw = extract_json(response.choices[0].message.content)
            data = json.loads(raw)
            return NewsSentiment(**data)
        except RateLimitError:
            time.sleep(2 ** attempt)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: failed to parse response for '{headline_text[:40]}...': {e}")
            return None
        except Exception as e:
            print(f"  Error on headline analysis: {e}")
            return None
    return None

# ─────────────────────────────────────────────────────────
# STEP 4: Synthesize the full daily digest (OpenAI API)
# ─────────────────────────────────────────────────────────
def generate_daily_digest(news_items, sentiments):
    context_lines = []
    for item, sent in zip(news_items, sentiments):
        if sent:
            context_lines.append(
                f"- [{sent.sentiment.upper()}, {sent.confidence:.0%} conf] "
                f"{item['title']} (source: {item['source']})"
            )
    
    context = "\n".join(context_lines)
    
    system_prompt = """You are a senior crypto market analyst preparing a daily brief for a trader.
Base your summary STRICTLY on the headline analyses provided. Do not invent facts, prices, or events not mentioned.
Respond with ONLY valid JSON:
{"overall_market_mood": "bullish|bearish|neutral|mixed", "summary": "2-3 sentence market overview"}"""

    user_message = f"Today's analyzed headlines:\n\n{context}\n\nSynthesize the overall market mood and a brief summary."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    raw = extract_json(response.choices[0].message.content)
    return json.loads(raw)

# ─────────────────────────────────────────────────────────
# STEP 5: Run the whole pipeline
# ─────────────────────────────────────────────────────────
def run_daily_digest():
    print("Fetching crypto news...")
    news_items = fetch_crypto_news(limit=8)
    print(f"Fetched {len(news_items)} headlines.\n")

    print("Analyzing each headline with OpenAI (gpt-4o-mini)...")
    sentiments = []
    total_tokens_used = 0
    
    for item in news_items:
        result = analyze_headline(item["title"])
        sentiments.append(result)
        total_tokens_used += len(enc.encode(item["title"])) + 150
        status = f"{result.sentiment.upper()} ({result.confidence:.0%})" if result else "FAILED"
        print(f"  [{status}] {item['title'][:60]}")

    print("\nSynthesizing daily digest...")
    digest_data = generate_daily_digest(news_items, sentiments)

    digest = DailyDigest(
        date=datetime.now().strftime("%Y-%m-%d"),
        overall_market_mood=digest_data["overall_market_mood"],
        summary=digest_data["summary"],
        key_headlines=[s for s in sentiments if s is not None]
    )

    # ── Print final report ────────────────────────────────
    print("\n" + "="*60)
    print(f"  CRYPTO DAILY DIGEST — {digest.date}")
    print(f"  Overall Mood: {digest.overall_market_mood.upper()}")
    print("="*60)
    print(f"\n{digest.summary}\n")
    print("-"*60)
    print("Individual headline analysis:")
    for h in digest.key_headlines:
        print(f"  [{h.sentiment.upper():8}] {h.affected_asset:6} ({h.confidence:.0%}) — {h.reasoning}")
    
    print(f"\nEstimated tokens used this run: ~{total_tokens_used}")
    print("="*60)

    return digest

if __name__ == "__main__":
    digest = run_daily_digest()
    
    output_file = f"digest_{digest.date}.json"
    with open(output_file, "w") as f:
        f.write(digest.model_dump_json(indent=2))
    print(f"\nSaved to {output_file}")
