# learnin_Bot: Quantitative Crypto Analytics & AI Due Diligence

A modular suite of quantitative market tools and AI-driven due-diligence pipelines powered by Python, Plotly, Binance API, DefiLlama, and OpenAI.

---

## Project Tools

### 1. DeFi Whitepaper Risk Analyzer ([`whitepaper_analyzer.py`](whitepaper_analyzer.py))
Extracts text from crypto/DeFi whitepaper PDFs, chunks sections with token counting, analyzes each section for concrete risk signals across 6 critical categories (Tokenomics, Centralization, Security, Economic Design, Transparency, Regulatory) using OpenAI (`gpt-4o-mini`), and synthesizes an executive risk report with Rich terminal UI and Markdown/JSON export.

```bash
python3 whitepaper_analyzer.py path/to/whitepaper.pdf
```

### 2. Crypto News Sentiment Digest ([`crypto_news_sentiment.py`](crypto_news_sentiment.py))
Fetches live crypto news headlines, extracts structured sentiment and affected asset tags using OpenAI (`gpt-4o-mini`), and synthesizes a Daily Digest with overall market mood.

```bash
python3 crypto_news_sentiment.py
```

### 3. Machine Learning Return Prediction ([`main.py`](main.py))
Extracts 20+ quantitative features from Binance OHLCV data, trains `LinearRegression` & `Ridge`, tests directional accuracy, and generates a 4-panel diagnostic plot (`day3_results.png`).

```bash
python3 main.py
```

### 4. Interactive Market Dashboard ([`day2_dashboard.html`](day2_dashboard.html))
Interactive 4-panel Plotly visualization with Candlesticks, Bollinger Bands, DefiLlama TVL, Volume, and Volatility.

---

## Setup & Prerequisites

```bash
cd /Users/macbookair/.gemini/antigravity-ide/scratch/learnin_Bot
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure your `.env` contains your OpenAI API key:
```env
OPENAI_API_KEY=sk-...
```
