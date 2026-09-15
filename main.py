#!/usr/bin/env python3
"""
Day 3 — Bitcoin (BTC) Daily Return Prediction with Linear Regression & Ridge
Fetches historical BTC/USDT daily data from Binance API, extracts 20+ quantitative &
technical features, splits using chronological time series splitting, trains
LinearRegression & Ridge models, evaluates cross-validation & directional accuracy,
and generates a dark-mode 4-panel diagnostic plot.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# Load environment variables from .env automatically
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler


# ── 1. Fetch data ──────────────────────────────────────────
def get_btc_data(days: int = 500, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch historical kline/candlestick data from Binance API."""
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1d", "limit": days}
    try:
        raw = requests.get(url, params=params, timeout=15).json()
    except Exception as e:
        print(f"Error fetching data from Binance API: {e}", file=sys.stderr)
        raise

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("timestamp", inplace=True)
    return df


# Backward-compatibility alias
get_eth_data = get_btc_data


# ── 2. Feature engineering ─────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract return, moving average crossover, volatility, Bollinger, volume, candlestick, and RSI features."""
    f = df.copy()
    f["return_1d"]       = f["close"].pct_change(1)
    f["return_3d"]       = f["close"].pct_change(3)
    f["return_7d"]       = f["close"].pct_change(7)
    f["return_14d"]      = f["close"].pct_change(14)
    f["MA_7"]            = f["close"].rolling(7).mean()
    f["MA_14"]           = f["close"].rolling(14).mean()
    f["MA_21"]           = f["close"].rolling(21).mean()
    f["MA_cross_7_21"]   = (f["MA_7"]  / f["MA_21"])  - 1
    f["MA_cross_7_14"]   = (f["MA_7"]  / f["MA_14"])  - 1
    f["MA_cross_14_21"]  = (f["MA_14"] / f["MA_21"])  - 1
    f["volatility_7d"]   = f["return_1d"].rolling(7).std()
    f["volatility_14d"]  = f["return_1d"].rolling(14).std()
    f["volatility_21d"]  = f["return_1d"].rolling(21).std()
    f["vol_ratio"]       = f["volatility_7d"] / f["volatility_21d"]
    f["BB_mid"]          = f["close"].rolling(20).mean()
    f["BB_std"]          = f["close"].rolling(20).std()
    f["BB_upper"]        = f["BB_mid"] + 2 * f["BB_std"]
    f["BB_lower"]        = f["BB_mid"] - 2 * f["BB_std"]
    bb_range             = f["BB_upper"] - f["BB_lower"]
    f["BB_position"]     = (f["close"] - f["BB_lower"]) / bb_range
    f["BB_width"]        = bb_range / f["BB_mid"]
    f["volume_MA_7"]     = f["volume"].rolling(7).mean()
    f["volume_ratio"]    = f["volume"] / f["volume_MA_7"]
    f["volume_return"]   = f["volume"].pct_change(1)
    f["daily_range"]     = (f["high"] - f["low"]) / f["close"]
    f["candle_body"]     = (f["close"] - f["open"]) / f["open"]
    f["upper_wick"]      = (f["high"] - f[["close", "open"]].max(axis=1)) / f["close"]
    f["lower_wick"]      = (f[["close", "open"]].min(axis=1) - f["low"]) / f["close"]
    delta                = f["close"].diff()
    gain                 = delta.clip(lower=0)
    loss                 = (-delta).clip(lower=0)
    rs                   = gain.rolling(14).mean() / loss.rolling(14).mean()
    f["RSI"]             = 100 - (100 / (1 + rs))
    f["RSI_normalized"]  = (f["RSI"] - 50) / 50
    f["target"]          = f["return_1d"].shift(-1)
    return f


def run_prediction(days: int = 500, symbol: str = "BTCUSDT", output_img: str = "day3_results.png", show: bool = False):
    feature_cols = [
        "return_1d", "return_3d", "return_7d", "return_14d",
        "MA_cross_7_21", "MA_cross_7_14", "MA_cross_14_21",
        "volatility_7d", "volatility_14d", "volatility_21d", "vol_ratio",
        "BB_position", "BB_width",
        "volume_ratio", "volume_return",
        "daily_range", "candle_body", "upper_wick", "lower_wick",
        "RSI_normalized"
    ]

    symbol_display = symbol.replace("USDT", "") if "USDT" in symbol else symbol
    print(f"Fetching {days} days of {symbol} data...")
    btc  = get_btc_data(days=days, symbol=symbol)
    btc  = build_features(btc)
    data = btc[feature_cols + ["target"]].dropna()
    print(f"Dataset ready: {len(data)} rows, {len(feature_cols)} features")

    # ── 3. Split & scale ───────────────────────────────────────
    X, y      = data[feature_cols].values, data["target"].values
    split_idx = int(len(X) * 0.80)
    X_train, X_test   = X[:split_idx], X[split_idx:]
    y_train, y_test   = y[:split_idx], y[split_idx:]
    dates_train = data.index[:split_idx]
    dates_test  = data.index[split_idx:]

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 4. Train models ────────────────────────────────────────
    lr    = LinearRegression().fit(X_train_scaled, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

    y_pred_lr    = lr.predict(X_test_scaled)
    y_pred_ridge = ridge.predict(X_test_scaled)

    r2_lr    = r2_score(y_test, y_pred_lr)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    r2_train = r2_score(y_train, lr.predict(X_train_scaled))

    # ── 5. Cross-validation ────────────────────────────────────
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(Ridge(alpha=1.0), X_train_scaled,
                                 y_train, cv=tscv, scoring="r2")

    # ── 6. Direction accuracy ──────────────────────────────────
    dir_acc    = (np.sign(y_test) == np.sign(y_pred_ridge)).mean() * 100
    baseline   = (np.sign(y_test) == 1).mean() * 100

    # ── 7. Feature importance ──────────────────────────────────
    coef_df = pd.DataFrame({
        "Feature": feature_cols, "Coefficient": lr.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)

    # ── 8. Print summary ───────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  LINEAR REGRESSION RESULTS ({symbol_display})")
    print(f"{'='*45}")
    print(f"  Train R²:          {r2_train:.4f}")
    print(f"  Test R² (Linear):  {r2_lr:.4f}")
    print(f"  Test R² (Ridge):   {r2_ridge:.4f}")
    print(f"  Overfitting gap:   {r2_train - r2_lr:.4f}")
    print(f"  CV Mean R²:        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Direction accuracy:{dir_acc:.1f}%  (baseline: {baseline:.1f}%)")
    print(f"{'='*45}")

    # ── 9. Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Day 3 — {symbol_display} Return Prediction with Linear Regression",
                 color="white", fontsize=14, fontweight="bold")

    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.15, color="white")

    ax1, ax2, ax3, ax4 = axes.flat

    # Actual vs predicted over time
    ax1.plot(dates_test, y_test * 100,      color="#F7931A", lw=1.5, label="Actual")
    ax1.plot(dates_test, y_pred_ridge * 100, color="#00BCD4", lw=1.5, label="Predicted (Ridge)", ls="--")
    ax1.axhline(0, color="white", lw=0.5, alpha=0.5)
    ax1.set_title("Actual vs Predicted Daily Returns", color="white")
    ax1.set_ylabel("Return (%)", color="white")
    ax1.legend(facecolor="#2d2d44", labelcolor="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # Scatter
    lim = max(abs(y_test * 100).max(), abs(y_pred_ridge * 100).max())
    ax2.scatter(y_test * 100, y_pred_ridge * 100,
                color="#F7931A", alpha=0.5, s=24, edgecolors="none")
    ax2.plot([-lim, lim], [-lim, lim], color="white", lw=1, ls="--", alpha=0.6)
    ax2.set_title(f"Predicted vs Actual  (R²={r2_ridge:.4f})", color="white")
    ax2.set_xlabel("Actual Return (%)", color="white")
    ax2.set_ylabel("Predicted Return (%)", color="white")

    # Feature importance
    top = coef_df.head(12)
    ax3.barh(range(12), top["Coefficient"],
             color=["#4CAF50" if c > 0 else "#F44336" for c in top["Coefficient"]], alpha=0.8)
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(top["Feature"], color="white", fontsize=9)
    ax3.axvline(0, color="white", lw=0.5)
    ax3.set_title("Feature Coefficients  (green=bullish, red=bearish)", color="white")

    # CV scores
    fold_labels = [f"Fold {i+1}" for i in range(5)]
    ax4.bar(fold_labels, cv_scores,
            color=["#4CAF50" if s > 0 else "#F44336" for s in cv_scores], alpha=0.8, width=0.5)
    ax4.axhline(cv_scores.mean(), color="#FF9800", lw=2, ls="--",
                label=f"Mean = {cv_scores.mean():.4f}")
    ax4.set_title("Time Series Cross-Validation", color="white")
    ax4.set_ylabel("R² Score", color="white")
    ax4.legend(facecolor="#2d2d44", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_img, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"\nSaved: {output_img}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def main():
    run_prediction(days=500, symbol="BTCUSDT", output_img="day3_results.png")


if __name__ == "__main__":
    main()
