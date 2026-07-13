"""
Pulls raw market data. No API key needed — Binance's market data endpoints
and alternative.me's Fear & Greed index are both public.
"""

import requests
import pandas as pd

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by timestamp with columns:
    Open, High, Low, Close, Volume (capitalized to match mplfinance's expectations).
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    rows = []
    for k in raw:
        rows.append({
            "timestamp": pd.to_datetime(k[0], unit="ms"),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })

    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def fetch_fear_greed_index() -> dict | None:
    """
    Returns {"value": int, "classification": str} or None if the request fails.
    Kept optional and non-fatal — the pipeline should still post without it.
    """
    from config import FEAR_GREED_URL
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return {
            "value": int(data["value"]),
            "classification": data["value_classification"],
        }
    except Exception:
        return None
