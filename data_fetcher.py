"""Pull crypto and macro market data from free public endpoints."""

from __future__ import annotations

from io import StringIO
import math
import os
import requests
import pandas as pd

import config

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
# Exchange-hosted derivatives APIs (Binance Futures, Bybit, ...) block traffic from
# cloud/datacenter IP ranges, including GitHub Actions' hosted runners — confirmed by
# both returning 451/403 in production. CoinGecko's /derivatives endpoint aggregates
# funding rate and open interest across exchanges and, being CoinGecko, isn't blocked.


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Returns only confirmed closed candles. Binance includes the currently
    forming candle in its most-recent results, so publishing the final row
    without checking its close time would make indicators change after a post.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    return _parse_closed_klines(resp.json())


def fetch_asset_klines(asset: dict, interval: str, limit: int) -> pd.DataFrame:
    """Prefer Binance and fall back to CoinGecko for unsupported pairs."""
    try:
        return fetch_klines(asset["symbol"], interval, limit)
    except requests.RequestException:
        return fetch_coingecko_ohlc(asset["coingecko_id"], interval, limit)


def _coingecko_headers() -> dict:
    key = os.environ.get("COINGECKO_API_KEY")
    return {"x-cg-demo-api-key": key} if key else {}


def _coingecko_get(path: str, params: dict | None = None) -> requests.Response:
    response = requests.get(
        f"{config.COINGECKO_BASE_URL}{path}",
        params=params,
        headers=_coingecko_headers(),
        timeout=20,
    )
    response.raise_for_status()
    return response


def fetch_coingecko_ohlc(coin_id: str, interval: str, limit: int) -> pd.DataFrame:
    """Build confirmed candles from free hourly CoinGecko price history.

    CoinGecko's free OHLC endpoint becomes four-day candles beyond 30 days.
    Market-chart range requests remain hourly for windows up to 90 days, so we
    fetch bounded windows and resample them locally into honest 4h/1d OHLC.
    """
    hours_per_candle = {"4h": 4, "1d": 24}.get(interval)
    if not hours_per_candle:
        raise ValueError(f"CoinGecko fallback does not support {interval}")

    end = pd.Timestamp.now(tz="UTC")
    history_days = math.ceil(limit * hours_per_candle / 24) + 2
    start = end - pd.Timedelta(days=history_days)
    cursor = start
    prices = []
    while cursor < end:
        window_end = min(cursor + pd.Timedelta(days=89), end)
        payload = _coingecko_get(
            f"/coins/{coin_id}/market_chart/range",
            {
                "vs_currency": "usd",
                "from": int(cursor.timestamp()),
                "to": int(window_end.timestamp()),
                "precision": "full",
            },
        ).json()
        prices.extend(payload.get("prices", []))
        cursor = window_end

    if not prices:
        raise ValueError(f"CoinGecko returned no price history for {coin_id}")
    result = _aggregate_coingecko_prices(prices, interval).tail(limit).copy()
    result.attrs["source"] = "CoinGecko"
    return result


def _aggregate_coingecko_prices(raw_prices: list, interval: str,
                                now: pd.Timestamp | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(raw_prices, columns=["timestamp", "Price"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    rule = "4h" if interval == "4h" else "1D"
    candles = frame["Price"].resample(rule, origin="epoch").ohlc().dropna()
    candles.columns = ["Open", "High", "Low", "Close"]
    candles["Volume"] = 0.0
    duration = pd.Timedelta(hours=4) if interval == "4h" else pd.Timedelta(days=1)
    candles["CloseTime"] = candles.index + duration - pd.Timedelta(milliseconds=1)
    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
    candles = candles.loc[candles["CloseTime"] <= now].copy()
    if candles.empty:
        raise ValueError(f"CoinGecko returned no confirmed {interval} candles")
    candles.attrs["source"] = "CoinGecko"
    return candles


def fetch_market_snapshot(assets: list[dict]) -> list[dict]:
    """Fetch all ten assets, with both 24h and 7d change, in one CoinGecko request."""
    ids = ",".join(asset["coingecko_id"] for asset in assets)
    rows = _coingecko_get(
        "/coins/markets",
        {
            "vs_currency": "usd",
            "ids": ids,
            "price_change_percentage": "24h,7d",
            "sparkline": "false",
        },
    ).json()
    by_id = {row["id"]: row for row in rows}
    snapshot = []
    for asset in assets:
        row = by_id.get(asset["coingecko_id"])
        if not row:
            continue
        snapshot.append({
            **asset,
            "price": float(row["current_price"]),
            "change_24h": float(row.get("price_change_percentage_24h_in_currency") or 0),
            "change_7d": float(row.get("price_change_percentage_7d_in_currency") or 0),
            "market_cap": float(row.get("market_cap") or 0),
            "volume_24h": float(row.get("total_volume") or 0),
        })
    if len(snapshot) != len(assets):
        found = {row["ticker"] for row in snapshot}
        missing = ", ".join(asset["ticker"] for asset in assets if asset["ticker"] not in found)
        raise ValueError(f"CoinGecko market snapshot is missing: {missing}")
    return snapshot


def fetch_global_snapshot() -> dict:
    """Total crypto market cap and BTC dominance in one CoinGecko request."""
    data = _coingecko_get("/global").json()["data"]
    return {
        "btc_dominance": float(data["market_cap_percentage"]["btc"]),
        "total_market_cap_usd": float(data["total_market_cap"]["usd"]),
    }


def fetch_stablecoin_market_cap() -> float:
    """Combined market cap of CoinGecko's 'stablecoins' category — a rough
    proxy for capital sitting on the sidelines rather than deployed."""
    rows = _coingecko_get(
        "/coins/markets",
        {"vs_currency": "usd", "category": "stablecoins", "per_page": 250, "page": 1},
    ).json()
    return float(sum(row.get("market_cap") or 0 for row in rows))


def fetch_trending_coins(limit: int = config.TRENDING_COINS_LIMIT) -> list[dict]:
    payload = _coingecko_get("/search/trending").json()
    trending = []
    for entry in payload.get("coins", [])[:limit]:
        item = entry.get("item", {})
        data = item.get("data", {})
        change = data.get("price_change_percentage_24h", {}).get("usd")
        trending.append({
            "name": item.get("name", "?"),
            "symbol": str(item.get("symbol", "?")).upper(),
            "market_cap_rank": item.get("market_cap_rank"),
            "change_24h": float(change) if change is not None else None,
        })
    return trending


def fetch_derivatives_snapshot(tickers: list[str]) -> list[dict]:
    """Funding rate and open interest per ticker, aggregated by CoinGecko across
    exchanges. For each ticker, picks the single highest-volume perpetual market
    as the representative reading."""
    rows = _coingecko_get("/derivatives").json()
    wanted = set(tickers)
    best: dict[str, dict] = {}
    for row in rows:
        index_id = row.get("index_id")
        funding = row.get("funding_rate")
        if index_id not in wanted or row.get("contract_type") != "perpetual" or funding is None:
            continue
        volume = float(row.get("volume_24h") or 0)
        if index_id not in best or volume > best[index_id]["_volume"]:
            best[index_id] = {
                "ticker": index_id,
                "funding_rate": float(funding),
                "open_interest": float(row.get("open_interest") or 0),
                "market": row.get("market", "?"),
                "_volume": volume,
            }
    result = []
    for ticker in tickers:
        row = best.get(ticker)
        if row:
            row.pop("_volume")
            result.append(row)
    return result


def fetch_fred_series(series_id: str, lookback: int = config.MACRO_LOOKBACK) -> pd.Series:
    response = requests.get(
        config.FRED_CSV_URL,
        params={"id": series_id},
        timeout=20,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    date_column = frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column])
    values = pd.to_numeric(frame[series_id], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame[date_column], name=series_id).dropna()
    if series.empty:
        raise ValueError(f"FRED returned no values for {series_id}")
    return series.tail(lookback)


def fetch_macro_snapshot() -> dict:
    return {
        "sp500": fetch_fred_series("SP500"),
        "dollar": fetch_fred_series("DTWEXBGS"),
        "btc_dominance": fetch_global_snapshot()["btc_dominance"],
    }


def _parse_closed_klines(raw: list, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Parse Binance kline rows and discard any candle still in progress."""
    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    else:
        now = pd.Timestamp(now)
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)

    rows = []
    for k in raw:
        rows.append({
            "timestamp": pd.to_datetime(k[0], unit="ms", utc=True).tz_localize(None),
            "CloseTime": pd.to_datetime(k[6], unit="ms", utc=True).tz_localize(None),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })

    if not rows:
        raise ValueError("Binance returned no candles")

    df = pd.DataFrame(rows).set_index("timestamp")
    closed = df.loc[df["CloseTime"] <= now].copy()
    if closed.empty:
        raise ValueError("Binance returned no confirmed closed candles")
    closed.attrs["source"] = "Binance"
    return closed


def _parse_fear_greed_payload(payload: dict) -> pd.Series:
    """Oldest-to-newest Fear & Greed values, indexed by date."""
    rows = payload["data"]
    index = [pd.to_datetime(int(row["timestamp"]), unit="s").normalize() for row in rows]
    values = [int(row["value"]) for row in rows]
    return pd.Series(values, index=index, name="fear_greed").sort_index()


def fetch_fear_greed_index() -> dict | None:
    """
    Returns {"value": int, "classification": str} or None if the request fails.
    Kept optional and non-fatal — the pipeline should still post without it.
    """
    try:
        resp = requests.get(config.FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return {
            "value": int(data["value"]),
            "classification": data["value_classification"],
        }
    except Exception:
        return None


def fetch_fear_greed_history(limit: int) -> pd.Series:
    resp = requests.get(config.FEAR_GREED_URL, params={"limit": limit}, timeout=10)
    resp.raise_for_status()
    return _parse_fear_greed_payload(resp.json())
