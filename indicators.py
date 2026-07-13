"""
Technical indicators, implemented directly with pandas rather than the
pandas_ta library. Two reasons:
  1. pandas_ta has a known import-breaking bug on numpy>=2.0
     (it does `from numpy import NaN`, which numpy removed).
  2. These formulas are short and standard — writing them directly means
     no hidden library behavior and one less dependency that can break
     silently on a GitHub Actions runner.
"""

import pandas as pd
import config


def compute_rsi(close: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_macd(close: pd.Series, fast=config.MACD_FAST, slow=config.MACD_SLOW,
                  signal=config.MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bbands(close: pd.Series, period=config.BB_PERIOD, std=config.BB_STD):
    sma = close.rolling(period).mean()
    rolling_std = close.rolling(period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    return upper, sma, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                 period=config.ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all indicator columns to a Binance OHLC DataFrame in place-safe manner."""
    df = df.copy()
    df["ema_fast"] = compute_ema(df["Close"], config.EMA_FAST)
    df["ema_slow"] = compute_ema(df["Close"], config.EMA_SLOW)
    df["rsi"] = compute_rsi(df["Close"])
    macd, signal, hist = compute_macd(df["Close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, signal, hist
    upper, mid, lower = compute_bbands(df["Close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = upper, mid, lower
    df["atr"] = compute_atr(df["High"], df["Low"], df["Close"])
    return df


def find_key_levels(df: pd.DataFrame, lookback=config.SWING_LOOKBACK) -> dict:
    """Recent swing high/low used as a support/resistance zone to watch."""
    recent = df.tail(lookback)
    return {
        "support": round(float(recent["Low"].min()), 2),
        "resistance": round(float(recent["High"].max()), 2),
    }


def summarize_for_prompt(df: pd.DataFrame, levels: dict) -> str:
    """Turns the last row of indicators into a compact text block for the LLM."""
    last = df.iloc[-1]
    trend = "bullish (fast EMA above slow EMA)" if last["ema_fast"] > last["ema_slow"] else "bearish (fast EMA below slow EMA)"
    macd_state = "MACD above signal (positive momentum)" if last["macd"] > last["macd_signal"] else "MACD below signal (negative momentum)"
    return (
        f"Last close: {last['Close']:.2f}\n"
        f"EMA{config.EMA_FAST}: {last['ema_fast']:.2f} | EMA{config.EMA_SLOW}: {last['ema_slow']:.2f} -> {trend}\n"
        f"RSI({config.RSI_PERIOD}): {last['rsi']:.1f}\n"
        f"{macd_state}\n"
        f"Bollinger Bands: upper {last['bb_upper']:.2f} / mid {last['bb_mid']:.2f} / lower {last['bb_lower']:.2f}\n"
        f"ATR({config.ATR_PERIOD}): {last['atr']:.2f}\n"
        f"Recent support zone: ~{levels['support']:.2f}\n"
        f"Recent resistance zone: ~{levels['resistance']:.2f}"
    )
