"""
Technical indicators, implemented directly with pandas rather than the
pandas_ta library. Two reasons:
  1. pandas_ta has a known import-breaking bug on numpy>=2.0
     (it does `from numpy import NaN`, which numpy removed).
  2. These formulas are short and standard — writing them directly means
     no hidden library behavior and one less dependency that can break
     silently on a GitHub Actions runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    source = df.attrs.get("source", "Binance")
    df = df.copy()
    df.attrs["source"] = source
    df["ema_fast"] = compute_ema(df["Close"], config.EMA_FAST)
    df["ema_slow"] = compute_ema(df["Close"], config.EMA_SLOW)
    df["rsi"] = compute_rsi(df["Close"])
    macd, signal, hist = compute_macd(df["Close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, signal, hist
    upper, mid, lower = compute_bbands(df["Close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = upper, mid, lower
    df["atr"] = compute_atr(df["High"], df["Low"], df["Close"])
    df["atr_pct"] = df["atr"] / df["Close"] * 100
    df["bb_width_pct"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    return df


@dataclass
class _LevelCluster:
    values: list[float] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)

    @property
    def center(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def touches(self) -> int:
        return len(self.values)

    @property
    def last_position(self) -> int:
        return max(self.positions)


def _cluster_pivots(points: list[tuple[float, int]], tolerance: float) -> list[_LevelCluster]:
    """Merge nearby pivot prices into repeated-touch support/resistance levels."""
    clusters: list[_LevelCluster] = []
    for value, position in sorted(points):
        nearest = min(clusters, key=lambda c: abs(c.center - value), default=None)
        if nearest is not None and abs(nearest.center - value) <= tolerance:
            nearest.values.append(value)
            nearest.positions.append(position)
        else:
            clusters.append(_LevelCluster([value], [position]))
    return clusters


def _pick_level(clusters: list[_LevelCluster], current_price: float, side: str,
                sample_size: int) -> _LevelCluster | None:
    if side == "support":
        eligible = [c for c in clusters if c.center < current_price]
    else:
        eligible = [c for c in clusters if c.center > current_price]

    repeated = [c for c in eligible if c.touches >= config.MIN_LEVEL_TOUCHES]
    candidates = repeated or eligible
    if not candidates:
        return None

    def score(cluster: _LevelCluster) -> float:
        recency = cluster.last_position / max(sample_size - 1, 1)
        distance_pct = abs(cluster.center - current_price) / current_price
        return cluster.touches + 0.75 * recency - 5 * distance_pct

    return max(candidates, key=score)


def _pivot_clusters(df: pd.DataFrame, lookback: int) -> tuple[list[_LevelCluster], list[_LevelCluster], float]:
    """Detects pivot highs/lows over the given lookback and merges nearby ones
    into clusters. Shared by find_key_levels (nearby walls for the chart) and
    find_extended_target (farther levels for a signal's reward:risk target)."""
    recent = df.tail(lookback)
    if len(recent) < config.PIVOT_WINDOW * 2 + 1:
        raise ValueError("Not enough candles to calculate pivot levels")

    window = config.PIVOT_WINDOW * 2 + 1
    pivot_highs = recent["High"].eq(recent["High"].rolling(window, center=True).max())
    pivot_lows = recent["Low"].eq(recent["Low"].rolling(window, center=True).min())

    high_points = [(float(recent["High"].iloc[i]), i) for i in range(len(recent)) if pivot_highs.iloc[i]]
    low_points = [(float(recent["Low"].iloc[i]), i) for i in range(len(recent)) if pivot_lows.iloc[i]]

    current_price = float(recent["Close"].iloc[-1])
    current_atr = float(recent["atr"].iloc[-1])
    tolerance = max(current_atr * config.LEVEL_CLUSTER_ATR_MULTIPLIER, current_price * 0.003)

    return _cluster_pivots(high_points, tolerance), _cluster_pivots(low_points, tolerance), tolerance


def find_key_levels(df: pd.DataFrame, lookback=config.LEVEL_LOOKBACK) -> dict:
    """Find significant nearby levels from repeated pivots over longer history."""
    high_clusters, low_clusters, tolerance = _pivot_clusters(df, lookback)
    recent = df.tail(lookback)
    current_price = float(recent["Close"].iloc[-1])

    support = _pick_level(low_clusters, current_price, "support", len(recent))
    resistance = _pick_level(high_clusters, current_price, "resistance", len(recent))

    support_value = support.center if support else float(recent["Low"].min())
    resistance_value = resistance.center if resistance else float(recent["High"].max())

    return {
        "support": round(support_value, 2),
        "resistance": round(resistance_value, 2),
        "support_touches": support.touches if support else 1,
        "resistance_touches": resistance.touches if resistance else 1,
        "lookback": len(recent),
        "zone_width": round(tolerance, 2),
    }


def find_extended_target(df: pd.DataFrame, direction: str, entry: float, stop: float,
                          min_reward_risk: float = config.MIN_SIGNAL_RISK_REWARD,
                          lookback: int | None = None) -> float | None:
    """Searches the farthest pivot level — across as much history as is
    available, not just the near lookback used for support/resistance — that
    still clears the minimum reward:risk ratio. Farther levels are tried
    first so a signal captures as much of a legitimate move as the chart
    history actually supports, instead of settling for the nearest wall.
    Returns None if nothing in the available history clears the bar, in
    which case no signal should be opened.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    lookback = len(df) if lookback is None else min(lookback, len(df))
    high_clusters, low_clusters, _ = _pivot_clusters(df, lookback)
    clusters = high_clusters if direction == "long" else low_clusters
    # Every cluster is already a verified local swing pivot (see _pivot_clusters),
    # so a single touch is still a legitimate level — unlike find_key_levels,
    # which prefers repeated touches, this is ranked by distance because the
    # goal here is the farthest level that clears the reward:risk bar.
    eligible = [
        cluster for cluster in clusters
        if (cluster.center > entry if direction == "long" else cluster.center < entry)
    ]
    for cluster in sorted(eligible, key=lambda c: abs(c.center - entry), reverse=True):
        if abs(cluster.center - entry) / risk >= min_reward_risk:
            return round(cluster.center, 2)
    return None


def rolling_correlation(price_a: pd.Series, price_b: pd.Series, window: int) -> float | None:
    """Trailing daily-return correlation between two price series aligned by date.

    Returns None when there isn't enough overlapping history to be meaningful.
    """
    a = price_a.copy()
    a.index = pd.to_datetime(a.index).normalize()
    b = price_b.copy()
    b.index = pd.to_datetime(b.index).normalize()
    combined = pd.concat([a.pct_change(), b.pct_change()], axis=1, join="inner").dropna()
    tail = combined.tail(window)
    if len(tail) < max(5, window // 3):
        return None
    return float(tail.iloc[:, 0].corr(tail.iloc[:, 1]))


def _percentile_rank(series: pd.Series, lookback: int) -> float:
    values = series.dropna().tail(lookback)
    if values.empty:
        return float("nan")
    return float((values <= values.iloc[-1]).mean() * 100)


def trend_state(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    earlier = df.iloc[-1 - min(config.TREND_SLOPE_LOOKBACK, len(df) - 1)]
    fast_rising = last["ema_fast"] > earlier["ema_fast"]

    if last["ema_fast"] > last["ema_slow"] and last["Close"] > last["ema_slow"] and fast_rising:
        return "bullish: fast EMA above slow EMA, price above slow EMA, fast EMA rising"
    if last["ema_fast"] < last["ema_slow"] and last["Close"] < last["ema_slow"] and not fast_rising:
        return "bearish: fast EMA below slow EMA, price below slow EMA, fast EMA falling"
    return "mixed/transitioning: EMA alignment, price position, and EMA slope do not fully agree"


def _structure_score(trend: str, rsi: float) -> float:
    """How decisively a single asset's structure lines up. A clean setup is one
    where trend and momentum agree — a clean bull or a clean bear both score
    higher than a muddled, transitioning tape. Direction-agnostic on purpose:
    'stronger setup' means clearer, not necessarily more bullish."""
    regime = trend.split(":", 1)[0]
    if regime == "bullish":
        return 2.0 + max(0.0, (rsi - 50) / 10)      # bullish trend, rewarded when RSI confirms
    if regime == "bearish":
        return 2.0 + max(0.0, (50 - rsi) / 10)      # bearish trend, rewarded when RSI confirms
    return 0.0                                       # mixed/transitioning: no clear structure


def compare_setups(analyses: list[dict]) -> str:
    """A deterministic one-line verdict for a two-asset deep dive, answering the
    'which has the stronger setup?' the headline poses. Ranks each asset on how
    cleanly its trend and momentum agree (see _structure_score)."""
    scored = [(_structure_score(a["trend"], a["rsi"]), a) for a in analyses]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    (top_score, top), (next_score, _) = scored[0], scored[1]
    if top_score == 0.0:
        return "On trend and momentum, both are still in transition with no clean structure yet"
    if abs(top_score - next_score) < 0.5:
        return "On trend and momentum, both setups look comparably balanced right now"
    return f"On trend and momentum alignment, {top['ticker']} shows the cleaner structure right now"


def level_proximity(price: float, levels: dict) -> dict:
    """Distance from a confirmed close to the nearer of its two key levels, as a
    signed percentage. Used by the weekly 'what to watch' post to surface which
    assets sit closest to a decision point a break or bounce would resolve."""
    support = levels["support"]
    resistance = levels["resistance"]
    to_resistance = (resistance - price) / price * 100 if price else 0.0
    to_support = (price - support) / price * 100 if price else 0.0
    if abs(to_resistance) <= abs(to_support):
        return {"level_type": "resistance", "level": resistance, "distance_pct": to_resistance}
    return {"level_type": "support", "level": support, "distance_pct": to_support}


def summarize_for_prompt(df: pd.DataFrame, levels: dict) -> str:
    """Turns the last row of indicators into a compact text block for the LLM."""
    last = df.iloc[-1]
    trend = trend_state(df)
    signal_relation = "above" if last["macd"] > last["macd_signal"] else "below"
    zero_relation = "above" if last["macd"] > 0 else "below"
    atr_percentile = _percentile_rank(df["atr_pct"], config.VOLATILITY_LOOKBACK)
    bb_percentile = _percentile_rank(df["bb_width_pct"], config.VOLATILITY_LOOKBACK)
    close_time = last.get("CloseTime", df.index[-1])
    source = df.attrs.get("source", "Binance")
    return (
        f"Price source: {source}\n"
        f"Confirmed candle close time (UTC): {pd.Timestamp(close_time).isoformat()}\n"
        f"Last close: {last['Close']:.2f}\n"
        f"EMA{config.EMA_FAST}: {last['ema_fast']:.2f} | EMA{config.EMA_SLOW}: {last['ema_slow']:.2f}\n"
        f"Trend classification: {trend}\n"
        f"RSI({config.RSI_PERIOD}): {last['rsi']:.1f}\n"
        f"MACD: {last['macd']:.2f} | signal: {last['macd_signal']:.2f} | histogram: {last['macd_hist']:.2f}; "
        f"MACD is {signal_relation} signal and {zero_relation} zero\n"
        f"Bollinger Bands: upper {last['bb_upper']:.2f} / mid {last['bb_mid']:.2f} / lower {last['bb_lower']:.2f}; "
        f"width {last['bb_width_pct']:.2f}% (percentile {bb_percentile:.0f} of the last {config.VOLATILITY_LOOKBACK} candles)\n"
        f"ATR({config.ATR_PERIOD}): {last['atr']:.2f} or {last['atr_pct']:.2f}% of price "
        f"(percentile {atr_percentile:.0f} of the last {config.VOLATILITY_LOOKBACK} candles)\n"
        f"Support: ~{levels['support']:.2f} ({levels['support_touches']} pivot touches)\n"
        f"Resistance: ~{levels['resistance']:.2f} ({levels['resistance_touches']} pivot touches)\n"
        f"Level analysis lookback: {levels['lookback']} candles"
    )
