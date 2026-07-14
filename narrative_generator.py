"""Generate a readable market overview with DeepSeek's OpenAI-compatible API."""

from __future__ import annotations

import os
import re

from openai import OpenAI

import config


SYSTEM_PROMPT = """You are a careful crypto market analyst writing educational market commentary for Telegram.

Write a richer but still concise overview using 7 or 8 short lines. Every line must start directly with one relevant emoji. Do not use hyphens, bullet characters, numbered lists, headings, markdown, or extra emoji decoration. Do not insert blank lines; the application adds spacing later.

Use this order:
📈, 📉, or ➡️ overall trend, with the exact technical reason
💰 confirmed closing price and timeframe
⚡ RSI and MACD momentum, including whether MACD is above or below zero when relevant
📊 ATR and Bollinger volatility, using their percentages and historical percentiles
🎯 support and resistance, noting repeated pivot touches when useful
👀 one conditional sentence explaining what a confirmed close beyond either level would technically change
Fear & Greed on its own line when supplied, using 😱 Extreme Fear, 😨 Fear, 😐 Neutral, 🙂 Greed, or 🤑 Extreme Greed
🕒 confirmed candle close time in UTC and Binance as the price source

Rules:
Use only facts and numbers in the supplied context. Never calculate or invent missing values.
Distinguish trend from short term momentum when they disagree.
Never tell readers to buy, sell, enter, exit, hold, or set stop loss or take profit levels.
Do not predict a future price or promise an outcome.
Write clear natural English for a general audience.
Keep the response below 850 characters.
"""

COMPARISON_PROMPT = """You write concise educational crypto commentary for a Telegram chart album containing two assets.

Write 7 or 8 short lines. Every line starts directly with one relevant emoji. Do not use hyphens, bullet characters, numbered lists, headings, markdown, or blank lines.

Give each asset one trend line and one momentum line. Then compare their volatility, state both confirmed closes, summarize both support/resistance zones, add one conditional line about confirmed level breaks, include Fear & Greed when supplied, and finish with the candle close time and data sources.

Use only supplied facts. Distinguish trend from momentum. Never advise buying, selling, holding, entries, exits, stop losses, or targets. Do not predict prices. Keep the response below 850 characters.
"""

MARKET_MAP_PROMPT = """You write a concise daily crypto market map for Telegram.

Write 6 or 7 short lines. Every line starts directly with one relevant emoji. Do not use hyphens, bullet characters, numbered lists, headings, markdown, or blank lines.

Cover overall breadth, strongest and weakest assets, BTC and ETH, market leadership, Fear & Greed when supplied, and finish with the source. Use only supplied facts. Do not give trade instructions or predict prices. Keep the response below 850 characters.
"""

MACRO_PROMPT = """You write concise educational macro context for a crypto audience on Telegram.

Write 6 or 7 short lines. Every line starts directly with one relevant emoji. Do not use hyphens, bullet characters, numbered lists, headings, markdown, or blank lines.

Cover the S&P 500, the Federal Reserve broad U.S. dollar index, BTC dominance, and the relationship between risk assets, dollar strength, and crypto without claiming causation. Clearly call the dollar series the Fed broad dollar index, not DXY. Use only supplied facts. Never give trade instructions or predict prices. Finish with data-source dates. Keep the response below 850 characters.
"""


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["DEEPSEEK_KEY"],
        base_url=config.DEEPSEEK_BASE_URL,
    )


def _format_for_telegram(text: str) -> str:
    """Remove accidental list markers and place one blank line between points."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-•*]|\d+[.)])\s*", "", line)
        lines.append(line)
    return "\n\n".join(lines)


def _fit_caption(body: str) -> str:
    """Keep complete lines and reserve room for the mandatory disclaimer."""
    limit = config.TELEGRAM_CAPTION_LIMIT - len(config.DISCLAIMER)
    if len(body) <= limit:
        return body + config.DISCLAIMER

    kept = []
    for line in body.split("\n\n"):
        candidate = "\n\n".join([*kept, line])
        if len(candidate) > limit:
            break
        kept.append(line)
    return "\n\n".join(kept) + config.DISCLAIMER


def _complete(system_prompt: str, context: str, headline: str, fallback: str) -> str:
    """Always return a useful titled caption, even if the model returns no text."""
    body = ""
    try:
        response = _client().chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_tokens=config.DEEPSEEK_MAX_TOKENS,
        )
        content = response.choices[0].message.content or ""
        body = _format_for_telegram(content.strip())
    except Exception as exc:
        print(f"DeepSeek narrative unavailable; using factual fallback: {exc}")

    if len(body) < 20:
        body = fallback
    return _fit_caption(f"{headline}\n\n{body}")


def _price(value: float) -> str:
    if abs(value) >= 1_000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    if abs(value) >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _sentiment_line(fear_greed: dict | None) -> str | None:
    if not fear_greed:
        return None
    value = fear_greed["value"]
    if value <= 24:
        emoji = "😱"
    elif value <= 44:
        emoji = "😨"
    elif value <= 55:
        emoji = "😐"
    elif value <= 74:
        emoji = "🙂"
    else:
        emoji = "🤑"
    return f"{emoji} Sentiment is {fear_greed['classification']} at {value}/100"


def generate_narrative(symbol: str, timeframe: str, indicator_summary: str,
                        fear_greed: dict | None) -> str:
    context_lines = [f"Symbol: {symbol}", f"Timeframe: {timeframe}", indicator_summary]
    if fear_greed:
        context_lines.append(
            f"Fear & Greed Index: {fear_greed['value']} ({fear_greed['classification']})"
        )

    ticker = symbol.removesuffix("USDT")
    fallback = (
        f"📊 {ticker} has a newly confirmed {timeframe.upper()} market update\n\n"
        f"👀 The chart shows trend, momentum, volatility and key pivot zones\n\n"
        "🕒 All values use the latest confirmed candle"
    )
    return _complete(
        SYSTEM_PROMPT,
        "\n".join(context_lines),
        f"🔎 {ticker} MARKET STRUCTURE: WHAT CHANGED?",
        fallback,
    )


def generate_comparison(analyses: list[dict], fear_greed: dict | None) -> str:
    sections = []
    for analysis in analyses:
        sections.append(
            f"Asset: {analysis['ticker']}\nTimeframe: {analysis['timeframe']}\n{analysis['summary']}"
        )
    if fear_greed:
        sections.append(
            f"Fear & Greed Index: {fear_greed['value']} ({fear_greed['classification']})"
        )
    first, second = analyses
    sentiment = _sentiment_line(fear_greed)
    fallback_lines = [
        f"📊 {first['ticker']} structure is {first['trend'].split(':', 1)[0]} with RSI at {first['rsi']:.1f}",
        f"📊 {second['ticker']} structure is {second['trend'].split(':', 1)[0]} with RSI at {second['rsi']:.1f}",
        f"💰 Confirmed closes: {first['ticker']} {_price(first['price'])} · {second['ticker']} {_price(second['price'])}",
        f"🎯 {first['ticker']} support {_price(first['levels']['support'])} · resistance {_price(first['levels']['resistance'])}",
        f"🎯 {second['ticker']} support {_price(second['levels']['support'])} · resistance {_price(second['levels']['resistance'])}",
        "👀 A confirmed close beyond either highlighted zone would change the current structure",
    ]
    if sentiment:
        fallback_lines.append(sentiment)
    fallback_lines.append(f"🕒 Confirmed {first['timeframe'].upper()} candles · sources shown on charts")
    return _complete(
        COMPARISON_PROMPT,
        "\n\n".join(sections),
        f"🔥 {first['ticker']} vs {second['ticker']}: WHICH HAS THE STRONGER SETUP?",
        "\n\n".join(fallback_lines),
    )


def generate_market_map(snapshot: list[dict], fear_greed: dict | None) -> str:
    ordered = sorted(snapshot, key=lambda row: row["change_24h"], reverse=True)
    advancing = sum(row["change_24h"] > 0 for row in snapshot)
    lines = [
        f"Market breadth: {advancing} advancing and {len(snapshot) - advancing} declining",
        *[
            f"{row['ticker']}: price ${row['price']:.8g}, 24h change {row['change_24h']:+.2f}%"
            for row in ordered
        ],
    ]
    if fear_greed:
        lines.append(f"Fear & Greed Index: {fear_greed['value']} ({fear_greed['classification']})")
    lines.append("Source: CoinGecko")
    strongest = ordered[0]
    weakest = ordered[-1]
    if advancing >= 8:
        headline = "🚀 CRYPTO MARKET SURGE: BUYERS TAKE CONTROL"
    elif advancing <= 2:
        headline = "🚨 CRYPTO MARKET FLASHES RED"
    else:
        headline = "⚡ CRYPTO MARKET SPLITS: MOMENTUM IS SHIFTING"
    fallback_lines = [
        f"📊 Breadth: {advancing} {'asset' if advancing == 1 else 'assets'} advancing and {len(snapshot) - advancing} declining",
        f"🚀 Leader: {strongest['ticker']} {strongest['change_24h']:+.2f}% at {_price(strongest['price'])}",
        f"🔻 Weakest: {weakest['ticker']} {weakest['change_24h']:+.2f}% at {_price(weakest['price'])}",
    ]
    by_ticker = {row["ticker"]: row for row in snapshot}
    for ticker, emoji in (("BTC", "🟠"), ("ETH", "🔵")):
        row = by_ticker.get(ticker)
        if row:
            fallback_lines.append(
                f"{emoji} {ticker}: {row['change_24h']:+.2f}% at {_price(row['price'])}"
            )
    sentiment = _sentiment_line(fear_greed)
    if sentiment:
        fallback_lines.append(sentiment)
    fallback_lines.append("🕒 24-hour market snapshot · source: CoinGecko")
    return _complete(
        MARKET_MAP_PROMPT,
        "\n".join(lines),
        headline,
        "\n\n".join(fallback_lines),
    )


def _change(series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


def generate_macro(macro: dict) -> str:
    sp500 = macro["sp500"]
    dollar = macro["dollar"]
    context = (
        f"S&P 500 latest close: {sp500.iloc[-1]:.2f}\n"
        f"S&P 500 1-session change: {_change(sp500, 1):+.2f}%\n"
        f"S&P 500 5-session change: {_change(sp500, 5):+.2f}%\n"
        f"S&P 500 observation date: {sp500.index[-1].date()}\n"
        f"Fed broad U.S. dollar index latest: {dollar.iloc[-1]:.2f}\n"
        f"Dollar index 1-session change: {_change(dollar, 1):+.2f}%\n"
        f"Dollar index 5-session change: {_change(dollar, 5):+.2f}%\n"
        f"Dollar observation date: {dollar.index[-1].date()}\n"
        f"Current BTC dominance: {macro['btc_dominance']:.2f}%\n"
        "Sources: FRED for S&P 500 and broad dollar index; CoinGecko for BTC dominance"
    )
    sp_1d = _change(sp500, 1)
    sp_5d = _change(sp500, 5)
    dollar_1d = _change(dollar, 1)
    dollar_5d = _change(dollar, 5)
    if sp_5d > 0 and dollar_5d < 0:
        headline = "🌍 RISK-ON MOMENTUM BUILDS"
    elif sp_5d < 0 and dollar_5d > 0:
        headline = "⚠️ DOLLAR PRESSURE HITS RISK ASSETS"
    else:
        headline = "🔍 WALL STREET AND THE DOLLAR DIVERGE"
    fallback = "\n\n".join([
        f"📈 S&P 500 closed at {sp500.iloc[-1]:,.2f} · 1D {sp_1d:+.2f}% · 5D {sp_5d:+.2f}%",
        f"💵 Fed broad dollar index is {dollar.iloc[-1]:,.2f} · 1D {dollar_1d:+.2f}% · 5D {dollar_5d:+.2f}%",
        f"₿ Bitcoin dominance stands at {macro['btc_dominance']:.2f}%",
        "🔄 Stocks and the dollar are showing the current balance between risk appetite and defensiveness",
        "👀 Dollar strength can pressure global liquidity, while a softer dollar can ease that pressure",
        f"🕒 S&P data: {sp500.index[-1].date()} · dollar data: {dollar.index[-1].date()} · FRED + CoinGecko",
    ])
    return _complete(MACRO_PROMPT, context, headline, fallback)


def generate_followup(symbol: str, timeframe: str, message: str) -> str:
    """Create a factual template reply when a previously flagged level breaks."""
    return f"📊 Update · {symbol} ({timeframe.upper()})\n\n{message}{config.DISCLAIMER}"


def generate_event_alert(ticker: str, price: float, event: str, levels: dict,
                         close_time: str, source: str) -> str:
    """A deterministic alert avoids an AI call for background 4h scans."""
    body = (
        f"🚨 {ticker} confirmed 4H technical update\n\n"
        f"📊 {event}\n\n"
        f"💰 Confirmed close ${price:,.4g}\n\n"
        f"🎯 Support ~{levels['support']:,.4g} · resistance ~{levels['resistance']:,.4g}\n\n"
        f"🕒 Closed {close_time[:16]} UTC · {source}"
    )
    return _fit_caption(body)
