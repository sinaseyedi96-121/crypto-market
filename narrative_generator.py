"""Generate a readable market overview with DeepSeek's OpenAI-compatible API."""

from __future__ import annotations

import html
import os
import re

from openai import OpenAI

import config
import indicators


# Every post ships as a photo (or album) plus this caption. The chart already
# carries the exact figures, so the caption's job is to explain what the picture
# means and teach one idea — not to restate numbers the reader can already see.
_EDUCATIONAL_RULES = (
    "The chart posted alongside your text already shows every exact number. Do NOT restate raw "
    "figures — no prices, percentages, RSI readings, or index values. Instead explain, in plain "
    "language, what the current picture means and why it matters, and teach the reader one idea "
    "they can reuse. Refer to levels and momentum qualitatively (for example 'pressing against its "
    "recent ceiling' or 'momentum has cooled from stretched levels'). Use only what the context "
    "supports; never invent a fact. Never tell readers to buy, sell, enter, exit, hold, or set "
    "stops or targets, and never predict a specific price. Write clear English an intelligent "
    "beginner could follow. Keep the response under 850 characters."
)

_FORMAT_RULES = (
    "Every line starts directly with one relevant emoji. Do not use hyphens, bullet characters, "
    "numbered lists, headings, markdown, or blank lines; the application adds spacing later."
)


SYSTEM_PROMPT = f"""You are a careful crypto market educator writing for a general Telegram audience about a single asset's chart.

Write 6 to 8 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Cover, in a natural order: what the trend structure is doing and what that implies; whether momentum agrees with or diverges from the trend, and what such agreement or divergence tends to signal; whether volatility is expanding or contracting and why that matters for what may come next; where the key levels sit and what a decisive break or hold of them would actually mean; and the market's fear or greed mood when supplied. Distinguish trend from short-term momentum when they disagree.
"""

COMPARISON_PROMPT = f"""You write educational crypto commentary for a Telegram album comparing two assets, shown as two charts side by side.

Write 7 or 8 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Open with the supplied verdict line stating which asset has the cleaner setup and why. Then give each asset a plain-language read of its trend and momentum, compare which is calmer or more volatile and what that implies, explain what a decisive break of each asset's key zone would signal, include the fear or greed mood when supplied, and teach one comparative idea such as why relative strength between two assets matters. Distinguish trend from momentum.
"""

MARKET_MAP_PROMPT = f"""You write a daily crypto market map for Telegram, shown with a ranked performance chart of the tracked majors.

Write 6 or 7 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Explain the day's story rather than listing movers: is this broad strength, broad weakness, or a split and rotation, and what that breadth says about the market's character; which corner led and which lagged and what that rotation hints at; how the majors are behaving relative to the rest; and the fear or greed mood when supplied. Teach one idea about reading market breadth.
"""

MACRO_PROMPT = f"""You write educational macro context for a crypto audience on Telegram, shown with a chart of the S&P 500 and the Federal Reserve broad U.S. dollar index.

Write 6 or 7 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Explain what the current mix of stocks, the dollar, equity volatility, the yield curve, and BTC's correlations means for the risk backdrop crypto trades within — without claiming one causes another. Always call the dollar series the Fed broad dollar index, not DXY. Teach one idea about why the dollar, interest rates, or equity volatility matter to crypto.
"""

DAILY_PULSE_PROMPT = f"""You write a daily derivatives and sentiment pulse for a crypto Telegram audience, shown with a funding-rate and open-interest chart.

Write 6 or 7 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Explain what the positioning means: what positive versus negative funding says about whether longs or shorts are crowded and paying to hold; what the level of open interest implies about how much leverage sits in the system and why that can amplify moves; and what any trending coins reflect about where attention is going. Teach one idea about reading derivatives positioning.
"""

WEEKLY_DIGEST_PROMPT = f"""You write a weekly crypto digest for Telegram, shown with a four-chart album: a 7-day performance ranking, an RSI and volatility scoreboard, a BTC-dominance and Fear & Greed trend, and a total and stablecoin market-cap trend.

Write 8 or 9 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Tell the week's story: who led and who lagged and what that rotation suggests; where momentum ran hot or cold across the board; how BTC dominance and the fear or greed mood shifted and what that says about risk appetite; and whether capital moved into or out of the market and toward or away from stablecoins waiting on the sidelines. Teach one idea about reading market structure over a week.
"""

SIGNAL_SCORECARD_PROMPT = f"""You write an educational weekly review of a crypto channel's hypothetical, educational trade scenarios, shown with a track-record scorecard chart.

Write 6 or 7 short lines. {_FORMAT_RULES}

The chart already shows the exact win rate, average R multiple, and each scenario's result. Do NOT restate those numbers. Instead explain honestly what the record shows and, above all, teach: what a reward-to-risk (R) multiple actually means, why a method can be profitable even when it is wrong more often than right and vice versa, why publishing losing scenarios alongside winning ones is the honest way to learn, and that a small sample proves very little. Frame everything as hypothetical and educational, never as advice or a promise of future results. Use only supplied facts; never invent a number. Keep the response under 850 characters.
"""

WHAT_TO_WATCH_PROMPT = f"""You write a forward-looking educational 'what to watch this week' post for a crypto Telegram audience, shown with a chart of which assets sit closest to a key level.

Write 6 or 7 short lines. {_FORMAT_RULES}

{_EDUCATIONAL_RULES}

Explain why each highlighted asset is worth watching: what a decisive break or hold of the level it is approaching would actually signal, and what a stretched or washed-out momentum reading tends to precede — a pause or shakeout, not an automatic reversal. Add the broad backdrop when supplied: the direction of BTC dominance, the fear or greed mood, and whether leverage looks crowded, and what each implies for the week. Teach one idea about watching decision levels rather than chasing price. This is educational context, not a forecast.
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


def _fit_caption(body: str, disclaimer: str = config.DISCLAIMER) -> str:
    """Keep complete lines and reserve room for the mandatory disclaimer.

    The disclaimer/footer carries a raw HTML link and every caption is sent
    with Telegram's HTML parse mode, so the body must be HTML-escaped here —
    the one place every caption passes through — or a stray '&', '<', or '>'
    in generated text (e.g. "S&P 500") would break entity parsing on send.
    """
    body = html.escape(body, quote=False)
    limit = config.TELEGRAM_CAPTION_LIMIT - len(disclaimer)
    if len(body) <= limit:
        return body + disclaimer

    kept = []
    for line in body.split("\n\n"):
        candidate = "\n\n".join([*kept, line])
        if len(candidate) > limit:
            break
        kept.append(line)
    return "\n\n".join(kept) + disclaimer


def _complete(system_prompt: str, context: str, headline: str, fallback: str,
              disclaimer: str = config.DISCLAIMER) -> str:
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
    return _fit_caption(f"{headline}\n\n{body}", disclaimer)


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
    verdict = indicators.compare_setups(analyses)
    sections = [
        "Verdict (state this as the opening line of your reply, in your own words): "
        + verdict
    ]
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
        f"📊 {verdict}",
        f"🧭 {first['ticker']} is {first['trend'].split(':', 1)[0]} while {second['ticker']} is {second['trend'].split(':', 1)[0]}",
        "⚡ Watch whether momentum keeps confirming each trend or starts to diverge from it",
        "👀 A confirmed close beyond either asset's highlighted zone is what would change its structure",
        "📚 Comparing two charts side by side is really a read on relative strength: which one leads when both move",
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
        breadth_read = "a broadly bullish tape with buyers in control almost everywhere"
    elif advancing <= 2:
        headline = "🚨 CRYPTO MARKET FLASHES RED"
        breadth_read = "a broadly bearish tape with sellers pressing almost everywhere"
    else:
        headline = "⚡ CRYPTO MARKET SPLITS: MOMENTUM IS SHIFTING"
        breadth_read = "a split, rotational session rather than one clear direction"

    fallback_lines = [
        f"📊 Breadth: {advancing} {'asset' if advancing == 1 else 'assets'} advancing against "
        f"{len(snapshot) - advancing} declining, {breadth_read}",
        f"🔀 {strongest['ticker']} leads the pack while {weakest['ticker']} lags furthest behind, "
        "a rotation worth watching for whether it persists or reverses",
    ]
    by_ticker = {row["ticker"]: row for row in snapshot}
    btc_row, eth_row = by_ticker.get("BTC"), by_ticker.get("ETH")
    if btc_row and eth_row:
        if (btc_row["change_24h"] > 0) == (eth_row["change_24h"] > 0):
            fallback_lines.append(
                "🟠🔵 BTC and ETH are pointed the same way today, which tends to anchor "
                "the rest of the market to their lead"
            )
        else:
            fallback_lines.append(
                "🟠🔵 BTC and ETH are pulling in different directions today, a split at "
                "the top that often shows up as indecision further down the board"
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
    context_lines = [
        f"S&P 500 latest close: {sp500.iloc[-1]:.2f}",
        f"S&P 500 1-session change: {_change(sp500, 1):+.2f}%",
        f"S&P 500 5-session change: {_change(sp500, 5):+.2f}%",
        f"S&P 500 observation date: {sp500.index[-1].date()}",
        f"Fed broad U.S. dollar index latest: {dollar.iloc[-1]:.2f}",
        f"Dollar index 1-session change: {_change(dollar, 1):+.2f}%",
        f"Dollar index 5-session change: {_change(dollar, 5):+.2f}%",
        f"Dollar observation date: {dollar.index[-1].date()}",
        f"Current BTC dominance: {macro['btc_dominance']:.2f}%",
    ]
    if macro.get("vix") is not None:
        context_lines.append(f"VIX (equity volatility index): {macro['vix']:.2f}")
    if macro.get("yield_10y") is not None and macro.get("yield_2y") is not None:
        spread = macro["yield_10y"] - macro["yield_2y"]
        context_lines.append(
            f"10-year Treasury yield: {macro['yield_10y']:.2f}% · 2-year: {macro['yield_2y']:.2f}% "
            f"· 10Y-2Y spread: {spread:+.2f} percentage points"
        )
    if macro.get("btc_sp_corr") is not None:
        context_lines.append(
            f"BTC vs S&P 500 rolling {config.CORRELATION_WINDOW}-day return correlation: {macro['btc_sp_corr']:+.2f}"
        )
    if macro.get("btc_dollar_corr") is not None:
        context_lines.append(
            f"BTC vs Fed broad dollar index rolling {config.CORRELATION_WINDOW}-day return correlation: {macro['btc_dollar_corr']:+.2f}"
        )
    context_lines.append("Sources: FRED for S&P 500, broad dollar index, VIX, and Treasury yields; CoinGecko for BTC dominance")
    context = "\n".join(context_lines)

    sp_5d = _change(sp500, 5)
    dollar_5d = _change(dollar, 5)
    if sp_5d > 0 and dollar_5d < 0:
        headline = "🌍 RISK-ON MOMENTUM BUILDS"
        fallback_lines = [
            "🌍 Stocks are firming while the dollar eases, a classic risk-on mix that "
            "tends to loosen the backdrop crypto trades within"
        ]
    elif sp_5d < 0 and dollar_5d > 0:
        headline = "⚠️ DOLLAR PRESSURE HITS RISK ASSETS"
        fallback_lines = [
            "⚠️ Stocks are softening while the dollar firms, a defensive mix that "
            "tends to tighten conditions for risk assets like crypto"
        ]
    else:
        headline = "🔍 WALL STREET AND THE DOLLAR DIVERGE"
        fallback_lines = [
            "🔍 Stocks and the dollar aren't telling one clean story right now, a "
            "sign the macro backdrop is unsettled rather than firmly risk-on or risk-off"
        ]

    if macro.get("vix") is not None:
        if macro["vix"] >= 25:
            fallback_lines.append(
                "😬 Equity volatility is running hot, which tends to spill into crypto as correlated risk-off flows"
            )
        elif macro["vix"] <= 15:
            fallback_lines.append(
                "😌 Equity volatility is calm, a backdrop that has historically let risk assets, crypto included, grind higher with fewer scares"
            )
        else:
            fallback_lines.append(
                "😐 Equity volatility sits in a middling range, neither fueling risk appetite nor draining it"
            )
    if macro.get("yield_10y") is not None and macro.get("yield_2y") is not None:
        spread = macro["yield_10y"] - macro["yield_2y"]
        if spread < 0:
            fallback_lines.append(
                "🏦 The Treasury curve is inverted, a signal that has historically preceded tighter, more cautious risk conditions"
            )
        else:
            fallback_lines.append(
                "🏦 The Treasury curve holds its normal upward slope, consistent with a steadier growth backdrop"
            )
    if macro.get("btc_sp_corr") is not None:
        if abs(macro["btc_sp_corr"]) >= 0.5:
            fallback_lines.append(
                "🔗 BTC is trading closely with stocks right now, so equity swings are likely spilling straight into crypto"
            )
        else:
            fallback_lines.append(
                "🔗 BTC's link to stocks is looser right now, leaving more room for crypto-specific drivers to take over"
            )
    fallback_lines.append(
        "📚 Watching stocks, the dollar, and rates together — rather than any one alone — is what "
        "separates a macro-driven crypto move from one crypto is making on its own"
    )
    fallback_lines.append(f"🕒 S&P data: {sp500.index[-1].date()} · dollar data: {dollar.index[-1].date()} · FRED + CoinGecko")
    return _complete(MACRO_PROMPT, context, headline, "\n\n".join(fallback_lines))


def generate_daily_pulse(derivatives_rows: list[dict], trending: list[dict]) -> str:
    context_lines = [
        f"{row['ticker']} funding rate: {row['funding_rate']:+.4f}% (market: {row['market']})"
        for row in derivatives_rows
    ] + [
        f"{row['ticker']} open interest: ${row['open_interest']:,.0f}" for row in derivatives_rows
    ]
    if trending:
        context_lines.append(
            "Trending coins on CoinGecko right now: "
            + ", ".join(f"{item['symbol']} ({item['name']})" for item in trending)
        )
    context_lines.append("Source: CoinGecko derivatives aggregation for funding/open interest and trending coins")
    context = "\n".join(context_lines)

    avg_funding = sum(row["funding_rate"] for row in derivatives_rows) / len(derivatives_rows)
    if avg_funding > 0.01:
        headline = "🔥 LEVERAGED LONGS ARE PAYING UP"
    elif avg_funding < -0.01:
        headline = "🧊 SHORTS ARE PAYING TO STAY SHORT"
    else:
        headline = "⚖️ DERIVATIVES MARKET SITS NEAR BALANCE"

    long_leaning = sum(1 for row in derivatives_rows if row["funding_rate"] > 0)
    short_leaning = sum(1 for row in derivatives_rows if row["funding_rate"] < 0)
    total = len(derivatives_rows)
    if long_leaning == total:
        crowd_line = (
            "🐂 Every tracked market is paying longs a premium right now, a sign "
            "leveraged buyers are firmly in control of positioning"
        )
    elif short_leaning == total:
        crowd_line = (
            "🐻 Every tracked market is paying shorts a premium right now, a sign "
            "leveraged sellers are firmly in control of positioning"
        )
    else:
        crowd_line = (
            f"⚖️ {long_leaning} of {total} tracked markets lean long on funding and "
            f"{short_leaning} lean short, a split field rather than a one-sided crowd"
        )

    biggest_oi = max(derivatives_rows, key=lambda row: row["open_interest"])
    leverage_line = (
        f"🏦 {biggest_oi['ticker']} still carries the deepest open interest of the group, "
        "so it's where a sharp funding flip or price swing would ripple through the most leverage"
    )

    fallback_lines = [crowd_line, leverage_line]
    if trending:
        names = ", ".join(item["symbol"] for item in trending)
        fallback_lines.append(
            f"🔎 Fresh attention is chasing {names}, and trending names typically carry "
            "thinner, more reflexive positioning than the majors"
        )
    fallback_lines.append(
        "📚 Funding shows who is crowded and paying to hold their side; open interest "
        "shows how much leverage could unwind if that crowd gets caught wrong"
    )
    fallback_lines.append("🕒 Perpetual derivatives snapshot · CoinGecko")
    return _complete(DAILY_PULSE_PROMPT, context, headline, "\n\n".join(fallback_lines))


def generate_weekly_digest(weekly_rows: list[dict], scoreboard_rows: list[dict],
                            dominance_history, feargreed_history,
                            total_mcap_history, stablecoin_history) -> str:
    strongest = max(weekly_rows, key=lambda row: row["change_7d"])
    weakest = min(weekly_rows, key=lambda row: row["change_7d"])
    most_overbought = max(scoreboard_rows, key=lambda row: row["rsi"])
    most_oversold = min(scoreboard_rows, key=lambda row: row["rsi"])
    most_volatile = max(scoreboard_rows, key=lambda row: row["atr_pct"])
    dominance_change = float(dominance_history.iloc[-1] - dominance_history.iloc[0])
    feargreed_change = float(feargreed_history.iloc[-1] - feargreed_history.iloc[0])
    mcap_change_pct = _change(total_mcap_history, len(total_mcap_history) - 1)
    stablecoin_change_pct = _change(stablecoin_history, len(stablecoin_history) - 1)

    context = "\n".join([
        f"Strongest 7-day performer: {strongest['ticker']} {strongest['change_7d']:+.2f}%",
        f"Weakest 7-day performer: {weakest['ticker']} {weakest['change_7d']:+.2f}%",
        f"Most overbought (highest RSI): {most_overbought['ticker']} RSI {most_overbought['rsi']:.1f}",
        f"Most oversold (lowest RSI): {most_oversold['ticker']} RSI {most_oversold['rsi']:.1f}",
        f"Most volatile (highest ATR%): {most_volatile['ticker']} {most_volatile['atr_pct']:.2f}%",
        f"BTC dominance moved {dominance_change:+.2f} percentage points over the window, now {dominance_history.iloc[-1]:.2f}%",
        f"Fear & Greed moved {feargreed_change:+.0f} points over the window, now {feargreed_history.iloc[-1]:.0f}/100",
        f"Total crypto market cap changed {mcap_change_pct:+.2f}% over the window",
        f"Stablecoin market cap changed {stablecoin_change_pct:+.2f}% over the window",
        "Sources: CoinGecko for prices, dominance, and market caps; Binance for RSI/ATR; alternative.me for Fear & Greed",
    ])

    advancing = sum(row["change_7d"] > 0 for row in weekly_rows)
    if advancing >= 8:
        headline = "📅 WEEKLY DIGEST: BROAD STRENGTH ACROSS THE BOARD"
    elif advancing <= 2:
        headline = "📅 WEEKLY DIGEST: A ROUGH WEEK FOR MOST ASSETS"
    else:
        headline = "📅 WEEKLY DIGEST: A MIXED WEEK OF ROTATION"

    fallback_lines = [
        f"🚀 {strongest['ticker']} led the tracked majors this week while {weakest['ticker']} lagged "
        "furthest behind, a rotation worth watching for whether it persists or reverses",
        f"📈 {most_overbought['ticker']} sits the most stretched to the upside and "
        f"{most_oversold['ticker']} the most stretched to the downside — worth watching for a pause "
        "or a snapback rather than assuming either just keeps going",
        f"⚡ {most_volatile['ticker']} swung the widest of the group this week, a reminder that the "
        "size of a move needs a matching size of risk",
    ]
    if dominance_change > 0:
        fallback_lines.append(
            "₿ Bitcoin dominance rose over the week, a sign capital leaned toward BTC over the rest of the market"
        )
    elif dominance_change < 0:
        fallback_lines.append(
            "₿ Bitcoin dominance fell over the week, a sign capital rotated out of BTC and into the broader market"
        )
    else:
        fallback_lines.append("₿ Bitcoin dominance held steady over the week, no clear rotation either way")
    if feargreed_change > 0:
        fallback_lines.append("🧭 Sentiment drifted toward greed over the week")
    elif feargreed_change < 0:
        fallback_lines.append("🧭 Sentiment drifted toward fear over the week")
    else:
        fallback_lines.append("🧭 Sentiment held steady over the week")
    if mcap_change_pct > 0 and stablecoin_change_pct < 0:
        fallback_lines.append(
            "💧 Total market cap grew while stablecoin supply shrank, consistent with capital moving "
            "off the sidelines and into risk assets"
        )
    elif mcap_change_pct < 0 and stablecoin_change_pct > 0:
        fallback_lines.append(
            "💧 Total market cap shrank while stablecoin supply grew, consistent with capital moving "
            "to the sidelines rather than staying in risk assets"
        )
    else:
        fallback_lines.append(
            "💧 Total market cap and stablecoin supply moved together this week rather than telling a "
            "clean story of capital rotating in or out"
        )
    fallback_lines.append("🕒 Weekly digest · Binance, CoinGecko, alternative.me")
    return _complete(WEEKLY_DIGEST_PROMPT, context, headline, "\n\n".join(fallback_lines))


def generate_followup(symbol: str, timeframe: str, message: str) -> str:
    """Create a factual template reply when a previously flagged level breaks."""
    return _fit_caption(f"📊 Update · {symbol} ({timeframe.upper()})\n\n{message}")


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


def generate_signal_post(ticker: str, timeframe: str, direction: str, entry: float,
                          target: float, stop: float, trend: str, rsi: float) -> str:
    """Deterministic hypothetical long/short scenario post (no AI call, so the
    numbers can never drift from what was actually computed)."""
    arrow = "🟢 LONG" if direction == "long" else "🔴 SHORT"
    risk = abs(entry - stop)
    reward = abs(target - entry)
    reward_risk = reward / risk if risk else 0.0
    body = (
        f"{arrow} scenario\n\n"
        f"📈 Trend basis: {trend.split(':', 1)[0]}\n\n"
        f"⚡ RSI: {rsi:.1f}\n\n"
        f"💰 Entry: {_price(entry)}\n\n"
        f"🎯 Target: {_price(target)}\n\n"
        f"🛑 Stop: {_price(stop)}\n\n"
        f"📐 Reward:risk ~{reward_risk:.2f}\n\n"
        f"🕒 {timeframe.upper()} confirmed candle"
    )
    headline = f"🧪 {ticker} HYPOTHETICAL {direction.upper()} SCENARIO"
    return _fit_caption(f"{headline}\n\n{body}", config.SIGNAL_DISCLAIMER)


def generate_signal_outcome(ticker: str, timeframe: str, direction: str, entry: float,
                             exit_price: float, outcome: str, r_multiple: float,
                             close_time: str, record_line: str | None = None) -> str:
    """Deterministic factual close-out for a previously posted hypothetical
    scenario. `record_line`, when supplied, appends the running track record so
    each close reinforces the scorecard rather than reading as an isolated result."""
    result_emoji = "✅" if outcome == "target" else "🛑"
    label = "TARGET HIT" if outcome == "target" else "STOP HIT"
    body = (
        f"{result_emoji} {direction.capitalize()} scenario: {label.lower()}\n\n"
        f"💰 Entry {_price(entry)} → close {_price(exit_price)}\n\n"
        f"📐 Result: {r_multiple:+.2f}R\n\n"
        f"🕒 Confirmed {timeframe.upper()} close · {close_time[:16]} UTC"
    )
    if record_line:
        body += f"\n\n{record_line}"
    headline = f"📊 {ticker} HYPOTHETICAL SCENARIO CLOSED: {label}"
    return _fit_caption(f"{headline}\n\n{body}", config.SIGNAL_DISCLAIMER)


def generate_signal_scorecard(scorecard: dict) -> str:
    """Weekly educational review of the hypothetical-signal track record. The
    scorecard chart carries the exact figures; the caption teaches what they mean."""
    total = scorecard["closed_count"]
    if not total:
        context = (
            "No hypothetical scenarios have reached their target or stop yet.\n"
            f"Scenarios still open: {scorecard['open_count']}."
        )
        fallback = (
            "📊 No hypothetical scenarios have closed yet, so there is no record to score\n\n"
            "🧭 Every scenario is defined in advance with a fixed entry, target and stop, then left to play out untouched\n\n"
            "📚 Reward to risk, written as R, compares what a scenario aimed to gain against what it put at risk\n\n"
            "👀 The scorecard fills in as scenarios close, wins and losses alike"
        )
    else:
        parts = [
            f"Closed scenarios: {total}",
            f"Reached target: {scorecard['win_count']}",
            f"Hit stop: {scorecard['loss_count']}",
            f"Win rate: {scorecard['win_rate_pct']:.0f}%",
            f"Average R multiple: {scorecard['avg_r']:+.2f}",
            f"Cumulative R: {scorecard['total_r']:+.2f}",
            f"Scenarios still open: {scorecard['open_count']}",
        ]
        if scorecard["best"]:
            parts.append(f"Best result: {scorecard['best']['ticker']} {scorecard['best']['r_multiple']:+.2f}R")
        if scorecard["worst"]:
            parts.append(f"Worst result: {scorecard['worst']['ticker']} {scorecard['worst']['r_multiple']:+.2f}R")
        context = "\n".join(parts)
        fallback = (
            "📊 Here is how the hypothetical scenarios have played out so far\n\n"
            "📚 R measures reward against risk: a method can still come out ahead while being wrong more often than right, if its wins run larger than its losses\n\n"
            "⚖️ Showing the losing scenarios next to the winning ones is the only honest way to judge an approach\n\n"
            "🔬 A handful of results proves little on its own; the value is in the repeatable discipline, not any single outcome\n\n"
            "👀 New scenarios open after deep dives and close when they reach target or stop"
        )
    return _complete(
        SIGNAL_SCORECARD_PROMPT,
        context,
        "🧪 HYPOTHETICAL SIGNAL SCORECARD",
        fallback,
        config.SIGNAL_DISCLAIMER,
    )


def generate_what_to_watch(watch_rows: list[dict], backdrop: dict | None = None) -> str:
    """Forward-looking weekly post: which assets sit closest to a decision level,
    plus the broad backdrop. The chart shows the distances; the caption explains
    what a break or hold would mean."""
    backdrop = backdrop or {}
    lines = ["Assets closest to a key decision level, from confirmed daily candles:"]
    for row in watch_rows:
        side = "below its resistance" if row["level_type"] == "resistance" else "above its support"
        lines.append(
            f"{row['ticker']}: {row['trend'].split(':', 1)[0]} trend, RSI {row['rsi']:.0f}, "
            f"closing about {abs(row['distance_pct']):.1f}% {side}"
        )
    if backdrop.get("dominance_note"):
        lines.append(backdrop["dominance_note"])
    fear_greed = backdrop.get("fear_greed")
    if fear_greed:
        lines.append(f"Fear and Greed mood: {fear_greed['classification']} ({fear_greed['value']}/100)")
    if backdrop.get("funding_note"):
        lines.append(backdrop["funding_note"])

    watch_tickers = ", ".join(row["ticker"] for row in watch_rows) or "the tracked majors"
    fallback = (
        f"🔭 On watch this week: {watch_tickers}\n\n"
        "🎯 Each is pressing toward a support or resistance zone, and it is a decisive close through that zone, or a clean bounce off it, that would actually change the picture\n\n"
        "⚡ Stretched momentum tends to precede a pause or a shakeout rather than an automatic reversal\n\n"
        "🧭 Watching how price behaves at a level teaches more than chasing it in the space between levels\n\n"
        "🕒 Levels drawn from confirmed daily candles"
    )
    return _complete(
        WHAT_TO_WATCH_PROMPT,
        "\n".join(lines),
        "🔭 WHAT TO WATCH THIS WEEK",
        fallback,
    )
