"""Generate a readable market overview with OpenAI's API.

Every public generator takes a ``language`` ("en" or "fa"). The chart image is
shared across channels — numbers are numbers — so only the caption text is
localized: the OpenAI prompt gains a Farsi output directive, and the
deterministic fallbacks, headlines, and disclaimers all have Farsi variants so
the Farsi channel stays coherent even when OpenAI is unavailable.
"""

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
    "stops or targets, and never predict a specific price. Write clear language an intelligent "
    "beginner could follow. Keep the response under 850 characters."
)

_FORMAT_RULES = (
    "Every line starts directly with one relevant emoji. Do not use hyphens, bullet characters, "
    "numbered lists, headings, markdown, or blank lines; the application adds spacing later."
)

# Appended to any system prompt when the target channel is Farsi. The English
# instructions still drive the analysis; this only switches the output language.
_FARSI_DIRECTIVE = (
    "\n\nIMPORTANT — OUTPUT LANGUAGE: Write your ENTIRE response in fluent, natural Persian "
    "(Farsi), in Persian script, for an Iranian crypto audience. Keep every formatting rule "
    "above: begin each line with one relevant emoji, no markdown, no blank lines. Keep asset "
    "tickers (BTC, ETH, SOL, …) and the abbreviation RSI in Latin letters. Do not write any "
    "full English sentences."
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

BLUESKY_PROMPT = (
    "You compress a longer crypto post into a single Bluesky post. Keep the one most useful, "
    "genuinely educational insight and drop everything secondary. Do not restate specific numbers, "
    "prices, or percentages. Write natural English, at most two short sentences, one leading emoji "
    "is fine, no hashtags, no markdown. Never give advice or predict a price. Output ONLY the post "
    "text, and keep it well under the character budget you are given."
)


# ---- Localization helpers ----------------------------------------------------

def _pick(language: str, en, fa):
    """Return the Farsi variant for the Farsi channel, English otherwise."""
    return fa if language == "fa" else en


def _disclaimer(language: str) -> str:
    return config.DISCLAIMERS.get(language, config.DISCLAIMERS["en"])


def _signal_disclaimer(language: str) -> str:
    return config.SIGNAL_DISCLAIMERS.get(language, config.SIGNAL_DISCLAIMERS["en"])


# Trend regime words (the part before ':' in indicators.trend_state output).
_REGIME_FA = {
    "bullish": "صعودی",
    "bearish": "نزولی",
    "neutral": "خنثی",
    "mixed/transitioning": "درهم و در حال گذار",
    "mixed": "درهم",
    "transitioning": "در حال گذار",
    "sideways": "خنثی",
}

_FEAR_GREED_FA = {
    "Extreme Fear": "ترس شدید",
    "Fear": "ترس",
    "Neutral": "خنثی",
    "Greed": "طمع",
    "Extreme Greed": "طمع شدید",
}

_DIRECTION_FA = {"long": "لانگ", "short": "شورت"}


def _regime(language: str, trend: str) -> str:
    regime = trend.split(":", 1)[0]
    if language == "fa":
        return _REGIME_FA.get(regime, regime)
    return regime


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_KEY"])


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
              disclaimer: str | None = None, language: str = "en") -> str:
    """Always return a useful titled caption, even if the model returns no text.

    `headline` and `fallback` must already be in `language`; only the OpenAI
    output is language-switched here, via the Farsi directive.
    """
    if disclaimer is None:
        disclaimer = _disclaimer(language)
    if language == "fa":
        system_prompt = system_prompt + _FARSI_DIRECTIVE

    body = ""
    source = "fallback"
    try:
        response = _client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            max_completion_tokens=config.OPENAI_MAX_TOKENS,
            reasoning_effort=config.OPENAI_REASONING_EFFORT,
        )
        content = response.choices[0].message.content or ""
        body = _format_for_telegram(content.strip())
    except Exception as exc:
        print(f"OpenAI narrative unavailable; using factual fallback: {exc}")

    if len(body) < 20:
        body = fallback
    else:
        source = "model"
    # Diagnostic: record whether the live caption came from Luna or the
    # deterministic template, so the model-vs-fallback rate is greppable in the
    # Actions logs ([caption-path]) without a human reading every post.
    print(f"[caption-path] source={source} lang={language} chars={len(body)}")
    return _fit_caption(f"{headline}\n\n{body}", disclaimer)


def _price(value: float) -> str:
    if abs(value) >= 1_000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    if abs(value) >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _sentiment_line(fear_greed: dict | None, language: str = "en") -> str | None:
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
    if language == "fa":
        classification = _FEAR_GREED_FA.get(fear_greed["classification"], fear_greed["classification"])
        return f"{emoji} احساسات بازار {classification} است ({value}/100)"
    return f"{emoji} Sentiment is {fear_greed['classification']} at {value}/100"


def generate_narrative(symbol: str, timeframe: str, indicator_summary: str,
                        fear_greed: dict | None, language: str = "en") -> str:
    context_lines = [f"Symbol: {symbol}", f"Timeframe: {timeframe}", indicator_summary]
    if fear_greed:
        context_lines.append(
            f"Fear & Greed Index: {fear_greed['value']} ({fear_greed['classification']})"
        )

    ticker = symbol.removesuffix("USDT")
    tf = timeframe.upper()
    headline = _pick(
        language,
        f"🔎 {ticker} MARKET STRUCTURE: WHAT CHANGED?",
        f"🔎 ساختار بازار {ticker}: چه چیزی تغییر کرد؟",
    )
    fallback = _pick(
        language,
        f"📊 {ticker} has a newly confirmed {tf} market update\n\n"
        f"👀 The chart shows trend, momentum, volatility and key pivot zones\n\n"
        "🕒 All values use the latest confirmed candle",
        f"📊 {ticker} یک به‌روزرسانی تازه‌ی تأییدشده در تایم‌فریم {tf} دارد\n\n"
        "👀 نمودار روند، مومنتوم، نوسان و نواحی کلیدی قیمت را نشان می‌دهد\n\n"
        "🕒 همه‌ی مقادیر بر پایه‌ی آخرین کندل بسته‌شده هستند",
    )
    return _complete(SYSTEM_PROMPT, "\n".join(context_lines), headline, fallback, language=language)


def generate_comparison(analyses: list[dict], fear_greed: dict | None,
                        language: str = "en") -> str:
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
    tf = first["timeframe"].upper()
    reg1, reg2 = _regime(language, first["trend"]), _regime(language, second["trend"])
    sentiment = _sentiment_line(fear_greed, language)

    if language == "fa":
        headline = f"🔥 {first['ticker']} در برابر {second['ticker']}: کدام ساختار قوی‌تری دارد؟"
        fallback_lines = [
            f"🧭 {first['ticker']} روندی {reg1} دارد و {second['ticker']} روندی {reg2}",
            "⚡ ببینید مومنتوم همچنان هر روند را تأیید می‌کند یا از آن فاصله می‌گیرد",
            "👀 تنها یک بسته‌شدن تأییدشده فراتر از ناحیه‌ی کلیدی هر دارایی ساختارش را تغییر می‌دهد",
            "📚 مقایسه‌ی دو نمودار در کنار هم در واقع سنجش قدرت نسبی است: کدام‌یک وقتی هر دو حرکت می‌کنند پیشتاز است",
        ]
        if sentiment:
            fallback_lines.append(sentiment)
        fallback_lines.append(f"🕒 کندل‌های تأییدشده‌ی {tf} · منابع روی نمودارها")
    else:
        headline = f"🔥 {first['ticker']} vs {second['ticker']}: WHICH HAS THE STRONGER SETUP?"
        fallback_lines = [
            f"📊 {verdict}",
            f"🧭 {first['ticker']} is {reg1} while {second['ticker']} is {reg2}",
            "⚡ Watch whether momentum keeps confirming each trend or starts to diverge from it",
            "👀 A confirmed close beyond either asset's highlighted zone is what would change its structure",
            "📚 Comparing two charts side by side is really a read on relative strength: which one leads when both move",
        ]
        if sentiment:
            fallback_lines.append(sentiment)
        fallback_lines.append(f"🕒 Confirmed {tf} candles · sources shown on charts")

    return _complete(COMPARISON_PROMPT, "\n\n".join(sections), headline,
                     "\n\n".join(fallback_lines), language=language)


def generate_market_map(snapshot: list[dict], fear_greed: dict | None,
                        language: str = "en") -> str:
    ordered = sorted(snapshot, key=lambda row: row["change_24h"], reverse=True)
    advancing = sum(row["change_24h"] > 0 for row in snapshot)
    declining = len(snapshot) - advancing
    lines = [
        f"Market breadth: {advancing} advancing and {declining} declining",
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
        headline = _pick(language, "🚀 CRYPTO MARKET SURGE: BUYERS TAKE CONTROL",
                         "🚀 هجوم خریداران به بازار کریپتو")
        breadth_read = _pick(language,
                             "a broadly bullish tape with buyers in control almost everywhere",
                             "فضایی کاملاً صعودی که خریداران تقریباً همه‌جا کنترل را در دست دارند")
    elif advancing <= 2:
        headline = _pick(language, "🚨 CRYPTO MARKET FLASHES RED", "🚨 بازار کریپتو قرمزپوش شد")
        breadth_read = _pick(language,
                             "a broadly bearish tape with sellers pressing almost everywhere",
                             "فضایی کاملاً نزولی که فروشندگان تقریباً همه‌جا فشار می‌آورند")
    else:
        headline = _pick(language, "⚡ CRYPTO MARKET SPLITS: MOMENTUM IS SHIFTING",
                         "⚡ بازار کریپتو دوپاره شد: چرخش در جریان است")
        breadth_read = _pick(language,
                             "a split, rotational session rather than one clear direction",
                             "جلسه‌ای دوپاره و چرخشی، نه یک جهت روشن")

    by_ticker = {row["ticker"]: row for row in snapshot}
    btc_row, eth_row = by_ticker.get("BTC"), by_ticker.get("ETH")
    same_way = btc_row and eth_row and (btc_row["change_24h"] > 0) == (eth_row["change_24h"] > 0)

    if language == "fa":
        fallback_lines = [
            f"📊 وسعت بازار: {advancing} دارایی رو به رشد در برابر {declining} دارایی ریزشی؛ {breadth_read}",
            f"🔀 {strongest['ticker']} پیشتاز گروه است و {weakest['ticker']} عقب‌مانده‌ترین؛ چرخشی که باید دید ادامه می‌یابد یا برمی‌گردد",
        ]
        if btc_row and eth_row:
            fallback_lines.append(
                "🟠🔵 امروز BTC و ETH هم‌جهت‌اند، که معمولاً بقیه‌ی بازار را به دنبال خود می‌کشد"
                if same_way else
                "🟠🔵 امروز BTC و ETH خلاف جهت هم حرکت می‌کنند؛ دودستگی در صدر که اغلب به شکل بلاتکلیفی در رده‌های پایین‌تر دیده می‌شود"
            )
        sentiment = _sentiment_line(fear_greed, language)
        if sentiment:
            fallback_lines.append(sentiment)
        fallback_lines.append("🕒 نمای ۲۴ساعته‌ی بازار · منبع: CoinGecko")
    else:
        fallback_lines = [
            f"📊 Breadth: {advancing} {'asset' if advancing == 1 else 'assets'} advancing against "
            f"{declining} declining, {breadth_read}",
            f"🔀 {strongest['ticker']} leads the pack while {weakest['ticker']} lags furthest behind, "
            "a rotation worth watching for whether it persists or reverses",
        ]
        if btc_row and eth_row:
            fallback_lines.append(
                "🟠🔵 BTC and ETH are pointed the same way today, which tends to anchor "
                "the rest of the market to their lead"
                if same_way else
                "🟠🔵 BTC and ETH are pulling in different directions today, a split at "
                "the top that often shows up as indecision further down the board"
            )
        sentiment = _sentiment_line(fear_greed, language)
        if sentiment:
            fallback_lines.append(sentiment)
        fallback_lines.append("🕒 24-hour market snapshot · source: CoinGecko")

    return _complete(MARKET_MAP_PROMPT, "\n".join(lines), headline,
                     "\n\n".join(fallback_lines), language=language)


def _change(series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


def generate_macro(macro: dict, language: str = "en") -> str:
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
    sp_date = sp500.index[-1].date()
    dollar_date = dollar.index[-1].date()

    if sp_5d > 0 and dollar_5d < 0:
        headline = _pick(language, "🌍 RISK-ON MOMENTUM BUILDS", "🌍 مومنتوم ریسک‌پذیری در حال شکل‌گیری")
        opening = _pick(language,
            "🌍 Stocks are firming while the dollar eases, a classic risk-on mix that "
            "tends to loosen the backdrop crypto trades within",
            "🌍 سهام تقویت می‌شود و دلار کوتاه می‌آید؛ ترکیبی کلاسیک از ریسک‌پذیری که معمولاً فضای معاملاتی کریپتو را باز می‌کند")
    elif sp_5d < 0 and dollar_5d > 0:
        headline = _pick(language, "⚠️ DOLLAR PRESSURE HITS RISK ASSETS", "⚠️ فشار دلار بر دارایی‌های پرریسک")
        opening = _pick(language,
            "⚠️ Stocks are softening while the dollar firms, a defensive mix that "
            "tends to tighten conditions for risk assets like crypto",
            "⚠️ سهام ضعیف می‌شود و دلار قوت می‌گیرد؛ ترکیبی دفاعی که شرایط را برای دارایی‌های پرریسک مانند کریپتو سخت‌تر می‌کند")
    else:
        headline = _pick(language, "🔍 WALL STREET AND THE DOLLAR DIVERGE", "🔍 وال‌استریت و دلار مسیرشان جدا شد")
        opening = _pick(language,
            "🔍 Stocks and the dollar aren't telling one clean story right now, a "
            "sign the macro backdrop is unsettled rather than firmly risk-on or risk-off",
            "🔍 سهام و دلار فعلاً یک داستان روشن نمی‌گویند؛ نشانه‌ی فضای کلانِ بی‌ثبات، نه کاملاً ریسک‌پذیر و نه ریسک‌گریز")

    fallback_lines = [opening]
    if macro.get("vix") is not None:
        if macro["vix"] >= 25:
            fallback_lines.append(_pick(language,
                "😬 Equity volatility is running hot, which tends to spill into crypto as correlated risk-off flows",
                "😬 نوسان بازار سهام بالاست، که معمولاً به‌صورت جریان‌های ریسک‌گریزِ همبسته به کریپتو سرایت می‌کند"))
        elif macro["vix"] <= 15:
            fallback_lines.append(_pick(language,
                "😌 Equity volatility is calm, a backdrop that has historically let risk assets, crypto included, grind higher with fewer scares",
                "😌 نوسان بازار سهام آرام است؛ فضایی که در گذشته به دارایی‌های پرریسک، از جمله کریپتو، اجازه‌ی رشد آرام با تلاطم کمتر داده است"))
        else:
            fallback_lines.append(_pick(language,
                "😐 Equity volatility sits in a middling range, neither fueling risk appetite nor draining it",
                "😐 نوسان بازار سهام در محدوده‌ای میانه است، نه محرک اشتهای ریسک و نه تضعیف‌کننده‌ی آن"))
    if macro.get("yield_10y") is not None and macro.get("yield_2y") is not None:
        spread = macro["yield_10y"] - macro["yield_2y"]
        if spread < 0:
            fallback_lines.append(_pick(language,
                "🏦 The Treasury curve is inverted, a signal that has historically preceded tighter, more cautious risk conditions",
                "🏦 منحنی بازده معکوس است، سیگنالی که در گذشته اغلب پیش‌درآمد شرایط سخت‌گیرانه‌تر و محتاط‌تر بوده است"))
        else:
            fallback_lines.append(_pick(language,
                "🏦 The Treasury curve holds its normal upward slope, consistent with a steadier growth backdrop",
                "🏦 منحنی بازده شیب صعودی معمول خود را حفظ کرده، سازگار با فضای رشد باثبات‌تر"))
    if macro.get("btc_sp_corr") is not None:
        if abs(macro["btc_sp_corr"]) >= 0.5:
            fallback_lines.append(_pick(language,
                "🔗 BTC is trading closely with stocks right now, so equity swings are likely spilling straight into crypto",
                "🔗 BTC این روزها نزدیک به سهام حرکت می‌کند، پس نوسان‌های بازار سهام احتمالاً مستقیم به کریپتو سرایت می‌کنند"))
        else:
            fallback_lines.append(_pick(language,
                "🔗 BTC's link to stocks is looser right now, leaving more room for crypto-specific drivers to take over",
                "🔗 پیوند BTC با سهام این روزها سست‌تر است و فضای بیشتری برای محرک‌های خاص کریپتو باز می‌گذارد"))
    fallback_lines.append(_pick(language,
        "📚 Watching stocks, the dollar, and rates together — rather than any one alone — is what "
        "separates a macro-driven crypto move from one crypto is making on its own",
        "📚 دیدن هم‌زمانِ سهام، دلار و نرخ بهره — به‌جای هر کدام به‌تنهایی — همان چیزی است که حرکتِ کلان‌محورِ کریپتو را از حرکتی که خودِ کریپتو می‌سازد جدا می‌کند"))
    fallback_lines.append(_pick(language,
        f"🕒 S&P data: {sp_date} · dollar data: {dollar_date} · FRED + CoinGecko",
        f"🕒 داده‌ی S&P: {sp_date} · داده‌ی دلار: {dollar_date} · FRED + CoinGecko"))
    return _complete(MACRO_PROMPT, context, headline, "\n\n".join(fallback_lines), language=language)


def generate_daily_pulse(derivatives_rows: list[dict], trending: list[dict],
                         language: str = "en") -> str:
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
        headline = _pick(language, "🔥 LEVERAGED LONGS ARE PAYING UP", "🔥 لانگ‌های اهرمی دارند هزینه می‌پردازند")
    elif avg_funding < -0.01:
        headline = _pick(language, "🧊 SHORTS ARE PAYING TO STAY SHORT", "🧊 شورت‌ها برای ماندن در شورت هزینه می‌دهند")
    else:
        headline = _pick(language, "⚖️ DERIVATIVES MARKET SITS NEAR BALANCE", "⚖️ بازار مشتقات نزدیک تعادل است")

    long_leaning = sum(1 for row in derivatives_rows if row["funding_rate"] > 0)
    short_leaning = sum(1 for row in derivatives_rows if row["funding_rate"] < 0)
    total = len(derivatives_rows)
    biggest_oi = max(derivatives_rows, key=lambda row: row["open_interest"])

    if language == "fa":
        if long_leaning == total:
            crowd_line = "🐂 هم‌اکنون همه‌ی بازارهای زیرِ نظر به لانگ‌ها پاداش می‌دهند، نشانه‌ی تسلط کامل خریداران اهرمی بر پوزیشن‌گیری"
        elif short_leaning == total:
            crowd_line = "🐻 هم‌اکنون همه‌ی بازارهای زیرِ نظر به شورت‌ها پاداش می‌دهند، نشانه‌ی تسلط کامل فروشندگان اهرمی بر پوزیشن‌گیری"
        else:
            crowd_line = f"⚖️ از {total} بازارِ زیرِ نظر، {long_leaning} به لانگ و {short_leaning} به شورت متمایل‌اند؛ میدانی دوپاره، نه یک ازدحام یک‌طرفه"
        leverage_line = f"🏦 {biggest_oi['ticker']} همچنان بیشترین اوپن‌اینترست گروه را دارد، پس جایی است که یک چرخش تند در فاندینگ یا نوسان قیمت بیشترین اهرم را به لرزه می‌اندازد"
        fallback_lines = [crowd_line, leverage_line]
        if trending:
            names = ", ".join(item["symbol"] for item in trending)
            fallback_lines.append(f"🔎 توجه تازه به سمت {names} رفته، و نام‌های ترند معمولاً پوزیشن‌گیریِ نازک‌تر و واکنشی‌تری نسبت به بزرگان بازار دارند")
        fallback_lines.append("📚 فاندینگ نشان می‌دهد چه کسی شلوغ شده و برای نگه‌داشتن پوزیشنش هزینه می‌دهد؛ اوپن‌اینترست نشان می‌دهد اگر آن جمعیت اشتباه کند چه حجمی از اهرم می‌تواند باز شود")
        fallback_lines.append("🕒 نمای مشتقات دائمی · CoinGecko")
    else:
        if long_leaning == total:
            crowd_line = ("🐂 Every tracked market is paying longs a premium right now, a sign "
                          "leveraged buyers are firmly in control of positioning")
        elif short_leaning == total:
            crowd_line = ("🐻 Every tracked market is paying shorts a premium right now, a sign "
                          "leveraged sellers are firmly in control of positioning")
        else:
            crowd_line = (f"⚖️ {long_leaning} of {total} tracked markets lean long on funding and "
                          f"{short_leaning} lean short, a split field rather than a one-sided crowd")
        leverage_line = (f"🏦 {biggest_oi['ticker']} still carries the deepest open interest of the group, "
                         "so it's where a sharp funding flip or price swing would ripple through the most leverage")
        fallback_lines = [crowd_line, leverage_line]
        if trending:
            names = ", ".join(item["symbol"] for item in trending)
            fallback_lines.append(f"🔎 Fresh attention is chasing {names}, and trending names typically carry "
                                  "thinner, more reflexive positioning than the majors")
        fallback_lines.append("📚 Funding shows who is crowded and paying to hold their side; open interest "
                              "shows how much leverage could unwind if that crowd gets caught wrong")
        fallback_lines.append("🕒 Perpetual derivatives snapshot · CoinGecko")

    return _complete(DAILY_PULSE_PROMPT, context, headline, "\n\n".join(fallback_lines), language=language)


def generate_weekly_digest(weekly_rows: list[dict], scoreboard_rows: list[dict],
                            dominance_history, feargreed_history,
                            total_mcap_history, stablecoin_history,
                            language: str = "en") -> str:
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
        headline = _pick(language, "📅 WEEKLY DIGEST: BROAD STRENGTH ACROSS THE BOARD",
                         "📅 جمع‌بندی هفتگی: قدرت فراگیر در سراسر بازار")
    elif advancing <= 2:
        headline = _pick(language, "📅 WEEKLY DIGEST: A ROUGH WEEK FOR MOST ASSETS",
                         "📅 جمع‌بندی هفتگی: هفته‌ای سخت برای بیشتر دارایی‌ها")
    else:
        headline = _pick(language, "📅 WEEKLY DIGEST: A MIXED WEEK OF ROTATION",
                         "📅 جمع‌بندی هفتگی: هفته‌ای درهم با چرخش")

    if language == "fa":
        fallback_lines = [
            f"🚀 {strongest['ticker']} این هفته پیشتاز بزرگان بازار بود و {weakest['ticker']} عقب‌مانده‌ترین؛ چرخشی که باید دید ادامه می‌یابد یا برمی‌گردد",
            f"📈 {most_overbought['ticker']} بیش از همه به سمت بالا کشیده شده و {most_oversold['ticker']} بیش از همه به سمت پایین — باید مراقب یک مکث یا بازگشت بود، نه فرض ادامه‌ی بی‌وقفه",
            f"⚡ {most_volatile['ticker']} این هفته پهن‌ترین نوسان گروه را داشت، یادآور اینکه اندازه‌ی حرکت به اندازه‌ی متناسبی از ریسک نیاز دارد",
        ]
        if dominance_change > 0:
            fallback_lines.append("₿ دامیننس بیت‌کوین طی هفته بالا رفت، نشانه‌ی تمایل سرمایه به BTC نسبت به بقیه‌ی بازار")
        elif dominance_change < 0:
            fallback_lines.append("₿ دامیننس بیت‌کوین طی هفته پایین آمد، نشانه‌ی چرخش سرمایه از BTC به بازار گسترده‌تر")
        else:
            fallback_lines.append("₿ دامیننس بیت‌کوین طی هفته کمابیش ثابت ماند، بدون چرخش روشن")
        if feargreed_change > 0:
            fallback_lines.append("🧭 احساسات بازار طی هفته به سمت طمع میل کرد")
        elif feargreed_change < 0:
            fallback_lines.append("🧭 احساسات بازار طی هفته به سمت ترس میل کرد")
        else:
            fallback_lines.append("🧭 احساسات بازار طی هفته ثابت ماند")
        if mcap_change_pct > 0 and stablecoin_change_pct < 0:
            fallback_lines.append("💧 ارزش کل بازار رشد کرد در حالی که عرضه‌ی استیبل‌کوین کوچک شد، سازگار با ورود سرمایه از حاشیه به دارایی‌های پرریسک")
        elif mcap_change_pct < 0 and stablecoin_change_pct > 0:
            fallback_lines.append("💧 ارزش کل بازار کوچک شد در حالی که عرضه‌ی استیبل‌کوین رشد کرد، سازگار با خروج سرمایه به حاشیه")
        else:
            fallback_lines.append("💧 ارزش کل بازار و عرضه‌ی استیبل‌کوین این هفته هم‌جهت حرکت کردند، بدون داستان روشنی از ورود یا خروج سرمایه")
        fallback_lines.append("🕒 جمع‌بندی هفتگی · Binance، CoinGecko، alternative.me")
    else:
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
            fallback_lines.append("₿ Bitcoin dominance rose over the week, a sign capital leaned toward BTC over the rest of the market")
        elif dominance_change < 0:
            fallback_lines.append("₿ Bitcoin dominance fell over the week, a sign capital rotated out of BTC and into the broader market")
        else:
            fallback_lines.append("₿ Bitcoin dominance held steady over the week, no clear rotation either way")
        if feargreed_change > 0:
            fallback_lines.append("🧭 Sentiment drifted toward greed over the week")
        elif feargreed_change < 0:
            fallback_lines.append("🧭 Sentiment drifted toward fear over the week")
        else:
            fallback_lines.append("🧭 Sentiment held steady over the week")
        if mcap_change_pct > 0 and stablecoin_change_pct < 0:
            fallback_lines.append("💧 Total market cap grew while stablecoin supply shrank, consistent with capital moving "
                                  "off the sidelines and into risk assets")
        elif mcap_change_pct < 0 and stablecoin_change_pct > 0:
            fallback_lines.append("💧 Total market cap shrank while stablecoin supply grew, consistent with capital moving "
                                  "to the sidelines rather than staying in risk assets")
        else:
            fallback_lines.append("💧 Total market cap and stablecoin supply moved together this week rather than telling a "
                                  "clean story of capital rotating in or out")
        fallback_lines.append("🕒 Weekly digest · Binance, CoinGecko, alternative.me")

    return _complete(WEEKLY_DIGEST_PROMPT, context, headline, "\n\n".join(fallback_lines), language=language)


def generate_followup(symbol: str, timeframe: str, event: dict, language: str = "en") -> str:
    """Deterministic factual reply when a previously flagged level breaks.

    `event` is {"side": "above"|"below", "level": float, "price": float}.
    """
    tf = timeframe.upper()
    level, price, side = event["level"], event["price"], event["side"]
    if language == "fa":
        header = f"📊 به‌روزرسانی · {symbol} ({tf})"
        if side == "above":
            message = f"قیمت از ناحیه‌ی مقاومت حدود {level:,.2f} در پست پیشین عبور کرد و اکنون روی {price:,.2f} است."
        else:
            message = f"قیمت به زیر ناحیه‌ی حمایت حدود {level:,.2f} در پست پیشین رفت و اکنون روی {price:,.2f} است."
    else:
        header = f"📊 Update · {symbol} ({tf})"
        if side == "above":
            message = f"Price moved above the ~{level:,.2f} resistance zone from the previous post and is now {price:,.2f}."
        else:
            message = f"Price moved below the ~{level:,.2f} support zone from the previous post and is now {price:,.2f}."
    return _fit_caption(f"{header}\n\n{message}", _disclaimer(language))


def _event_description(event: dict, language: str) -> str:
    """Render an alert's structured technical event as a localized sentence."""
    kind = event["kind"]
    if language == "fa":
        if kind == "break_up":
            return f"بسته‌شدن بالای ناحیه‌ی مقاومت پیشین نزدیک {event['level']:,.4g}"
        if kind == "break_down":
            return f"بسته‌شدن زیر ناحیه‌ی حمایت پیشین نزدیک {event['level']:,.4g}"
        frm = _REGIME_FA.get(event["from"], event["from"])
        to = _REGIME_FA.get(event["to"], event["to"])
        return f"ساختار میانگین متحرک و قیمت از {frm} به {to} تغییر کرد"
    if kind == "break_up":
        return f"Closed above the previous resistance zone near {event['level']:,.4g}"
    if kind == "break_down":
        return f"Closed below the previous support zone near {event['level']:,.4g}"
    return f"EMA and price structure changed from {event['from']} to {event['to']}"


def generate_event_alert(ticker: str, price: float, event: dict, levels: dict,
                         close_time: str, source: str, language: str = "en") -> str:
    """A deterministic alert avoids an AI call for background 4h scans. `event`
    is the structured technical event (see main._technical_event)."""
    description = _event_description(event, language)
    if language == "fa":
        body = (
            f"🚨 به‌روزرسانی تکنیکال تأییدشده‌ی ۴ساعته‌ی {ticker}\n\n"
            f"📊 {description}\n\n"
            f"💰 بسته‌شدن تأییدشده ${price:,.4g}\n\n"
            f"🎯 حمایت ~{levels['support']:,.4g} · مقاومت ~{levels['resistance']:,.4g}\n\n"
            f"🕒 بسته‌شد {close_time[:16]} UTC · {source}"
        )
    else:
        body = (
            f"🚨 {ticker} confirmed 4H technical update\n\n"
            f"📊 {description}\n\n"
            f"💰 Confirmed close ${price:,.4g}\n\n"
            f"🎯 Support ~{levels['support']:,.4g} · resistance ~{levels['resistance']:,.4g}\n\n"
            f"🕒 Closed {close_time[:16]} UTC · {source}"
        )
    return _fit_caption(body, _disclaimer(language))


# Trade-style label by the timeframe the scenario's chart is built on — purely
# descriptive of how the underlying candles behave, not a recommendation.
# Deep-dive signals are 1D today; the others are here so the label stays
# correct if a signal is ever opened from a faster timeframe.
_STYLE_LABELS = {
    "1d": {
        "en": "Swing/position trade — 1D structure, typically held days to weeks",
        "fa": "معامله‌ی نوسانی/میان‌مدت — بر پایه‌ی ساختار ۱روزه، معمولاً چند روز تا چند هفته",
    },
    "4h": {
        "en": "Short-term swing trade — 4H structure, typically held hours to a few days",
        "fa": "معامله‌ی نوسانی کوتاه‌مدت — بر پایه‌ی ساختار ۴ساعته، معمولاً چند ساعت تا چند روز",
    },
    "1h": {
        "en": "Intraday trade — 1H structure, typically held within a single day",
        "fa": "معامله‌ی درون‌روزی — بر پایه‌ی ساختار ۱ساعته، معمولاً در طول یک روز",
    },
    "15m": {
        "en": "Scalp trade — 15M structure, typically held minutes to a few hours",
        "fa": "معامله‌ی اسکالپ — بر پایه‌ی ساختار ۱۵دقیقه‌ای، معمولاً چند دقیقه تا چند ساعت",
    },
}


def _style_label(timeframe: str, language: str = "en") -> str:
    labels = _STYLE_LABELS.get(timeframe)
    if not labels:
        return timeframe.upper()
    return labels.get(language, labels["en"])


def _pivot_word(direction: str, language: str = "en") -> str:
    if language == "fa":
        return "سقف نوسانی" if direction == "long" else "کف نوسانی"
    return "swing high" if direction == "long" else "swing low"


def _stop_reason(direction: str, language: str = "en") -> str:
    if language == "fa":
        return "زیر حمایت پیشین" if direction == "long" else "بالای مقاومت پیشین"
    return "below prior support" if direction == "long" else "above prior resistance"


def _touch_note(touches: int, language: str = "en") -> str:
    if touches <= 1:
        return ""
    return f" (تست‌شده {touches} بار)" if language == "fa" else f" (tested {touches}×)"


def _pct_move(price: float, entry: float) -> float:
    return abs(price - entry) / entry * 100 if entry else 0.0


def generate_signal_post(ticker: str, timeframe: str, direction: str, entry: float,
                          targets: list[dict], initial_stop: float, stop_touches: int,
                          trend: str, rsi: float, sizing: dict, language: str = "en") -> str:
    """Deterministic hypothetical long/short scenario post (no AI call, so the
    numbers can never drift from what was actually computed). `targets` is the
    scaled ladder from indicators.find_extended_targets, nearest to farthest;
    as each is reached the stop ladders up (see main._stop_for_index) so a
    reader knows this isn't just one static target/stop pair. `sizing` is
    signal_record.compute_position_sizing's result — an illustrative
    position size/leverage figure against a fixed hypothetical account, not
    advice for the reader's own capital (see the disclaimer this post
    carries)."""
    regime = _regime(language, trend)
    tf = timeframe.upper()
    style = _style_label(timeframe, language)
    stop_pct = _pct_move(initial_stop, entry)
    stop_touch_note = _touch_note(stop_touches, language)
    level_word = _pivot_word(direction, language)
    final_rr = targets[-1]["r_multiple"]
    lev = sizing["leverage"]
    lev_label = f"{lev:.2g}x" if lev > 1 else "1x"

    fa = language == "fa"
    target_lines = []
    for i, t in enumerate(targets):
        pct = _pct_move(t["price"], entry)
        touch_note = _touch_note(t["touches"], language)
        is_final = i == len(targets) - 1
        if fa:
            reach = "نزدیک‌ترین" if i == 0 else ("دورترین، فراتر از آستانه‌ی پاداش به ریسک" if is_final else "دورتر")
            label = f"هدف {i + 1}" + (" (نهایی)" if is_final else "")
            target_lines.append(
                f"🎯 {label}: {_price(t['price'])} · {pct:.1f}% فاصله از ورود · "
                f"{reach} {level_word}{touch_note} · {t['r_multiple']:.2f}R"
            )
        else:
            reach = "nearest" if i == 0 else ("farthest, clearing the reward:risk floor" if is_final else "further out")
            label = f"TP{i + 1}" + (" (final)" if is_final else "")
            target_lines.append(
                f"🎯 {label}: {_price(t['price'])} · {pct:.1f}% move from entry · "
                f"{reach} {level_word}{touch_note} · {t['r_multiple']:.2f}R"
            )

    cap_note = ""
    if sizing.get("capped"):
        cap_note = (" (اهرم محدود شده، پس ریسک واقعی کمتر از هدف معمول است)" if fa
                    else " (leverage capped, so realized risk is below the usual target)")

    if fa:
        arrow = "🟢 لانگ" if direction == "long" else "🔴 شورت"
        headline = f"🧪 سناریوی فرضی {_DIRECTION_FA[direction]} روی {ticker}"
        body = "\n\n".join([
            f"{arrow} سناریو · 🧭 {style}",
            f"📈 مبنای روند: {regime}",
            f"⚡ RSI: {rsi:.1f}",
            f"💰 ورود: {_price(entry)}",
            f"🛑 حد ضرر اولیه: {_price(initial_stop)} · {stop_pct:.1f}% فاصله از ورود · "
            f"{_stop_reason(direction, language)}{stop_touch_note}",
            *target_lines,
            f"📊 اندازه‌ی پوزیشن: {sizing['position_size_pct']:.0f}% از یک حساب فرضی با اهرم {lev_label}، "
            f"ریسک ~{sizing['risk_pct']:.2g}% از آن در صورت خوردن حد ضرر{cap_note}",
            f"📐 پاداش به ریسک تا هدف نهایی ~{final_rr:.2f}",
            f"🕒 کندل تأییدشده‌ی {tf}",
        ])
    else:
        arrow = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        headline = f"🧪 {ticker} HYPOTHETICAL {direction.upper()} SCENARIO"
        body = "\n\n".join([
            f"{arrow} scenario · 🧭 {style}",
            f"📈 Trend basis: {regime}",
            f"⚡ RSI: {rsi:.1f}",
            f"💰 Entry: {_price(entry)}",
            f"🛑 Initial stop: {_price(initial_stop)} · {stop_pct:.1f}% move from entry · "
            f"{_stop_reason(direction, language)}{stop_touch_note}",
            *target_lines,
            f"📊 Position size: {sizing['position_size_pct']:.0f}% of a hypothetical account at "
            f"{lev_label} leverage, risking ~{sizing['risk_pct']:.2g}% of it if stopped{cap_note}",
            f"📐 Reward:risk to final target ~{final_rr:.2f}",
            f"🕒 {tf} confirmed candle",
        ])
    return _fit_caption(f"{headline}\n\n{body}", _signal_disclaimer(language))


def generate_signal_partial(ticker: str, timeframe: str, direction: str, hit_targets: list[dict],
                             new_stop: float, remaining_targets: list[dict], close_time: str,
                             language: str = "en") -> str:
    """Deterministic reply when a hypothetical scenario reaches one or more
    targets but others remain open: names what was hit, where the stop now
    sits (it ladders up — see main._stop_for_index), and what's left. Without
    this, a multi-target scenario would stay silent until it fully closed."""
    tf = timeframe.upper()
    next_target = remaining_targets[0]
    if language == "fa":
        headline = f"📊 به‌روزرسانی سناریوی {ticker}: هدف خورد"
        if len(hit_targets) == 1:
            hit_line = (f"🎯 هدف خورد: {_price(hit_targets[0]['price'])} · "
                        f"{hit_targets[0]['r_multiple']:.2f}R روی این بخش از پوزیشن")
        else:
            prices = "، ".join(_price(t["price"]) for t in hit_targets)
            hit_line = f"🎯 اهداف خورد: {prices}"
        body = (
            f"{hit_line}\n\n"
            f"🛑 حد ضرر جابه‌جا شد به {_price(new_stop)}\n\n"
            f"👀 هدف بعدی: {_price(next_target['price'])} ({next_target['r_multiple']:.2f}R)\n\n"
            f"🕒 بسته‌شدن تأییدشده‌ی {tf} · {close_time[:16]} UTC"
        )
    else:
        headline = f"📊 {ticker} scenario update: target hit"
        if len(hit_targets) == 1:
            hit_line = (f"🎯 Target hit: {_price(hit_targets[0]['price'])} · "
                        f"{hit_targets[0]['r_multiple']:.2f}R on that portion of the position")
        else:
            prices = ", ".join(_price(t["price"]) for t in hit_targets)
            hit_line = f"🎯 Targets hit: {prices}"
        body = (
            f"{hit_line}\n\n"
            f"🛑 Stop moved to {_price(new_stop)}\n\n"
            f"👀 Next target: {_price(next_target['price'])} ({next_target['r_multiple']:.2f}R)\n\n"
            f"🕒 Confirmed {tf} close · {close_time[:16]} UTC"
        )
    return _fit_caption(f"{headline}\n\n{body}", _signal_disclaimer(language))


def generate_signal_outcome(ticker: str, timeframe: str, direction: str, entry: float,
                             exit_price: float, outcome: str, r_multiple: float,
                             close_time: str, targets_hit: int = 1, total_targets: int = 1,
                             record_line: str | None = None, language: str = "en") -> str:
    """Deterministic factual close-out for a previously posted hypothetical
    scenario. `outcome` is "target" (every target reached), "stop" (stopped
    before any target), or "partial_stop" (some targets banked, then the
    remainder stopped out) — `r_multiple` is already the blended result
    across every portion of the position. `record_line`, when supplied,
    appends the running track record so each close reinforces the scorecard
    rather than reading as an isolated result."""
    result_emoji = "✅" if outcome == "target" else ("🟡" if outcome == "partial_stop" else "🛑")
    tf = timeframe.upper()
    if language == "fa":
        label = {
            "target": "همه‌ی اهداف رسید",
            "partial_stop": "بخشی از اهداف رسید، سپس حد ضرر خورد",
            "stop": "حد ضرر خورد",
        }[outcome]
        dir_fa = _DIRECTION_FA[direction]
        headline = f"📊 سناریوی فرضی {ticker} بسته شد: {label}"
        body = (
            f"{result_emoji} سناریوی {dir_fa}: {label}\n\n"
            f"💰 ورود {_price(entry)} ← بسته‌شدن {_price(exit_price)}\n\n"
            f"🎯 اهداف رسیده: {targets_hit} از {total_targets}\n\n"
            f"📐 نتیجه‌ی ترکیبی: {r_multiple:+.2f}R\n\n"
            f"🕒 بسته‌شدن تأییدشده‌ی {tf} · {close_time[:16]} UTC"
        )
    else:
        label = {
            "target": "ALL TARGETS REACHED",
            "partial_stop": "PARTIAL TARGET, THEN STOP",
            "stop": "STOP HIT",
        }[outcome]
        headline = f"📊 {ticker} HYPOTHETICAL SCENARIO CLOSED: {label}"
        body = (
            f"{result_emoji} {direction.capitalize()} scenario: {label.lower()}\n\n"
            f"💰 Entry {_price(entry)} → close {_price(exit_price)}\n\n"
            f"🎯 Targets reached: {targets_hit} of {total_targets}\n\n"
            f"📐 Blended result: {r_multiple:+.2f}R\n\n"
            f"🕒 Confirmed {tf} close · {close_time[:16]} UTC"
        )
    if record_line:
        body += f"\n\n{record_line}"
    return _fit_caption(f"{headline}\n\n{body}", _signal_disclaimer(language))


def generate_signal_scorecard(scorecard: dict, language: str = "en") -> str:
    """Weekly educational review of the hypothetical-signal track record. The
    scorecard chart carries the exact figures; the caption teaches what they mean."""
    total = scorecard["closed_count"]
    if not total:
        context = (
            "No hypothetical scenarios have reached their target or stop yet.\n"
            f"Scenarios still open: {scorecard['open_count']}."
        )
        fallback = _pick(language,
            "📊 No hypothetical scenarios have closed yet, so there is no record to score\n\n"
            "🧭 Every scenario is defined in advance with a fixed entry, target and stop, then left to play out untouched\n\n"
            "📚 Reward to risk, written as R, compares what a scenario aimed to gain against what it put at risk\n\n"
            "👀 The scorecard fills in as scenarios close, wins and losses alike",
            "📊 هنوز هیچ سناریوی فرضی‌ای بسته نشده، پس کارنامه‌ای برای امتیازدهی وجود ندارد\n\n"
            "🧭 هر سناریو از پیش با ورود، هدف و حد ضرر ثابت تعریف می‌شود و سپس بدون دخالت رها می‌شود تا نتیجه‌اش مشخص شود\n\n"
            "📚 پاداش به ریسک که با R نشان داده می‌شود، هدفِ سود یک سناریو را با چیزی که به خطر انداخته مقایسه می‌کند\n\n"
            "👀 کارنامه با بسته‌شدن سناریوها، چه برد و چه باخت، پر می‌شود")
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
        fallback = _pick(language,
            "📊 Here is how the hypothetical scenarios have played out so far\n\n"
            "📚 R measures reward against risk: a method can still come out ahead while being wrong more often than right, if its wins run larger than its losses\n\n"
            "⚖️ Showing the losing scenarios next to the winning ones is the only honest way to judge an approach\n\n"
            "🔬 A handful of results proves little on its own; the value is in the repeatable discipline, not any single outcome\n\n"
            "👀 New scenarios open after deep dives and close when they reach target or stop",
            "📊 اینجا می‌بینید سناریوهای فرضی تا اینجا چطور پیش رفته‌اند\n\n"
            "📚 R پاداش را در برابر ریسک می‌سنجد: یک روش می‌تواند حتی وقتی بیشتر اوقات اشتباه می‌کند سودده باشد، به شرطی که بردهایش بزرگ‌تر از باخت‌هایش باشند\n\n"
            "⚖️ نشان‌دادن سناریوهای بازنده در کنار برنده‌ها تنها راه صادقانه برای قضاوت درباره‌ی یک روش است\n\n"
            "🔬 مشتی نتیجه به‌تنهایی چیز زیادی ثابت نمی‌کند؛ ارزش در نظمِ تکرارپذیر است، نه در هر نتیجه‌ی منفرد\n\n"
            "👀 سناریوهای تازه پس از تحلیل‌های عمیق باز می‌شوند و با رسیدن به هدف یا حد ضرر بسته می‌شوند")
    headline = _pick(language, "🧪 HYPOTHETICAL SIGNAL SCORECARD", "🧪 کارنامه‌ی سیگنال‌های فرضی")
    return _complete(SIGNAL_SCORECARD_PROMPT, context, headline, fallback,
                     _signal_disclaimer(language), language=language)


_FOOTER_LINK_RE = re.compile(r'\s*<a\s+href="[^"]*">.*?</a>\s*$', re.DOTALL)


def plain_text(caption: str) -> str:
    """Strip the HTML channel-link footer and unescape entities, so a caption
    built for Telegram (HTML parse mode) can seed a plain-text Bluesky post."""
    without_footer = _FOOTER_LINK_RE.sub("", caption)
    return html.unescape(without_footer).strip()


def generate_bluesky_caption(source_text: str, max_len: int = 250) -> str:
    """Compress an already-written English caption into a short Bluesky teaser
    (English only). Falls back to the source's headline if OpenAI is down."""
    stripped = source_text.strip()
    headline = stripped.splitlines()[0] if stripped else "📊 Crypto market update"
    fallback = headline[:max_len].rstrip()
    context = (
        f"Character budget: {max_len}. Compress this crypto post into one Bluesky post:\n\n{stripped}"
    )
    body = ""
    source = "fallback"
    try:
        response = _client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": BLUESKY_PROMPT},
                {"role": "user", "content": context},
            ],
            max_completion_tokens=config.OPENAI_MAX_TOKENS,
            reasoning_effort=config.OPENAI_REASONING_EFFORT,
        )
        body = (response.choices[0].message.content or "").strip().strip('"').strip()
    except Exception as exc:
        print(f"OpenAI Bluesky caption unavailable; using headline: {exc}")
    if len(body) < 10:
        body = fallback
    else:
        source = "model"
    print(f"[caption-path] source={source} lang=en kind=bluesky chars={len(body)}")
    body = " ".join(body.split())          # collapse to a single line
    if len(body) > max_len:
        body = body[: max_len - 1].rstrip() + "…"
    return body


def generate_what_to_watch(watch_rows: list[dict], backdrop: dict | None = None,
                           language: str = "en") -> str:
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

    watch_tickers = ", ".join(row["ticker"] for row in watch_rows)
    if language == "fa":
        headline = "🔭 این هفته حواستان به اینها باشد"
        tickers_text = watch_tickers or "بزرگان بازار"
        fallback = (
            f"🔭 زیرِ نظر این هفته: {tickers_text}\n\n"
            "🎯 هرکدام به یک ناحیه‌ی حمایت یا مقاومت نزدیک می‌شوند، و این یک بسته‌شدن تعیین‌کننده از آن ناحیه یا یک بازگشت تمیز از روی آن است که واقعاً تصویر را عوض می‌کند\n\n"
            "⚡ مومنتومِ کشیده معمولاً پیش‌درآمد یک مکث یا تکانه است، نه یک بازگشت خودکار\n\n"
            "🧭 تماشای رفتار قیمت روی یک سطح، بیش از دنبال‌کردنش در فاصله‌ی میان سطوح می‌آموزد\n\n"
            "🕒 سطوح از کندل‌های تأییدشده‌ی روزانه"
        )
    else:
        headline = "🔭 WHAT TO WATCH THIS WEEK"
        tickers_text = watch_tickers or "the tracked majors"
        fallback = (
            f"🔭 On watch this week: {tickers_text}\n\n"
            "🎯 Each is pressing toward a support or resistance zone, and it is a decisive close through that zone, or a clean bounce off it, that would actually change the picture\n\n"
            "⚡ Stretched momentum tends to precede a pause or a shakeout rather than an automatic reversal\n\n"
            "🧭 Watching how price behaves at a level teaches more than chasing it in the space between levels\n\n"
            "🕒 Levels drawn from confirmed daily candles"
        )
    return _complete(WHAT_TO_WATCH_PROMPT, "\n".join(lines), headline, fallback, language=language)
