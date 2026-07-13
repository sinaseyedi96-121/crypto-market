"""
Generates the human-readable overview text via DeepSeek's API (OpenAI-compatible).

Design choice: the system prompt explicitly forbids the model from issuing
buy/sell/entry/stop-loss/take-profit instructions. It's asked to *describe*
conditions (trend, momentum, volatility, zones), not *prescribe* actions.
The disclaimer is appended in code, not left to the model to remember.
"""

import os
from openai import OpenAI
import config

SYSTEM_PROMPT = """You are a crypto market analyst writing brief overviews for a \
Telegram channel that publishes educational market commentary.

Always format the reply exactly like this:
1. One opening line naming the trend, starting with 📈 (bullish), 📉 (bearish), \
or ➡️ (sideways/mixed).
2. Then a bulleted list, one short line per bullet, each starting with an emoji:
   - ⚡ Momentum — the RSI and MACD state
   - 📊 Volatility — the ATR / Bollinger Band state
   - 🎯 Key zone — the support/resistance zone to watch
   - A sentiment bullet ONLY if a Fear & Greed value is given in the context, \
using the exact value provided and the emoji matching its classification: \
😱 Extreme Fear, 😨 Fear, 😐 Neutral, 🙂 Greed, 🤑 Extreme Greed.

Rules you always follow:
- Only use numbers given in the context — never invent or restate them incorrectly.
- Never tell the reader to buy, sell, enter, exit, or set a specific stop-loss \
or take-profit price. Describe conditions, not instructions.
- Do not predict a specific future price. You can describe what a break above \
resistance or below support would technically indicate, without saying it \
will happen.
- One emoji per line max — no decorative spam, no "guaranteed" language.
- Write for a general audience — clear and non-technical. Keep the whole \
message to 5-6 short lines total.
"""


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["DEEPSEEK_KEY"],
        base_url=config.DEEPSEEK_BASE_URL,
    )


def generate_narrative(symbol: str, timeframe: str, indicator_summary: str,
                        fear_greed: dict | None) -> str:
    context_lines = [f"Symbol: {symbol}", f"Timeframe: {timeframe}", indicator_summary]
    if fear_greed:
        context_lines.append(
            f"Fear & Greed Index: {fear_greed['value']} ({fear_greed['classification']})"
        )
    user_prompt = "\n".join(context_lines)

    resp = _client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.DEEPSEEK_MAX_TOKENS,
    )
    text = resp.choices[0].message.content.strip()
    return text + config.DISCLAIMER


def generate_followup(symbol: str, timeframe: str, message: str) -> str:
    """Short reply text for when price breaks a previously-flagged zone.
    This is a template, not an LLM call — keeps it factual and avoids any risk
    of the model drifting into instruction-like language for a live update."""
    return f"📊 Update — {symbol} ({timeframe}): {message}{config.DISCLAIMER}"
