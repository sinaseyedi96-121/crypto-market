"""
Entry point. Run this from GitHub Actions (or locally) — it does one full
pass over all SYMBOLS x TIMEFRAMES defined in config.py:

  1. Pull real OHLC data from Binance
  2. Compute indicators + a support/resistance zone
  3. Check whether price has broken a zone flagged in a previous run,
     and if so, post a factual follow-up reply on that original post
  4. Render a real chart (candles + EMAs + zone lines)
  5. Ask DeepSeek for a short descriptive write-up (no trade instructions)
  6. Post chart + caption to Telegram
  7. Save state back to post_history.json for the next run
"""

import sys
import time

import config
import data_fetcher
import indicators
import chart_generator
import narrative_generator
import telegram_publisher
import state_manager


def check_followup(prev_entry: dict, current_price: float) -> str | None:
    """Returns a factual update string if price has moved meaningfully beyond
    the zone flagged in the previous post, otherwise None."""
    if not prev_entry:
        return None
    support = prev_entry["levels"]["support"]
    resistance = prev_entry["levels"]["resistance"]
    pct = config.FOLLOWUP_THRESHOLD_PCT / 100

    if current_price > resistance * (1 + pct):
        return f"price has moved above the ~{resistance:.2f} resistance zone flagged in the previous post (now {current_price:.2f})."
    if current_price < support * (1 - pct):
        return f"price has moved below the ~{support:.2f} support zone flagged in the previous post (now {current_price:.2f})."
    return None


def run() -> None:
    state = state_manager.load_state()
    fear_greed = data_fetcher.fetch_fear_greed_index()

    chat_id = _require_env("TELEGRAM_CHANNEL")

    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            print(f"--- {symbol} {timeframe} ---")
            try:
                df = data_fetcher.fetch_klines(symbol, timeframe, config.CANDLE_LIMIT)
                df = indicators.enrich(df)
                levels = indicators.find_key_levels(df)
                current_price = float(df["Close"].iloc[-1])

                # 1) follow-up on the PREVIOUS post for this symbol/timeframe, if warranted
                prev_entry = state_manager.get_entry(state, symbol, timeframe)
                followup_msg = check_followup(prev_entry, current_price)
                if followup_msg and prev_entry.get("message_id"):
                    reply_text = narrative_generator.generate_followup(symbol, timeframe, followup_msg)
                    telegram_publisher.reply_to_message(chat_id, prev_entry["message_id"], reply_text)
                    print("Posted follow-up.")

                # 2) new chart + narrative post
                chart_path = chart_generator.generate_chart(df, symbol, timeframe, levels)
                summary = indicators.summarize_for_prompt(df, levels)
                caption = narrative_generator.generate_narrative(symbol, timeframe, summary, fear_greed)

                posted = telegram_publisher.post_chart(chat_id, chart_path, caption)

                # 3) persist state for the next run's follow-up check
                state_manager.set_entry(state, symbol, timeframe, {
                    "levels": levels,
                    "price_at_post": current_price,
                    "message_id": posted.get("message_id"),
                    "posted_at": posted.get("date"),
                })

            except Exception as e:
                # one symbol/timeframe failing shouldn't kill the whole run
                print(f"ERROR on {symbol} {timeframe}: {e}", file=sys.stderr)

            time.sleep(1)  # small pause between Telegram calls

    state_manager.save_state(state)
    print("State saved.")


def _require_env(name: str) -> str:
    import os
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


if __name__ == "__main__":
    run()
