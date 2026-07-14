"""Balanced publisher for the crypto channel.

Scheduled runs produce one daily market map, one two-asset chart album, and a
weekday macro close. Confirmed candles and per-slot state prevent unfinished or
duplicate analysis from being published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
import time
from zoneinfo import ZoneInfo

import chart_generator
import config
import data_fetcher
import indicators
import narrative_generator
import state_manager
import telegram_publisher


ROME = ZoneInfo("Europe/Rome")
NEW_YORK = ZoneInfo("America/New_York")


def check_followup(prev_entry: dict, current_price: float) -> str | None:
    if not prev_entry:
        return None
    support = prev_entry["levels"]["support"]
    resistance = prev_entry["levels"]["resistance"]
    pct = config.FOLLOWUP_THRESHOLD_PCT / 100
    if current_price > resistance * (1 + pct):
        return f"Price moved above the ~{resistance:.2f} resistance zone from the previous post and is now {current_price:.2f}."
    if current_price < support * (1 - pct):
        return f"Price moved below the ~{support:.2f} support zone from the previous post and is now {current_price:.2f}."
    return None


def _today() -> str:
    return datetime.now(ROME).date().isoformat()


def _slot_already_posted(state: dict, slot: str) -> bool:
    return state.get("_schedule", {}).get(slot) == _today()


def _mark_slot(state: dict, slot: str) -> None:
    state.setdefault("_schedule", {})[slot] = _today()


def _asset(ticker: str) -> dict:
    return next(asset for asset in config.ASSETS if asset["ticker"] == ticker)


def _publish(chat_id: str, image_paths: list[str], caption: str) -> dict:
    return telegram_publisher.post_charts(chat_id, image_paths, caption)


def run_market_map(state: dict, chat_id: str, force: bool = False) -> None:
    if not force and _slot_already_posted(state, "market_map"):
        print("Today's market map was already posted; skipping duplicate.")
        return
    snapshot = data_fetcher.fetch_market_snapshot(config.ASSETS)
    fear_greed = data_fetcher.fetch_fear_greed_index()
    chart_path = chart_generator.generate_market_map_chart(snapshot)
    caption = narrative_generator.generate_market_map(snapshot, fear_greed)
    _publish(chat_id, [chart_path], caption)
    _mark_slot(state, "market_map")
    print("Posted daily market map.")


def run_deep_dive(state: dict, chat_id: str, force: bool = False) -> None:
    if not force and _slot_already_posted(state, "deep_dive"):
        print("Today's deep dive was already posted; skipping duplicate.")
        return

    pair = config.DEEP_DIVE_ROTATION[datetime.now(ROME).weekday()]
    timeframe = config.DEEP_DIVE_TIMEFRAME
    analyses = []
    chart_paths = []
    pending_state = []

    for ticker in pair:
        asset = _asset(ticker)
        print(f"Preparing {ticker} {timeframe} deep dive.")
        frame = data_fetcher.fetch_asset_klines(asset, timeframe, config.CANDLE_LIMIT)
        frame = indicators.enrich(frame)
        levels = indicators.find_key_levels(frame)
        current_price = float(frame["Close"].iloc[-1])
        candle_closed_at = frame["CloseTime"].iloc[-1].isoformat()
        previous = state_manager.get_entry(state, asset["symbol"], timeframe)

        followup = check_followup(previous, current_price)
        if followup and previous.get("message_id"):
            text = narrative_generator.generate_followup(ticker, timeframe, followup)
            telegram_publisher.reply_to_message(chat_id, previous["message_id"], text)
            print(f"Posted {ticker} level follow-up.")

        chart_paths.append(chart_generator.generate_chart(frame, asset["symbol"], timeframe, levels))
        analyses.append({
            "ticker": ticker,
            "timeframe": timeframe,
            "summary": indicators.summarize_for_prompt(frame, levels),
            "trend": indicators.trend_state(frame),
            "rsi": float(frame["rsi"].iloc[-1]),
            "price": current_price,
            "levels": levels,
        })
        pending_state.append((asset, levels, current_price, candle_closed_at))
        time.sleep(1)

    fear_greed = data_fetcher.fetch_fear_greed_index()
    caption = narrative_generator.generate_comparison(analyses, fear_greed)
    posted = _publish(chat_id, chart_paths, caption)
    for asset, levels, current_price, candle_closed_at in pending_state:
        state_manager.set_entry(state, asset["symbol"], timeframe, {
            "levels": levels,
            "price_at_post": current_price,
            "message_id": posted.get("message_id"),
            "posted_at": posted.get("date"),
            "candle_closed_at": candle_closed_at,
        })
    _mark_slot(state, "deep_dive")
    print(f"Posted deep-dive album for {' + '.join(pair)}.")


def _technical_event(previous: dict | None, price: float, trend: str) -> tuple[int, str] | None:
    if not previous:
        return None
    prior_levels = previous["levels"]
    zone_width = float(prior_levels.get("zone_width", 0))
    if price > prior_levels["resistance"] + zone_width:
        return 2, f"Closed above the previous resistance zone near {prior_levels['resistance']:,.4g}"
    if price < prior_levels["support"] - zone_width:
        return 2, f"Closed below the previous support zone near {prior_levels['support']:,.4g}"

    current_regime = trend.split(":", 1)[0]
    previous_regime = previous.get("trend", "").split(":", 1)[0]
    directional = {"bullish", "bearish"}
    if current_regime in directional and previous_regime in directional and current_regime != previous_regime:
        return 1, f"EMA and price structure changed from {previous_regime} to {current_regime}"
    return None


def run_alert_scan(state: dict, chat_id: str, force: bool = False) -> None:
    """Scan confirmed 4h candles but publish only significant, capped events."""
    timeframe = "4h"
    candidates = []
    staged_entries = []
    scanned = 0

    for asset in config.ASSETS:
        try:
            frame = indicators.enrich(
                data_fetcher.fetch_asset_klines(asset, timeframe, config.CANDLE_LIMIT)
            )
            levels = indicators.find_key_levels(frame)
            price = float(frame["Close"].iloc[-1])
            close_time = frame["CloseTime"].iloc[-1].isoformat()
            trend = indicators.trend_state(frame)
            previous = state_manager.get_entry(state, asset["symbol"], timeframe)
            scanned += 1
        except Exception as exc:
            print(f"Skipping {asset['ticker']} alert scan: {exc}", file=sys.stderr)
            continue

        if not force and previous and previous.get("candle_closed_at") == close_time:
            continue

        event = _technical_event(previous, price, trend)
        entry = {
            "levels": levels,
            "price_at_post": price,
            "message_id": previous.get("message_id") if previous else None,
            "posted_at": previous.get("posted_at") if previous else None,
            "candle_closed_at": close_time,
            "trend": trend,
        }
        staged_entries.append((asset, entry))
        if event:
            severity, description = event
            candidates.append({
                "severity": severity,
                "asset": asset,
                "frame": frame,
                "levels": levels,
                "price": price,
                "close_time": close_time,
                "description": description,
                "entry": entry,
            })

    if not scanned:
        raise RuntimeError("No assets could be scanned for confirmed 4h events")

    tracker = state.get("_alerts", {})
    used = tracker.get("count", 0) if tracker.get("date") == _today() else 0
    available = max(config.MAX_EVENT_ALERTS_PER_DAY - used, 0)
    selected = sorted(candidates, key=lambda item: item["severity"], reverse=True)[:available]

    for candidate in selected:
        asset = candidate["asset"]
        path = chart_generator.generate_chart(
            candidate["frame"], asset["symbol"], timeframe, candidate["levels"]
        )
        caption = narrative_generator.generate_event_alert(
            asset["ticker"], candidate["price"], candidate["description"],
            candidate["levels"], candidate["close_time"],
            candidate["frame"].attrs.get("source", "Binance"),
        )
        posted = _publish(chat_id, [path], caption)
        candidate["entry"]["message_id"] = posted.get("message_id")
        candidate["entry"]["posted_at"] = posted.get("date")
        used += 1
        print(f"Posted confirmed 4h alert for {asset['ticker']}.")

    for asset, entry in staged_entries:
        state_manager.set_entry(state, asset["symbol"], timeframe, entry)
    state["_alerts"] = {"date": _today(), "count": used}
    if not selected:
        print("4h scan complete; no publishable event within today's alert cap.")


def run_macro_close(state: dict, chat_id: str, force: bool = False) -> None:
    new_york_now = datetime.now(NEW_YORK)
    if not force and new_york_now.weekday() < 5 and (new_york_now.hour, new_york_now.minute) < (16, 10):
        print("US cash session has not closed; waiting for the next macro cron.")
        return
    if not force and _slot_already_posted(state, "macro_close"):
        print("Today's macro close was already posted; skipping duplicate.")
        return
    macro = data_fetcher.fetch_macro_snapshot()
    chart_path = chart_generator.generate_macro_chart(macro)
    caption = narrative_generator.generate_macro(macro)
    _publish(chat_id, [chart_path], caption)
    _mark_slot(state, "macro_close")
    print("Posted macro close.")


def _automatic_mode() -> str:
    """Map the three UTC GitHub cron times to their content slots."""
    if os.environ.get("CRON_SCHEDULE") == "10 */4 * * *":
        return "alerts"
    hour = datetime.now(timezone.utc).hour
    if hour < 9:
        return "market_map"
    if hour < 17:
        return "deep_dive"
    return "macro_close"


def run(mode: str = "auto", force: bool = False) -> None:
    state = state_manager.load_state()
    chat_id = _require_env("TELEGRAM_CHANNEL")
    resolved = _automatic_mode() if mode == "auto" else mode
    modes = ["market_map", "deep_dive", "macro_close"] if resolved == "all" else [resolved]
    handlers = {
        "market_map": run_market_map,
        "deep_dive": run_deep_dive,
        "macro_close": run_macro_close,
        "alerts": run_alert_scan,
    }

    try:
        for selected in modes:
            try:
                handlers[selected](state, chat_id, force=force)
            except Exception as exc:
                print(f"ERROR in {selected}: {exc}", file=sys.stderr)
                if len(modes) == 1:
                    raise
            state_manager.save_state(state)
    finally:
        state_manager.save_state(state)
    print("State saved.")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("auto", "market_map", "deep_dive", "macro_close", "alerts", "all"),
        default=os.environ.get("POST_MODE", "auto"),
    )
    parser.add_argument("--force", action="store_true", help="republish even if today's slot is recorded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    run(arguments.mode, arguments.force)
