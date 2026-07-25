"""Balanced publisher for the crypto channel.

Scheduled runs produce a daily market map, a daily derivatives pulse, a
two-asset deep-dive album, a weekday macro close, and a Sunday weekly digest.
Confirmed candles and per-slot state prevent unfinished or duplicate analysis
from being published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd

import bluesky_publisher
import chart_generator
import config
import data_fetcher
import indicators
import narrative_generator
import signal_record
import state_manager
import telegram_publisher


ROME = ZoneInfo("Europe/Rome")
NEW_YORK = ZoneInfo("America/New_York")


def _signal_direction(trend: str) -> str | None:
    regime = trend.split(":", 1)[0]
    if regime == "bullish":
        return "long"
    if regime == "bearish":
        return "short"
    return None


def _stop_for_index(signal: dict, idx: int) -> float:
    """The stop that is (or would be) in force once `idx` targets have been
    banked: the initial technical stop before any target, breakeven after the
    first, then each further target locks in the one before it — so a runner
    can never give back more than it has already banked."""
    if idx == 0:
        return signal["initial_stop"]
    if idx == 1:
        return signal["entry"]
    return signal["targets"][idx - 2]["price"]


def _active_stop(signal: dict) -> float:
    return _stop_for_index(signal, signal["next_target_index"])


def _signal_progress(signal: dict, price: float) -> dict | None:
    """Checks a confirmed close against an open hypothetical signal's current
    target ladder and active stop. Returns None if nothing has changed since
    the last check, else one of:
      {"event": "partial", "hit_indices": [...]}       - some target(s) reached, still open
      {"event": "final_target", "hit_indices": [...]}  - the last target reached, fully closed
      {"event": "stop"}                                 - active stop breached, fully closed
    Targets are ordered nearest-to-farthest, so a confirmed close clears a
    contiguous prefix of whatever remains — checked in order, stopping at the
    first one it doesn't clear.
    """
    direction = signal["direction"]
    remaining = signal["targets"][signal["next_target_index"]:]

    hit_indices = []
    for offset, target in enumerate(remaining):
        hit = price >= target["price"] if direction == "long" else price <= target["price"]
        if not hit:
            break
        hit_indices.append(signal["next_target_index"] + offset)

    if hit_indices:
        event = "final_target" if len(hit_indices) == len(remaining) else "partial"
        return {"event": event, "hit_indices": hit_indices}

    stop = _active_stop(signal)
    stop_hit = price <= stop if direction == "long" else price >= stop
    return {"event": "stop"} if stop_hit else None


def _apply_target_hits(signal: dict, hit_indices: list[int]) -> None:
    """Mutates an open (still-active) signal in place to bank the given
    targets: advances the ladder, moves the stop up, and folds each target's
    R (weighted by its equal share of the position) into `realized_r`."""
    portion_pct = 100.0 / len(signal["targets"])
    for idx in hit_indices:
        signal["realized_r"] = signal.get("realized_r", 0.0) + \
            (portion_pct / 100.0) * signal["targets"][idx]["r_multiple"]
    signal["closed_portion_pct"] = signal.get("closed_portion_pct", 0.0) + portion_pct * len(hit_indices)
    signal["next_target_index"] = max(hit_indices) + 1


def _finalize_signal(signal: dict, event: dict, price: float) -> tuple[str, float]:
    """Computes the final blended outcome/R for a signal that is fully
    closing (the last target reached, or the active stop breached). Pure —
    the caller discards this signal record right after (see
    _close_open_signal), so nothing is mutated here."""
    realized_r = signal.get("realized_r", 0.0)
    if event["event"] == "final_target":
        portion_pct = 100.0 / len(signal["targets"])
        for idx in event["hit_indices"]:
            realized_r += (portion_pct / 100.0) * signal["targets"][idx]["r_multiple"]
        return "target", realized_r

    original_risk = abs(signal["entry"] - signal["initial_stop"])
    direction = signal["direction"]
    reward_actual = (price - signal["entry"]) if direction == "long" else (signal["entry"] - price)
    exit_r = reward_actual / original_risk if original_risk else 0.0
    remaining_pct = 100.0 - signal.get("closed_portion_pct", 0.0)
    total_r = realized_r + (remaining_pct / 100.0) * exit_r
    outcome = "partial_stop" if signal.get("closed_portion_pct", 0.0) > 0 else "stop"
    return outcome, total_r


def _maybe_open_signal(state: dict, channels: list[dict], asset: dict, timeframe: str,
                        analysis: dict, frame: pd.DataFrame, chart_message_ids: dict) -> None:
    """Opens one hypothetical long/short scenario per asset, as a chart reply
    under its own post on each channel, if none is already open. Educational
    only — see README. The initial stop is the nearby support/resistance
    level; targets are a scaled ladder of pivot levels (searched across all
    fetched history, nearest to farthest, capped at config.SIGNAL_MAX_TARGETS)
    whose farthest member still clears config.MIN_SIGNAL_RISK_REWARD. If
    nothing clears it, no signal is opened."""
    symbol = asset["symbol"]
    if state_manager.get_open_signal(state, symbol):
        return
    direction = _signal_direction(analysis["trend"])
    if not direction or not chart_message_ids:
        return

    levels = analysis["levels"]
    entry = analysis["price"]
    initial_stop = levels["support"] if direction == "long" else levels["resistance"]
    stop_touches = levels["support_touches"] if direction == "long" else levels["resistance_touches"]
    targets = indicators.find_extended_targets(frame, direction, entry, initial_stop)
    if not targets:
        print(f"Skipped hypothetical {direction} scenario for {asset['ticker']}: "
              f"no level clears the minimum {config.MIN_SIGNAL_RISK_REWARD:.0f}:1 reward:risk.")
        return
    sizing = signal_record.compute_position_sizing(entry, initial_stop)
    signal_chart = chart_generator.generate_chart(
        frame, symbol, timeframe, levels,
        signal={
            "direction": direction, "entry": entry, "stop": initial_stop,
            "targets": [t["price"] for t in targets],
        },
    )

    signal_message_ids = {}
    en_caption = None
    for channel in channels:
        lang = channel["language"]
        reply_to = chart_message_ids.get(lang)
        if not reply_to:
            continue
        try:
            caption = narrative_generator.generate_signal_post(
                asset["ticker"], timeframe, direction, entry, targets, initial_stop, stop_touches,
                analysis["trend"], analysis["rsi"], sizing, language=lang,
            )
            if lang == "en":
                en_caption = caption
            posted = telegram_publisher.reply_with_photo(
                channel["chat_id"], reply_to, signal_chart, caption)
            signal_message_ids[lang] = posted.get("message_id")
            state_manager.append_post_log({
                "timestamp": time.time(),
                "mode": "signal_open",
                "language": lang,
                "ticker": asset["ticker"],
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "entry": entry,
                "targets": targets,
                "stop": initial_stop,
                "position_size_pct": sizing["position_size_pct"],
                "leverage": sizing["leverage"],
                "caption": caption,
                "message_id": posted.get("message_id"),
            })
        except Exception as exc:
            print(f"ERROR opening scenario for {asset['ticker']} on '{lang}' channel: {exc}",
                  file=sys.stderr)

    if not signal_message_ids:
        print(f"Could not open hypothetical scenario for {asset['ticker']} on any channel.")
        return
    state_manager.open_signal(state, symbol, {
        "direction": direction,
        "entry": entry,
        "initial_stop": initial_stop,
        "targets": targets,
        "next_target_index": 0,
        "realized_r": 0.0,
        "closed_portion_pct": 0.0,
        "position_size_pct": sizing["position_size_pct"],
        "leverage": sizing["leverage"],
        "timeframe": timeframe,
        "opened_at": time.time(),
        "message_ids": signal_message_ids,
        "message_id": signal_message_ids.get("en"),
    })
    _post_to_bluesky(en_caption, [signal_chart], "signal_open", signal=True)
    print(f"Opened hypothetical {direction} scenario for {asset['ticker']} "
          f"with {len(targets)} scaled target(s).")


def check_followup(prev_entry: dict, current_price: float) -> dict | None:
    """Return a structured level-break event (or None) so the caption can be
    rendered in each channel's language at publish time."""
    if not prev_entry:
        return None
    support = prev_entry["levels"]["support"]
    resistance = prev_entry["levels"]["resistance"]
    pct = config.FOLLOWUP_THRESHOLD_PCT / 100
    if current_price > resistance * (1 + pct):
        return {"side": "above", "level": resistance, "price": current_price}
    if current_price < support * (1 - pct):
        return {"side": "below", "level": support, "price": current_price}
    return None


def _today() -> str:
    return datetime.now(ROME).date().isoformat()


def _slot_already_posted(state: dict, slot: str) -> bool:
    return state.get("_schedule", {}).get(slot) == _today()


def _mark_slot(state: dict, slot: str) -> None:
    state.setdefault("_schedule", {})[slot] = _today()


def _week_key() -> str:
    iso = datetime.now(ROME).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _slot_already_posted_week(state: dict, slot: str) -> bool:
    return state.get("_schedule_weekly", {}).get(slot) == _week_key()


def _mark_slot_week(state: dict, slot: str) -> None:
    state.setdefault("_schedule_weekly", {})[slot] = _week_key()


def _record_daily_snapshot(state: dict, total_mcap: float, btc_dominance: float,
                           stablecoin_mcap: float) -> None:
    """Accumulate one data point per day so the weekly digest can chart trends
    CoinGecko's free tier has no historical endpoint for (total mcap, dominance)."""
    snapshots = state.setdefault("_daily_snapshots", [])
    today = _today()
    if snapshots and snapshots[-1]["date"] == today:
        return
    snapshots.append({
        "date": today,
        "total_mcap": total_mcap,
        "btc_dominance": btc_dominance,
        "stablecoin_mcap": stablecoin_mcap,
    })
    del snapshots[: max(0, len(snapshots) - config.DAILY_SNAPSHOT_HISTORY_DAYS)]


def _snapshot_series(state: dict, field: str) -> pd.Series:
    snapshots = state.get("_daily_snapshots", [])
    if not snapshots:
        return pd.Series(dtype=float)
    index = pd.to_datetime([row["date"] for row in snapshots])
    return pd.Series([row[field] for row in snapshots], index=index, name=field)


def _asset(ticker: str) -> dict:
    return next(asset for asset in config.ASSETS if asset["ticker"] == ticker)


def _resolve_channels() -> list[dict]:
    """Resolve configured channels to those whose chat-id env var is set. The
    primary (English) channel is required; an optional channel (e.g. Farsi, if
    its secret isn't added yet) is skipped with a warning so the primary keeps
    posting."""
    resolved = []
    for channel in config.CHANNELS:
        chat_id = os.environ.get(channel["env"])
        if not chat_id:
            if channel.get("required"):
                raise RuntimeError(f"Missing required environment variable: {channel['env']}")
            print(f"Channel '{channel['language']}' ({channel['env']}) is not configured; "
                  "skipping it.", file=sys.stderr)
            continue
        resolved.append({**channel, "chat_id": chat_id})
    if not resolved:
        raise RuntimeError("No channels are configured to publish to")
    return resolved


def _message_ids(obj: dict | None) -> dict:
    """Per-channel Telegram message IDs for a stored entry/signal, tolerant of
    legacy single-id records written before dual-channel publishing."""
    if not obj:
        return {}
    ids = obj.get("message_ids")
    if ids:
        return ids
    legacy = obj.get("message_id")
    return {"en": legacy} if legacy else {}


def _post_to_bluesky(source_caption: str | None, image_paths: list[str] | None,
                     mode: str, signal: bool = False) -> None:
    """Mirror an English post to Bluesky as a concise, standalone teaser with a
    link back to the Telegram channel. No-op unless the Bluesky secrets are set;
    failures never affect Telegram publishing. `signal=True` keeps the mandatory
    hypothetical/educational framing on scenario posts, which the 300-char
    compression would otherwise drop."""
    if not source_caption or not bluesky_publisher.is_configured():
        return
    try:
        source_plain = narrative_generator.plain_text(source_caption)
        teaser = narrative_generator.generate_bluesky_caption(
            source_plain, max_len=200 if signal else 250)
        if signal:
            teaser = f"{teaser} ⚠️ Hypothetical & educational, not advice."
        posted = bluesky_publisher.post(
            teaser, image_paths,
            link_url=config.CHANNELS[0]["url"], link_label="📲 Full analysis on Telegram")
        state_manager.append_post_log({
            "timestamp": time.time(),
            "mode": "bluesky",
            "content_mode": mode,
            "caption": teaser,
            "uri": (posted or {}).get("uri"),
        })
        print(f"Mirrored {mode} to Bluesky.")
    except Exception as exc:
        print(f"ERROR mirroring {mode} to Bluesky: {exc}", file=sys.stderr)


def _publish_localized(channels: list[dict], image_paths: list[str], caption_for,
                       mode: str, log_extra: dict | None = None,
                       bluesky_signal: bool = False) -> dict:
    """Publish the same chart(s) to every channel with a per-language caption,
    then mirror the English caption to Bluesky (if configured). Charts are shared
    (numbers are numbers); only the caption is localized. Per-channel failures are
    isolated so one channel can't sink the others.
    Returns {language: posted_result} for the channels that succeeded."""
    results = {}
    captions = {}
    for channel in channels:
        lang = channel["language"]
        try:
            caption = caption_for(lang)
            captions[lang] = caption
            posted = telegram_publisher.post_charts(channel["chat_id"], image_paths, caption)
            results[lang] = posted
            log = {"timestamp": time.time(), "mode": mode, "language": lang,
                   "caption": caption, "message_id": posted.get("message_id")}
            if log_extra:
                log.update(log_extra)
            state_manager.append_post_log(log)
        except Exception as exc:
            print(f"ERROR publishing {mode} to '{lang}' channel: {exc}", file=sys.stderr)
    if not results:
        raise RuntimeError(f"Failed to publish {mode} to any channel")
    _post_to_bluesky(captions.get("en"), image_paths, mode, signal=bluesky_signal)
    return results


def run_market_map(state: dict, channels: list[dict], force: bool = False) -> None:
    try:
        global_snapshot = data_fetcher.fetch_global_snapshot()
        stablecoin_mcap = data_fetcher.fetch_stablecoin_market_cap()
        _record_daily_snapshot(
            state, global_snapshot["total_market_cap_usd"],
            global_snapshot["btc_dominance"], stablecoin_mcap,
        )
    except Exception as exc:
        print(f"Could not record today's market-structure snapshot: {exc}", file=sys.stderr)

    if not force and _slot_already_posted(state, "market_map"):
        print("Today's market map was already posted; skipping duplicate.")
        return
    snapshot = data_fetcher.fetch_market_snapshot(config.ASSETS)
    fear_greed = data_fetcher.fetch_fear_greed_index()
    chart_path = chart_generator.generate_market_map_chart(snapshot)
    _publish_localized(
        channels, [chart_path],
        lambda lang: narrative_generator.generate_market_map(snapshot, fear_greed, lang),
        "market_map",
    )
    _mark_slot(state, "market_map")
    print("Posted daily market map.")


def _post_followup(channels: list[dict], ticker: str, timeframe: str,
                   previous: dict, followup: dict) -> None:
    """Reply with a level-break follow-up under each channel's original post,
    and mirror the English note to Bluesky as a standalone post."""
    prev_ids = _message_ids(previous)
    en_text = None
    for channel in channels:
        lang = channel["language"]
        reply_to = prev_ids.get(lang)
        if not reply_to:
            continue
        try:
            text = narrative_generator.generate_followup(ticker, timeframe, followup, language=lang)
            if lang == "en":
                en_text = text
            posted = telegram_publisher.reply_to_message(channel["chat_id"], reply_to, text)
            state_manager.append_post_log({
                "timestamp": time.time(),
                "mode": "followup",
                "language": lang,
                "ticker": ticker,
                "timeframe": timeframe,
                "caption": text,
                "message_id": posted.get("message_id"),
            })
        except Exception as exc:
            print(f"ERROR posting {ticker} follow-up to '{lang}' channel: {exc}", file=sys.stderr)
    _post_to_bluesky(en_text, None, "followup")
    print(f"Posted {ticker} level follow-up.")


def run_deep_dive(state: dict, channels: list[dict], force: bool = False) -> None:
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
        if followup and _message_ids(previous):
            _post_followup(channels, ticker, timeframe, previous, followup)

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
        pending_state.append((asset, levels, current_price, candle_closed_at, frame))
        time.sleep(1)

    fear_greed = data_fetcher.fetch_fear_greed_index()
    posted_by_lang = _publish_localized(
        channels, chart_paths,
        lambda lang: narrative_generator.generate_comparison(analyses, fear_greed, lang),
        "deep_dive", log_extra={"assets": list(pair), "timeframe": timeframe},
    )

    # Per-channel message IDs: the album's first message anchors follow-ups, and
    # each asset's own chart within the album anchors its hypothetical scenario.
    entry_ids = {lang: posted.get("message_id") for lang, posted in posted_by_lang.items()}
    posted_date = next(iter(posted_by_lang.values())).get("date")
    album_ids_by_lang = {
        lang: (posted.get("album_message_ids") or [posted.get("message_id")] * len(pair))
        for lang, posted in posted_by_lang.items()
    }
    for index, ((asset, levels, current_price, candle_closed_at, frame), analysis) in enumerate(
        zip(pending_state, analyses)
    ):
        state_manager.set_entry(state, asset["symbol"], timeframe, {
            "levels": levels,
            "price_at_post": current_price,
            "message_ids": dict(entry_ids),
            "message_id": entry_ids.get("en"),
            "posted_at": posted_date,
            "candle_closed_at": candle_closed_at,
        })
        chart_message_ids = {lang: ids[index] for lang, ids in album_ids_by_lang.items()}
        _maybe_open_signal(state, channels, asset, timeframe, analysis, frame, chart_message_ids)
    _mark_slot(state, "deep_dive")
    print(f"Posted deep-dive album for {' + '.join(pair)}.")


def _technical_event(previous: dict | None, price: float, trend: str):
    """Return (severity, english_description, event_meta) or None. The meta lets
    the caption be re-rendered per channel language; the description is retained
    for logging and the test contract."""
    if not previous:
        return None
    prior_levels = previous["levels"]
    zone_width = float(prior_levels.get("zone_width", 0))
    if price > prior_levels["resistance"] + zone_width:
        level = prior_levels["resistance"]
        return 2, f"Closed above the previous resistance zone near {level:,.4g}", \
            {"kind": "break_up", "level": level}
    if price < prior_levels["support"] - zone_width:
        level = prior_levels["support"]
        return 2, f"Closed below the previous support zone near {level:,.4g}", \
            {"kind": "break_down", "level": level}

    current_regime = trend.split(":", 1)[0]
    previous_regime = previous.get("trend", "").split(":", 1)[0]
    directional = {"bullish", "bearish"}
    if current_regime in directional and previous_regime in directional and current_regime != previous_regime:
        return 1, f"EMA and price structure changed from {previous_regime} to {current_regime}", \
            {"kind": "regime_change", "from": previous_regime, "to": current_regime}
    return None


def _post_partial_hit(state: dict, channels: list[dict], asset: dict, open_signal: dict,
                      event: dict, close_time: str) -> None:
    """A target was reached but others remain: reply under each channel's
    signal post announcing it, then bank the hit(s) and ladder the stop up.
    The mutation is applied only once at least one channel confirms posting,
    same retry-safe pattern as _close_open_signal, so a total outage next
    tick simply re-evaluates from the still-unbanked state."""
    hit_targets = [open_signal["targets"][i] for i in event["hit_indices"]]
    projected_next_index = max(event["hit_indices"]) + 1
    projected_stop = _stop_for_index(open_signal, projected_next_index)
    remaining_targets = open_signal["targets"][projected_next_index:]

    signal_ids = _message_ids(open_signal)
    posted_any = False
    en_text = None
    for channel in channels:
        lang = channel["language"]
        reply_to = signal_ids.get(lang)
        if not reply_to:
            continue
        try:
            text = narrative_generator.generate_signal_partial(
                asset["ticker"], open_signal["timeframe"], open_signal["direction"],
                hit_targets, projected_stop, remaining_targets, close_time, language=lang,
            )
            if lang == "en":
                en_text = text
            posted = telegram_publisher.reply_to_message(channel["chat_id"], reply_to, text)
            state_manager.append_post_log({
                "timestamp": time.time(),
                "mode": "signal_partial",
                "language": lang,
                "ticker": asset["ticker"],
                "symbol": asset["symbol"],
                "hit_targets": hit_targets,
                "new_stop": projected_stop,
                "caption": text,
                "message_id": posted.get("message_id"),
            })
            posted_any = True
        except Exception as exc:
            print(f"ERROR posting {asset['ticker']} partial target hit to '{lang}' channel: {exc}",
                  file=sys.stderr)

    if not posted_any:
        print(f"Could not post {asset['ticker']} partial target hit on any channel; will retry.")
        return
    _apply_target_hits(open_signal, event["hit_indices"])
    _post_to_bluesky(en_text, None, "signal_partial", signal=True)
    print(f"{asset['ticker']} hit {len(hit_targets)} target(s); "
          f"{len(remaining_targets)} remaining, stop now {projected_stop:,.4g}.")


def _close_open_signal(state: dict, channels: list[dict], asset: dict, open_signal: dict,
                       event: dict, price: float, close_time: str) -> None:
    """Reply with the scenario's outcome under each channel's signal post, then
    close it. `event` is the terminal event from _signal_progress — the final
    target reached, or the active stop breached — either of which fully
    closes the scenario even if some earlier targets were already banked via
    partial hits. The track-record tally is logged exactly once (canonical
    `signal_close`); mirror channels log under `signal_close_mirror` so the
    scorecard, which counts `signal_close` records, never double-counts."""
    outcome, r_multiple = _finalize_signal(open_signal, event, price)
    total_targets = len(open_signal["targets"])
    if event["event"] == "final_target":
        targets_hit = total_targets
        exit_price = open_signal["targets"][event["hit_indices"][-1]]["price"]
    else:
        targets_hit = open_signal["next_target_index"]
        exit_price = price

    # Fold this just-closed scenario into the tally before it is logged, so the
    # reply's running record includes itself. One asset is closing here, so its
    # open slot no longer counts.
    open_count = len(state.get("_signals", {})) - 1
    scorecard = signal_record.load_scorecard(
        open_count=max(open_count, 0),
        include={"ticker": asset["ticker"], "outcome": outcome, "r_multiple": r_multiple},
    )

    signal_ids = _message_ids(open_signal)
    canonical_logged = False
    en_caption = None
    for channel in channels:
        lang = channel["language"]
        reply_to = signal_ids.get(lang)
        if not reply_to:
            continue
        try:
            record_line = signal_record.format_record_line(scorecard, lang)
            outcome_caption = narrative_generator.generate_signal_outcome(
                asset["ticker"], open_signal["timeframe"], open_signal["direction"],
                open_signal["entry"], exit_price, outcome, r_multiple, close_time,
                targets_hit=targets_hit, total_targets=total_targets,
                record_line=record_line, language=lang,
            )
            if lang == "en":
                en_caption = outcome_caption
            outcome_posted = telegram_publisher.reply_to_message(
                channel["chat_id"], reply_to, outcome_caption)
            state_manager.append_post_log({
                "timestamp": time.time(),
                "mode": "signal_close" if not canonical_logged else "signal_close_mirror",
                "language": lang,
                "ticker": asset["ticker"],
                "symbol": asset["symbol"],
                "timeframe": open_signal["timeframe"],
                "direction": open_signal["direction"],
                "entry": open_signal["entry"],
                "exit_price": exit_price,
                "outcome": outcome,
                "r_multiple": r_multiple,
                "targets_hit": targets_hit,
                "total_targets": total_targets,
                "caption": outcome_caption,
                "message_id": outcome_posted.get("message_id"),
            })
            canonical_logged = True
        except Exception as exc:
            print(f"ERROR closing scenario for {asset['ticker']} on '{lang}' channel: {exc}",
                  file=sys.stderr)

    if not canonical_logged:
        print(f"Could not post {asset['ticker']} scenario close on any channel; will retry.")
        return
    state_manager.close_signal(state, asset["symbol"])
    _post_to_bluesky(en_caption, None, "signal_close", signal=True)
    print(f"Closed hypothetical {open_signal['direction']} scenario for "
          f"{asset['ticker']}: {outcome} ({r_multiple:+.2f}R, {targets_hit}/{total_targets} targets).")


def run_alert_scan(state: dict, channels: list[dict], force: bool = False) -> None:
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

        open_signal = state_manager.get_open_signal(state, asset["symbol"])
        if open_signal:
            progress = _signal_progress(open_signal, price)
            if progress and progress["event"] == "partial":
                _post_partial_hit(state, channels, asset, open_signal, progress, close_time)
            elif progress:
                _close_open_signal(state, channels, asset, open_signal, progress, price, close_time)

        if not force and previous and previous.get("candle_closed_at") == close_time:
            continue

        event = _technical_event(previous, price, trend)
        prev_ids = _message_ids(previous)
        entry = {
            "levels": levels,
            "price_at_post": price,
            "message_ids": prev_ids,
            "message_id": prev_ids.get("en"),
            "posted_at": previous.get("posted_at") if previous else None,
            "candle_closed_at": close_time,
            "trend": trend,
        }
        staged_entries.append((asset, entry))
        if event:
            severity, description, event_meta = event
            candidates.append({
                "severity": severity,
                "asset": asset,
                "frame": frame,
                "levels": levels,
                "price": price,
                "close_time": close_time,
                "event": event_meta,
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
        source = candidate["frame"].attrs.get("source", "Binance")
        try:
            results = _publish_localized(
                channels, [path],
                lambda lang: narrative_generator.generate_event_alert(
                    asset["ticker"], candidate["price"], candidate["event"],
                    candidate["levels"], candidate["close_time"], source, language=lang),
                "alert", log_extra={"ticker": asset["ticker"], "timeframe": timeframe},
            )
        except Exception as exc:
            print(f"ERROR posting {asset['ticker']} alert to any channel: {exc}", file=sys.stderr)
            continue
        candidate["entry"]["message_ids"] = {lang: r.get("message_id") for lang, r in results.items()}
        candidate["entry"]["message_id"] = candidate["entry"]["message_ids"].get("en")
        candidate["entry"]["posted_at"] = next(iter(results.values())).get("date")
        used += 1
        print(f"Posted confirmed 4h alert for {asset['ticker']}.")

    for asset, entry in staged_entries:
        state_manager.set_entry(state, asset["symbol"], timeframe, entry)
    state["_alerts"] = {"date": _today(), "count": used}
    if not selected:
        print("4h scan complete; no publishable event within today's alert cap.")


def _enrich_macro_snapshot(macro: dict) -> None:
    """Best-effort extras: VIX, yield spread, BTC correlation. Never blocks the core post."""
    try:
        macro["vix"] = float(data_fetcher.fetch_fred_series(config.FRED_VIX_SERIES).iloc[-1])
    except Exception as exc:
        print(f"VIX fetch failed: {exc}", file=sys.stderr)
    try:
        macro["yield_10y"] = float(data_fetcher.fetch_fred_series(config.FRED_10Y_SERIES).iloc[-1])
        macro["yield_2y"] = float(data_fetcher.fetch_fred_series(config.FRED_2Y_SERIES).iloc[-1])
    except Exception as exc:
        print(f"Treasury yield fetch failed: {exc}", file=sys.stderr)
    try:
        btc = data_fetcher.fetch_asset_klines(_asset("BTC"), "1d", config.CORRELATION_WINDOW + 20)
        macro["btc_sp_corr"] = indicators.rolling_correlation(
            btc["Close"], macro["sp500"], config.CORRELATION_WINDOW
        )
        macro["btc_dollar_corr"] = indicators.rolling_correlation(
            btc["Close"], macro["dollar"], config.CORRELATION_WINDOW
        )
    except Exception as exc:
        print(f"BTC correlation calculation failed: {exc}", file=sys.stderr)


def run_macro_close(state: dict, channels: list[dict], force: bool = False) -> None:
    new_york_now = datetime.now(NEW_YORK)
    if not force and new_york_now.weekday() < 5 and (new_york_now.hour, new_york_now.minute) < (16, 10):
        print("US cash session has not closed; waiting for the next macro cron.")
        return
    if not force and _slot_already_posted(state, "macro_close"):
        print("Today's macro close was already posted; skipping duplicate.")
        return
    macro = data_fetcher.fetch_macro_snapshot()
    _enrich_macro_snapshot(macro)
    chart_path = chart_generator.generate_macro_chart(macro)
    _publish_localized(
        channels, [chart_path],
        lambda lang: narrative_generator.generate_macro(macro, lang),
        "macro_close",
    )
    _mark_slot(state, "macro_close")
    print("Posted macro close.")


def run_daily_pulse(state: dict, channels: list[dict], force: bool = False) -> None:
    if not force and _slot_already_posted(state, "daily_pulse"):
        print("Today's daily pulse was already posted; skipping duplicate.")
        return

    derivatives_rows = data_fetcher.fetch_derivatives_snapshot(config.PULSE_ASSETS)
    if not derivatives_rows:
        raise RuntimeError("Not enough derivatives data to publish the daily pulse")

    try:
        trending = data_fetcher.fetch_trending_coins()
    except Exception as exc:
        print(f"Trending coins fetch failed: {exc}", file=sys.stderr)
        trending = []

    chart_path = chart_generator.generate_pulse_chart(derivatives_rows)
    _publish_localized(
        channels, [chart_path],
        lambda lang: narrative_generator.generate_daily_pulse(derivatives_rows, trending, lang),
        "daily_pulse",
    )
    _mark_slot(state, "daily_pulse")
    print("Posted daily derivatives pulse.")


def run_weekly_digest(state: dict, channels: list[dict], force: bool = False) -> None:
    if not force and datetime.now(ROME).weekday() != config.WEEKLY_DIGEST_WEEKDAY:
        print("Not the weekly digest's scheduled day; skipping.")
        return
    if not force and _slot_already_posted_week(state, "weekly_digest"):
        print("This week's digest was already posted; skipping duplicate.")
        return

    weekly_rows = data_fetcher.fetch_market_snapshot(config.ASSETS)

    scoreboard_rows = []
    for asset in config.ASSETS:
        try:
            frame = indicators.enrich(
                data_fetcher.fetch_asset_klines(asset, config.DEEP_DIVE_TIMEFRAME, config.CANDLE_LIMIT)
            )
        except Exception as exc:
            print(f"Skipping {asset['ticker']} in the technical scoreboard: {exc}", file=sys.stderr)
            continue
        last = frame.iloc[-1]
        scoreboard_rows.append({
            "ticker": asset["ticker"],
            "rsi": float(last["rsi"]),
            "atr_pct": float(last["atr_pct"]),
        })
        time.sleep(0.5)
    if len(scoreboard_rows) < len(config.ASSETS) // 2:
        raise RuntimeError("Not enough assets returned candle data for the technical scoreboard")

    feargreed_history = data_fetcher.fetch_fear_greed_history(30)
    dominance_history = _snapshot_series(state, "btc_dominance")
    total_mcap_history = _snapshot_series(state, "total_mcap")
    stablecoin_history = _snapshot_series(state, "stablecoin_mcap")
    if dominance_history.empty or total_mcap_history.empty or stablecoin_history.empty:
        raise RuntimeError("No accumulated market-structure history yet; the daily market map "
                           "must run at least once before the weekly digest can chart trends")

    chart_paths = [
        chart_generator.generate_weekly_recap_chart(weekly_rows),
        chart_generator.generate_technical_scoreboard_chart(scoreboard_rows),
        chart_generator.generate_market_structure_chart(dominance_history, feargreed_history),
        chart_generator.generate_liquidity_chart(total_mcap_history, stablecoin_history),
    ]
    _publish_localized(
        channels, chart_paths,
        lambda lang: narrative_generator.generate_weekly_digest(
            weekly_rows, scoreboard_rows, dominance_history, feargreed_history,
            total_mcap_history, stablecoin_history, lang),
        "weekly_digest",
    )
    _mark_slot_week(state, "weekly_digest")
    print("Posted weekly digest.")


def run_signal_scorecard(state: dict, channels: list[dict], force: bool = False) -> None:
    """Weekly standalone track record of the hypothetical scenarios. Reads the
    same posts_log the on-close replies write, so the two can never disagree."""
    if not force and datetime.now(ROME).weekday() != config.SIGNAL_SCORECARD_WEEKDAY:
        print("Not the signal scorecard's scheduled day; skipping.")
        return
    if not force and _slot_already_posted_week(state, "signal_scorecard"):
        print("This week's signal scorecard was already posted; skipping duplicate.")
        return

    open_count = len(state.get("_signals", {}))
    closed = signal_record._load_closed_signals()
    scorecard = signal_record.compute_scorecard(closed, open_count=open_count)
    chart_path = chart_generator.generate_signal_scorecard_chart(scorecard, closed)
    _publish_localized(
        channels, [chart_path],
        lambda lang: narrative_generator.generate_signal_scorecard(scorecard, lang),
        "signal_scorecard", bluesky_signal=True,
    )
    _mark_slot_week(state, "signal_scorecard")
    print(f"Posted signal scorecard ({scorecard['closed_count']} closed, {open_count} open).")


def _watch_backdrop(state: dict) -> dict:
    """Best-effort market backdrop for the what-to-watch post. Each piece
    soft-fails so a single flaky fetch never blocks the core watchlist."""
    backdrop: dict = {}
    dominance = _snapshot_series(state, "btc_dominance")
    if len(dominance) >= 2:
        move = float(dominance.iloc[-1] - dominance.iloc[0])
        if move > 0.3:
            backdrop["dominance_note"] = "BTC dominance has been rising, which historically pressures altcoins"
        elif move < -0.3:
            backdrop["dominance_note"] = "BTC dominance has been falling, which often coincides with altcoin strength"
        else:
            backdrop["dominance_note"] = "BTC dominance has been broadly flat"
    try:
        backdrop["fear_greed"] = data_fetcher.fetch_fear_greed_index()
    except Exception as exc:
        print(f"What-to-watch Fear & Greed fetch failed: {exc}", file=sys.stderr)
    try:
        derivatives = data_fetcher.fetch_derivatives_snapshot(config.PULSE_ASSETS)
        if derivatives:
            avg_funding = sum(row["funding_rate"] for row in derivatives) / len(derivatives)
            if avg_funding > 0.01:
                backdrop["funding_note"] = "Perpetual funding is positive on balance, so leveraged longs are crowded"
            elif avg_funding < -0.01:
                backdrop["funding_note"] = "Perpetual funding is negative on balance, so leveraged shorts are crowded"
    except Exception as exc:
        print(f"What-to-watch funding fetch failed: {exc}", file=sys.stderr)
    return backdrop


def run_what_to_watch(state: dict, channels: list[dict], force: bool = False) -> None:
    """Forward-looking weekly post: which assets sit closest to a decision level,
    with the broad backdrop. Built entirely from data the pipeline already fetches."""
    if not force and datetime.now(ROME).weekday() != config.WHAT_TO_WATCH_WEEKDAY:
        print("Not the what-to-watch scheduled day; skipping.")
        return
    if not force and _slot_already_posted_week(state, "what_to_watch"):
        print("This week's what-to-watch was already posted; skipping duplicate.")
        return

    timeframe = config.DEEP_DIVE_TIMEFRAME
    candidates = []
    for asset in config.ASSETS:
        try:
            frame = indicators.enrich(
                data_fetcher.fetch_asset_klines(asset, timeframe, config.CANDLE_LIMIT)
            )
            levels = indicators.find_key_levels(frame)
            price = float(frame["Close"].iloc[-1])
            proximity = indicators.level_proximity(price, levels)
            candidates.append({
                "ticker": asset["ticker"],
                "price": price,
                "rsi": float(frame["rsi"].iloc[-1]),
                "trend": indicators.trend_state(frame),
                **proximity,
            })
        except Exception as exc:
            print(f"Skipping {asset['ticker']} in what-to-watch: {exc}", file=sys.stderr)
        time.sleep(0.5)

    if not candidates:
        raise RuntimeError("No assets could be evaluated for the what-to-watch post")

    # Flag genuine decision points first (near a level or momentum stretched);
    # if too few qualify, just take the assets nearest a level so the post is
    # never empty. Either way, rank by imminence and cap the list.
    def _is_watchworthy(row: dict) -> bool:
        return (abs(row["distance_pct"]) <= config.WATCH_LEVEL_PROXIMITY_PCT
                or row["rsi"] >= config.RSI_OVERBOUGHT
                or row["rsi"] <= config.RSI_OVERSOLD)

    flagged = [row for row in candidates if _is_watchworthy(row)]
    ranked = sorted(flagged or candidates, key=lambda row: abs(row["distance_pct"]))
    watch_rows = ranked[: config.WATCH_MAX_ITEMS]

    backdrop = _watch_backdrop(state)
    chart_path = chart_generator.generate_watchlist_chart(watch_rows)
    _publish_localized(
        channels, [chart_path],
        lambda lang: narrative_generator.generate_what_to_watch(watch_rows, backdrop, lang),
        "what_to_watch",
    )
    _mark_slot_week(state, "what_to_watch")
    print(f"Posted what-to-watch ({len(watch_rows)} assets flagged).")


# Maps each GitHub Actions cron string (github.event.schedule, UTC) to its content slot.
CRON_MODE_MAP = {
    "10 */4 * * *": "alerts",
    "15 4 * * *": "market_map",
    "15 8 * * *": "daily_pulse",
    "15 12 * * *": "deep_dive",
    "15 16 * * 0": "weekly_digest",
    "15 16 * * 6": "signal_scorecard",
    "15 6 * * 1": "what_to_watch",
    "15 20 * * 1-5": "macro_close",
    "15 21 * * 1-5": "macro_close",
}


def _automatic_mode() -> str:
    """Map the GitHub cron schedule to its content slot. Falls back to an hour
    bucket for local/manual runs where CRON_SCHEDULE isn't set."""
    cron = os.environ.get("CRON_SCHEDULE")
    if cron in CRON_MODE_MAP:
        return CRON_MODE_MAP[cron]
    hour = datetime.now(timezone.utc).hour
    if hour < 6:
        return "market_map"
    if hour < 10:
        return "daily_pulse"
    if hour < 17:
        return "deep_dive"
    return "macro_close"


def run(mode: str = "auto", force: bool = False) -> None:
    state = state_manager.load_state()
    channels = _resolve_channels()
    resolved = _automatic_mode() if mode == "auto" else mode
    modes = (
        ["market_map", "daily_pulse", "deep_dive", "macro_close", "weekly_digest",
         "signal_scorecard", "what_to_watch"]
        if resolved == "all" else [resolved]
    )
    handlers = {
        "market_map": run_market_map,
        "deep_dive": run_deep_dive,
        "macro_close": run_macro_close,
        "alerts": run_alert_scan,
        "daily_pulse": run_daily_pulse,
        "weekly_digest": run_weekly_digest,
        "signal_scorecard": run_signal_scorecard,
        "what_to_watch": run_what_to_watch,
    }

    try:
        for selected in modes:
            try:
                handlers[selected](state, channels, force=force)
            except Exception as exc:
                print(f"ERROR in {selected}: {exc}", file=sys.stderr)
                if len(modes) == 1:
                    raise
            state_manager.save_state(state)
    finally:
        state_manager.save_state(state)
    print("State saved.")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("auto", "market_map", "deep_dive", "macro_close", "alerts",
                 "daily_pulse", "weekly_digest", "signal_scorecard", "what_to_watch", "all"),
        default=os.environ.get("POST_MODE", "auto"),
    )
    parser.add_argument("--force", action="store_true", help="republish even if today's slot is recorded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    run(arguments.mode, arguments.force)
