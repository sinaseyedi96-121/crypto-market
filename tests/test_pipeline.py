import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

import chart_generator
import config
import data_fetcher
import indicators
import main
import narrative_generator
import signal_record


def synthetic_market(candles: int = 241) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=candles, freq="D")
    phase = np.linspace(0, 16 * np.pi, candles)
    close = 100 + 6 * np.sin(phase)
    frame = pd.DataFrame(
        {
            "Open": close + 0.4 * np.cos(phase),
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 1_000_000 + 200_000 * (1 + np.sin(phase + 0.8)),
            "CloseTime": index + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
        },
        index=index,
    )
    return indicators.enrich(frame)


class ClosedCandleTests(unittest.TestCase):
    def test_open_candle_is_removed(self):
        now = pd.Timestamp("2026-07-13 12:00:00")
        closed_open = int(pd.Timestamp("2026-07-13 04:00:00").timestamp() * 1000)
        closed_end = int(pd.Timestamp("2026-07-13 07:59:59.999").timestamp() * 1000)
        open_open = int(pd.Timestamp("2026-07-13 12:00:00").timestamp() * 1000)
        open_end = int(pd.Timestamp("2026-07-13 15:59:59.999").timestamp() * 1000)
        raw = [
            [closed_open, "100", "110", "90", "105", "123", closed_end],
            [open_open, "105", "112", "103", "108", "45", open_end],
        ]

        result = data_fetcher._parse_closed_klines(raw, now=now)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[-1]["Close"], 105)
        self.assertLessEqual(result.iloc[-1]["CloseTime"], now)

    def test_open_coingecko_aggregate_candle_is_removed(self):
        now = pd.Timestamp("2026-07-13 12:00:00")
        raw = [
            [int(pd.Timestamp("2026-07-13 04:00:00").timestamp() * 1000), 100],
            [int(pd.Timestamp("2026-07-13 05:00:00").timestamp() * 1000), 105],
            [int(pd.Timestamp("2026-07-13 12:00:00").timestamp() * 1000), 108],
        ]
        result = data_fetcher._aggregate_coingecko_prices(raw, "4h", now=now)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[-1]["Close"], 105)
        self.assertEqual(result.iloc[-1]["High"], 105)


class DataFetcherTests(unittest.TestCase):
    def test_fear_greed_payload_parses_oldest_to_newest(self):
        payload = {
            "data": [
                {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1784073600"},
                {"value": "22", "value_classification": "Extreme Fear", "timestamp": "1783987200"},
            ]
        }
        series = data_fetcher._parse_fear_greed_payload(payload)
        self.assertEqual(len(series), 2)
        self.assertTrue(series.index.is_monotonic_increasing)
        self.assertEqual(series.iloc[-1], 25)


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_market()
        self.levels = indicators.find_key_levels(self.frame, lookback=180)

    def test_levels_use_repeated_pivots(self):
        self.assertLess(self.levels["support"], self.frame["Close"].iloc[-1])
        self.assertGreater(self.levels["resistance"], self.frame["Close"].iloc[-1])
        self.assertGreaterEqual(self.levels["support_touches"], 2)
        self.assertGreaterEqual(self.levels["resistance_touches"], 2)
        self.assertEqual(self.levels["lookback"], 180)

    def test_prompt_summary_contains_evidence(self):
        summary = indicators.summarize_for_prompt(self.frame, self.levels)
        self.assertIn("Confirmed candle close time", summary)
        self.assertIn("percentile", summary)
        self.assertIn("pivot touches", summary)
        self.assertIn("MACD is", summary)

    def test_rolling_correlation_detects_direction(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        rng = np.random.default_rng(42)
        returns = rng.normal(0.01, 0.02, 59)
        base = pd.Series(100 * np.cumprod(np.concatenate([[1], 1 + returns])), index=dates)
        # Scaling a series leaves its % returns unchanged, so this is perfectly positively correlated.
        positively_correlated = base * 2 + 5
        # Sign-flipped returns, by construction, are perfectly negatively correlated.
        negatively_correlated = pd.Series(100 * np.cumprod(np.concatenate([[1], 1 - returns])), index=dates)

        positive = indicators.rolling_correlation(base, positively_correlated, window=30)
        negative = indicators.rolling_correlation(base, negatively_correlated, window=30)

        self.assertGreater(positive, 0.9)
        self.assertLess(negative, -0.9)

    def test_rolling_correlation_none_when_not_enough_overlap(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        series = pd.Series([1, 2, 3], index=dates)
        self.assertIsNone(indicators.rolling_correlation(series, series, window=30))


def tiered_pivot_frame() -> pd.DataFrame:
    """A frame with three isolated pivot highs at increasing distances from a
    flat 100 baseline, so tests can check that the farthest one that clears a
    reward:risk bar is preferred over the nearest one."""
    segments = [[100.0] * 20]
    for peak in (110.0, 130.0, 160.0):
        segments.append([100.0, 100.0 + (peak - 100.0) * 0.5, peak,
                          100.0 + (peak - 100.0) * 0.5, 100.0])
        segments.append([100.0] * 5)
    segments.append([100.0] * 5)
    values = [v for seg in segments for v in seg]
    index = pd.date_range("2025-01-01", periods=len(values), freq="D")
    close = pd.Series(values, index=index)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1_000_000.0,
            "CloseTime": index + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
        },
        index=index,
    )
    return indicators.enrich(frame)


class ExtendedTargetTests(unittest.TestCase):
    def test_prefers_the_farthest_level_that_clears_the_bar(self):
        frame = tiered_pivot_frame()
        target = indicators.find_extended_target(frame, "long", entry=100.0, stop=95.0,
                                                   min_reward_risk=3.0)
        self.assertAlmostEqual(target, 160.0, delta=1.0)

    def test_returns_none_when_nothing_clears_the_bar(self):
        frame = tiered_pivot_frame()
        target = indicators.find_extended_target(frame, "long", entry=100.0, stop=-100.0,
                                                   min_reward_risk=3.0)
        self.assertIsNone(target)

    def test_returns_none_for_zero_risk(self):
        frame = tiered_pivot_frame()
        self.assertIsNone(indicators.find_extended_target(frame, "long", entry=100.0, stop=100.0))


class SetupVerdictTests(unittest.TestCase):
    def test_clean_trend_beats_transitioning(self):
        analyses = [
            {"ticker": "BTC", "trend": "bullish: aligned", "rsi": 62.0},
            {"ticker": "SOL", "trend": "mixed/transitioning: no agreement", "rsi": 50.0},
        ]
        verdict = indicators.compare_setups(analyses)
        self.assertIn("BTC", verdict)
        self.assertIn("cleaner structure", verdict)

    def test_two_muddled_setups_report_no_clean_structure(self):
        analyses = [
            {"ticker": "BTC", "trend": "mixed/transitioning: no agreement", "rsi": 50.0},
            {"ticker": "SOL", "trend": "mixed/transitioning: no agreement", "rsi": 49.0},
        ]
        verdict = indicators.compare_setups(analyses)
        self.assertIn("transition", verdict)

    def test_level_proximity_picks_nearer_side(self):
        near_resistance = indicators.level_proximity(99.0, {"support": 80.0, "resistance": 100.0})
        self.assertEqual(near_resistance["level_type"], "resistance")
        self.assertGreater(near_resistance["distance_pct"], 0)
        near_support = indicators.level_proximity(82.0, {"support": 80.0, "resistance": 120.0})
        self.assertEqual(near_support["level_type"], "support")


class SignalRecordTests(unittest.TestCase):
    def _write_log(self, records):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "posts_log.jsonl")
        import json
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return path

    def test_scorecard_counts_wins_losses_and_best_worst(self):
        path = self._write_log([
            {"mode": "signal_open", "ticker": "BTC"},
            {"mode": "signal_close", "ticker": "BTC", "outcome": "target", "r_multiple": 3.2},
            {"mode": "market_map"},
            {"mode": "signal_close", "ticker": "ETH", "outcome": "stop", "r_multiple": -1.0},
            {"mode": "signal_close", "ticker": "SOL", "outcome": "target", "r_multiple": 4.5},
        ])
        card = signal_record.load_scorecard(open_count=2, path=path)
        self.assertEqual(card["closed_count"], 3)
        self.assertEqual(card["win_count"], 2)
        self.assertEqual(card["loss_count"], 1)
        self.assertAlmostEqual(card["win_rate_pct"], 200 / 3)
        self.assertAlmostEqual(card["avg_r"], (3.2 - 1.0 + 4.5) / 3)
        self.assertEqual(card["best"]["ticker"], "SOL")
        self.assertEqual(card["worst"]["ticker"], "ETH")
        self.assertEqual(card["open_count"], 2)

    def test_include_folds_in_an_unlogged_close(self):
        path = self._write_log([
            {"mode": "signal_close", "ticker": "BTC", "outcome": "target", "r_multiple": 3.0},
        ])
        card = signal_record.load_scorecard(
            open_count=0, path=path,
            include={"ticker": "ETH", "outcome": "stop", "r_multiple": -1.0},
        )
        self.assertEqual(card["closed_count"], 2)
        self.assertEqual(card["win_count"], 1)

    def test_missing_log_is_empty_record(self):
        card = signal_record.load_scorecard(open_count=3, path="/tmp/does_not_exist_xyz.jsonl")
        self.assertEqual(card["closed_count"], 0)
        self.assertIsNone(card["win_rate_pct"])
        self.assertIn("first hypothetical scenario", signal_record.format_record_line(card))


class SchedulingTests(unittest.TestCase):
    def test_universe_has_ten_unique_non_stablecoin_assets(self):
        tickers = [asset["ticker"] for asset in config.ASSETS]
        self.assertEqual(len(tickers), 10)
        self.assertEqual(len(set(tickers)), 10)
        self.assertFalse({"USDT", "USDC"} & set(tickers))

    def test_alert_detects_confirmed_break_beyond_zone(self):
        previous = {
            "levels": {"support": 90, "resistance": 110, "zone_width": 2},
            "trend": "bearish: example",
        }
        event = main._technical_event(previous, 113, "bearish: still bearish")
        self.assertEqual(event[0], 2)
        self.assertIn("above", event[1])

    def test_alert_ignores_price_inside_zone(self):
        previous = {
            "levels": {"support": 90, "resistance": 110, "zone_width": 2},
            "trend": "bullish: example",
        }
        self.assertIsNone(main._technical_event(previous, 105, "bullish: unchanged"))

    def test_cron_schedule_maps_to_expected_modes(self):
        expected = {
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
        self.assertEqual(main.CRON_MODE_MAP, expected)
        for cron, mode in expected.items():
            with patch.dict(os.environ, {"CRON_SCHEDULE": cron}):
                self.assertEqual(main._automatic_mode(), mode)

    def test_signal_direction_from_trend(self):
        self.assertEqual(main._signal_direction("bullish: fast EMA above slow EMA"), "long")
        self.assertEqual(main._signal_direction("bearish: fast EMA below slow EMA"), "short")
        self.assertIsNone(main._signal_direction("mixed/transitioning: no agreement"))

    def test_evaluate_signal_long_target_and_stop(self):
        signal = {"direction": "long", "entry": 100.0, "target": 110.0, "stop": 95.0}
        outcome, r_multiple = main._evaluate_signal(signal, 111.0)
        self.assertEqual(outcome, "target")
        self.assertAlmostEqual(r_multiple, 2.2)

        outcome, r_multiple = main._evaluate_signal(signal, 94.0)
        self.assertEqual(outcome, "stop")
        self.assertAlmostEqual(r_multiple, -1.2)

        self.assertIsNone(main._evaluate_signal(signal, 102.0))

    def test_evaluate_signal_short_target_and_stop(self):
        signal = {"direction": "short", "entry": 100.0, "target": 90.0, "stop": 105.0}
        outcome, r_multiple = main._evaluate_signal(signal, 89.0)
        self.assertEqual(outcome, "target")
        self.assertAlmostEqual(r_multiple, 2.2)

        outcome, r_multiple = main._evaluate_signal(signal, 106.0)
        self.assertEqual(outcome, "stop")
        self.assertAlmostEqual(r_multiple, -1.2)

        self.assertIsNone(main._evaluate_signal(signal, 98.0))


class NarrativeTests(unittest.TestCase):
    def test_formatting_removes_list_markers_and_adds_spacing(self):
        raw = "- 📉 Trend line\n• ⚡ Momentum line\n3. 📊 Volatility line"
        result = narrative_generator._format_for_telegram(raw)
        self.assertEqual(
            result,
            "📉 Trend line\n\n⚡ Momentum line\n\n📊 Volatility line",
        )

    def test_caption_keeps_disclaimer_inside_telegram_limit(self):
        long_body = "\n\n".join(f"📊 Line {i} " + "x" * 180 for i in range(10))
        result = narrative_generator._fit_caption(long_body)
        self.assertLessEqual(len(result), config.TELEGRAM_CAPTION_LIMIT)
        self.assertTrue(result.endswith(config.DISCLAIMER))

    def test_disclaimer_links_the_channel_name_not_a_warning(self):
        self.assertIn(config.CHANNELS[0]["url"], config.DISCLAIMER)
        self.assertIn(config.CHANNELS[0]["name"], config.DISCLAIMER)  # "To the Moon 🚀"
        self.assertNotIn("financial advice", config.DISCLAIMER)
        self.assertNotIn("Follow", config.DISCLAIMER)

    def test_farsi_disclaimer_links_the_farsi_channel(self):
        fa = config.DISCLAIMERS["fa"]
        self.assertIn("https://t.me/crypto_market_farsi", fa)
        self.assertIn("تحلیل بازار کریپتو", fa)

    def test_fit_caption_escapes_html_so_the_follow_link_still_parses(self):
        result = narrative_generator._fit_caption("📈 S&P 500 vs <BTC> is a common comparison")
        self.assertIn("S&amp;P 500", result)
        self.assertIn("&lt;BTC&gt;", result)
        # The disclaimer's own anchor tag must stay raw HTML, not get escaped.
        self.assertIn(f'<a href="{config.CHANNELS[0]["url"]}">', result)

    def test_empty_ai_response_uses_titled_factual_fallback(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
        )
        with patch.object(narrative_generator, "_client") as client:
            client.return_value.chat.completions.create.return_value = response
            result = narrative_generator._complete(
                "prompt",
                "context",
                "🚨 ATTENTION-GRABBING TITLE",
                "📊 Clear factual fallback",
            )
        self.assertTrue(result.startswith("🚨 ATTENTION-GRABBING TITLE"))
        self.assertIn("📊 Clear factual fallback", result)
        self.assertTrue(result.endswith(config.DISCLAIMER))

    def test_daily_pulse_falls_back_without_network(self):
        derivatives_rows = [
            {"ticker": "BTC", "funding_rate": 0.012, "open_interest": 6.8e9, "market": "Binance (Futures)"},
            {"ticker": "ETH", "funding_rate": -0.004, "open_interest": 4.5e9, "market": "Binance (Futures)"},
        ]
        trending = [{"name": "Some Coin", "symbol": "SC", "market_cap_rank": 120, "change_24h": 12.0}]
        with patch.object(narrative_generator, "_client", side_effect=RuntimeError("no network in tests")):
            result = narrative_generator.generate_daily_pulse(derivatives_rows, trending)
        self.assertIn("BTC", result)
        # The fallback must tell the story, not restate the exact figures the chart already shows.
        self.assertNotIn("+0.0120%", result)
        self.assertNotIn("$6,800,000,000", result)
        self.assertTrue(result.endswith(config.DISCLAIMER))

    def test_daily_pulse_farsi_fallback_uses_farsi_and_farsi_disclaimer(self):
        derivatives_rows = [
            {"ticker": "BTC", "funding_rate": 0.012, "open_interest": 6.8e9, "market": "Binance (Futures)"},
            {"ticker": "ETH", "funding_rate": -0.004, "open_interest": 4.5e9, "market": "Binance (Futures)"},
        ]
        trending = [{"name": "Some Coin", "symbol": "SC", "market_cap_rank": 120, "change_24h": 12.0}]
        with patch.object(narrative_generator, "_client", side_effect=RuntimeError("no network in tests")):
            result = narrative_generator.generate_daily_pulse(derivatives_rows, trending, language="fa")
        self.assertIn("BTC", result)                       # tickers stay Latin
        self.assertIn("فاندینگ", result)                    # body is Farsi
        self.assertTrue(result.endswith(config.DISCLAIMERS["fa"]))

    def test_signal_post_farsi_uses_farsi_signal_disclaimer(self):
        result = narrative_generator.generate_signal_post(
            "ETH", "1d", "long", 1800.0, 1900.0, 1700.0, "bullish: trend up", 55.0, language="fa"
        )
        self.assertIn("لانگ", result)
        self.assertIn("ETH", result)
        self.assertTrue(result.endswith(config.SIGNAL_DISCLAIMERS["fa"]))

    def test_followup_localizes_structured_event(self):
        event = {"side": "above", "level": 110.0, "price": 113.0}
        en = narrative_generator.generate_followup("BTC", "4h", event, language="en")
        fa = narrative_generator.generate_followup("BTC", "4h", event, language="fa")
        self.assertIn("resistance", en)
        self.assertIn("مقاومت", fa)
        self.assertTrue(fa.endswith(config.DISCLAIMERS["fa"]))

    def test_weekly_digest_falls_back_without_network(self):
        weekly_rows = [
            {"ticker": ticker, "price": 100 + i, "change_7d": i - 5}
            for i, ticker in enumerate(("BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "HYPE", "DOGE", "LEO", "ZEC"))
        ]
        scoreboard_rows = [
            {"ticker": row["ticker"], "rsi": 40 + i * 4, "atr_pct": 1 + i * 0.3}
            for i, row in enumerate(weekly_rows)
        ]
        dates = pd.date_range("2026-01-01", periods=7, freq="D")
        dominance_history = pd.Series(np.linspace(55, 57, 7), index=dates)
        feargreed_history = pd.Series(np.linspace(30, 45, 7), index=dates)
        total_mcap_history = pd.Series(np.linspace(2.1e12, 2.2e12, 7), index=dates)
        stablecoin_history = pd.Series(np.linspace(1.6e11, 1.62e11, 7), index=dates)
        with patch.object(narrative_generator, "_client", side_effect=RuntimeError("no network in tests")):
            result = narrative_generator.generate_weekly_digest(
                weekly_rows, scoreboard_rows, dominance_history, feargreed_history,
                total_mcap_history, stablecoin_history,
            )
        self.assertIn("WEEKLY DIGEST", result)
        self.assertTrue(result.endswith(config.DISCLAIMER))

    def test_signal_post_labels_direction_and_hypothetical_disclaimer(self):
        result = narrative_generator.generate_signal_post(
            "ETH", "1d", "long", 1800.0, 1900.0, 1700.0, "bullish: trend up", 55.0
        )
        self.assertIn("LONG", result)
        self.assertIn("ETH", result)
        self.assertIn("Hypothetical", result)
        self.assertTrue(result.endswith(config.SIGNAL_DISCLAIMER))

    def test_signal_outcome_reports_result_and_disclaimer(self):
        result = narrative_generator.generate_signal_outcome(
            "ETH", "1d", "long", 1800.0, 1900.0, "target", 2.0, "2026-01-01T00:00:00"
        )
        self.assertIn("TARGET HIT", result)
        self.assertIn("+2.00R", result)
        self.assertTrue(result.endswith(config.SIGNAL_DISCLAIMER))

    def test_signal_outcome_appends_record_line(self):
        result = narrative_generator.generate_signal_outcome(
            "ETH", "1d", "long", 1800.0, 1900.0, "target", 2.0, "2026-01-01T00:00:00",
            record_line="📊 Track record so far: 3 scenarios closed",
        )
        self.assertIn("Track record so far", result)

    def test_signal_scorecard_falls_back_and_teaches(self):
        scorecard = {
            "closed_count": 3, "win_count": 2, "loss_count": 1, "win_rate_pct": 66.7,
            "avg_r": 1.2, "total_r": 3.6, "open_count": 4,
            "best": {"ticker": "SOL", "r_multiple": 4.5},
            "worst": {"ticker": "ETH", "r_multiple": -1.0},
        }
        with patch.object(narrative_generator, "_client", side_effect=RuntimeError("no network in tests")):
            result = narrative_generator.generate_signal_scorecard(scorecard)
        self.assertIn("SCORECARD", result)
        self.assertTrue(result.endswith(config.SIGNAL_DISCLAIMER))

    def test_signal_scorecard_empty_state(self):
        scorecard = {
            "closed_count": 0, "win_count": 0, "loss_count": 0, "win_rate_pct": None,
            "avg_r": 0.0, "total_r": 0.0, "open_count": 2, "best": None, "worst": None,
        }
        with patch.object(narrative_generator, "_client", side_effect=RuntimeError("no network in tests")):
            result = narrative_generator.generate_signal_scorecard(scorecard)
        self.assertIn("SCORECARD", result)
        self.assertTrue(result.endswith(config.SIGNAL_DISCLAIMER))

    def test_what_to_watch_falls_back_without_network(self):
        watch_rows = [
            {"ticker": "BTC", "level_type": "resistance", "distance_pct": 2.1, "rsi": 68.0,
             "trend": "bullish: aligned", "price": 65000.0},
            {"ticker": "SOL", "level_type": "support", "distance_pct": 1.4, "rsi": 32.0,
             "trend": "bearish: aligned", "price": 76.0},
        ]
        backdrop = {"dominance_note": "BTC dominance has been rising",
                    "fear_greed": {"value": 30, "classification": "Fear"}}
        with patch.object(narrative_generator, "_client", side_effect=RuntimeError("no network in tests")):
            result = narrative_generator.generate_what_to_watch(watch_rows, backdrop)
        self.assertIn("WHAT TO WATCH", result)
        self.assertIn("BTC", result)
        self.assertTrue(result.endswith(config.DISCLAIMER))


class ChartTests(unittest.TestCase):
    def test_technical_headline_describes_structure(self):
        frame = synthetic_market()
        levels = indicators.find_key_levels(frame, lookback=180)
        headline = chart_generator._technical_headline(frame.iloc[-1], levels)
        self.assertIn(
            headline,
            {
                "CONFIRMED RESISTANCE BREAK",
                "CONFIRMED SUPPORT BREAK",
                "BULLISH MOMENTUM BUILDS",
                "BEARS STILL CONTROL THE TREND",
                "PRICE ENTERS A DECISION ZONE",
            },
        )

    def test_chart_is_wide_and_renderable(self):
        frame = synthetic_market()
        levels = indicators.find_key_levels(frame, lookback=180)
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_chart(frame, "ETHUSDT", "1d", levels)
            image = mpimg.imread(path)

        self.assertTrue(os.path.basename(path).startswith("ETHUSDT_1d"))
        self.assertGreater(image.shape[1] / image.shape[0], 1.4)

    def test_signal_chart_renders_entry_target_stop_labels(self):
        frame = synthetic_market()
        levels = indicators.find_key_levels(frame, lookback=180)
        signal = {"direction": "long", "entry": 1800.0, "target": 1900.0, "stop": 1700.0}
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_chart(frame, "ETHUSDT", "1d", levels, signal=signal)
            image = mpimg.imread(path)

        self.assertTrue(os.path.basename(path).startswith("ETHUSDT_1d_signal"))
        self.assertGreater(image.shape[1] / image.shape[0], 1.4)

    def test_coin_gecko_chart_renders_without_volume(self):
        frame = synthetic_market()
        frame["Volume"] = 0
        frame.attrs["source"] = "CoinGecko"
        levels = indicators.find_key_levels(frame, lookback=180)
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_chart(frame, "LEOUSDT", "1d", levels)
            image = mpimg.imread(path)
        self.assertGreater(image.shape[1] / image.shape[0], 1.4)

    def test_market_map_and_macro_charts_render(self):
        snapshot = [
            {"ticker": ticker, "price": 100 + index, "change_24h": index - 5, "change_7d": index - 5}
            for index, ticker in enumerate(("BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "HYPE", "DOGE", "LEO", "ZEC"))
        ]
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        macro = {
            "sp500": pd.Series(np.linspace(6000, 6200, 30), index=dates),
            "dollar": pd.Series(np.linspace(120, 118, 30), index=dates),
            "btc_dominance": 57.25,
        }
        macro_extended = {
            **macro,
            "vix": 18.4,
            "yield_10y": 4.2,
            "yield_2y": 3.9,
            "btc_sp_corr": 0.35,
            "btc_dollar_corr": -0.22,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            market_path = chart_generator.generate_market_map_chart(snapshot)
            weekly_path = chart_generator.generate_weekly_recap_chart(snapshot)
            macro_path = chart_generator.generate_macro_chart(macro)
            macro_extended_path = chart_generator.generate_macro_chart(macro_extended)
            market_image = mpimg.imread(market_path)
            weekly_image = mpimg.imread(weekly_path)
            macro_image = mpimg.imread(macro_path)
            macro_extended_image = mpimg.imread(macro_extended_path)
        self.assertGreater(market_image.shape[1] / market_image.shape[0], 1.4)
        self.assertGreater(weekly_image.shape[1] / weekly_image.shape[0], 1.4)
        self.assertGreater(macro_image.shape[1] / macro_image.shape[0], 1.4)
        self.assertGreater(macro_extended_image.shape[1] / macro_extended_image.shape[0], 1.4)

    def test_technical_scoreboard_chart_renders(self):
        rows = [
            {"ticker": ticker, "rsi": rsi, "atr_pct": atr}
            for ticker, rsi, atr in zip(
                ("BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "HYPE", "DOGE", "LEO", "ZEC"),
                (82, 74, 65, 55, 45, 35, 25, 18, 60, 50),
                (1.1, 2.3, 0.8, 3.4, 1.9, 0.5, 4.1, 2.0, 1.2, 0.9),
            )
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_technical_scoreboard_chart(rows)
            image = mpimg.imread(path)
        self.assertGreater(image.shape[1] / image.shape[0], 1.2)

    def test_market_structure_and_liquidity_charts_render(self):
        dates = pd.date_range("2026-01-01", periods=14, freq="D")
        dominance_history = pd.Series(np.linspace(55, 58, 14), index=dates)
        feargreed_history = pd.Series(np.linspace(30, 60, 30), index=pd.date_range("2026-01-01", periods=30, freq="D"))
        total_mcap_history = pd.Series(np.linspace(2.1e12, 2.3e12, 14), index=dates)
        stablecoin_history = pd.Series(np.linspace(1.6e11, 1.65e11, 14), index=dates)
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            structure_path = chart_generator.generate_market_structure_chart(dominance_history, feargreed_history)
            liquidity_path = chart_generator.generate_liquidity_chart(total_mcap_history, stablecoin_history)
            structure_image = mpimg.imread(structure_path)
            liquidity_image = mpimg.imread(liquidity_path)
        self.assertGreater(structure_image.shape[1] / structure_image.shape[0], 1.4)
        self.assertGreater(liquidity_image.shape[1] / liquidity_image.shape[0], 1.4)

    def test_signal_scorecard_chart_renders_with_closed_signals(self):
        scorecard = {
            "closed_count": 3, "win_count": 2, "loss_count": 1, "win_rate_pct": 66.7,
            "avg_r": 1.2, "total_r": 3.6, "open_count": 4,
            "best": {"ticker": "SOL", "r_multiple": 4.5},
            "worst": {"ticker": "ETH", "r_multiple": -1.0},
        }
        closed = [
            {"ticker": "BTC", "outcome": "target", "r_multiple": 3.2},
            {"ticker": "ETH", "outcome": "stop", "r_multiple": -1.0},
            {"ticker": "SOL", "outcome": "target", "r_multiple": 4.5},
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_signal_scorecard_chart(scorecard, closed)
            image = mpimg.imread(path)
        self.assertTrue(os.path.basename(path).startswith("signal_scorecard"))
        self.assertGreater(image.shape[1] / image.shape[0], 1.2)

    def test_signal_scorecard_chart_renders_empty_state(self):
        scorecard = {
            "closed_count": 0, "win_count": 0, "loss_count": 0, "win_rate_pct": None,
            "avg_r": 0.0, "total_r": 0.0, "open_count": 2, "best": None, "worst": None,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_signal_scorecard_chart(scorecard, [])
            image = mpimg.imread(path)
        self.assertGreater(image.shape[1] / image.shape[0], 1.2)

    def test_watchlist_chart_renders(self):
        rows = [
            {"ticker": "BTC", "level_type": "resistance", "distance_pct": 2.1, "rsi": 68.0,
             "trend": "bullish: aligned", "price": 65000.0},
            {"ticker": "SOL", "level_type": "support", "distance_pct": 1.4, "rsi": 32.0,
             "trend": "bearish: aligned", "price": 76.0},
            {"ticker": "ETH", "level_type": "resistance", "distance_pct": 3.8, "rsi": 55.0,
             "trend": "mixed/transitioning: no agreement", "price": 1900.0},
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_watchlist_chart(rows)
            image = mpimg.imread(path)
        self.assertTrue(os.path.basename(path).startswith("watchlist"))
        self.assertGreater(image.shape[1] / image.shape[0], 1.2)

    def test_pulse_chart_renders(self):
        derivatives_rows = [
            {"ticker": ticker, "funding_rate": rate, "open_interest": oi, "market": "Binance (Futures)"}
            for ticker, rate, oi in zip(
                ("BTC", "ETH", "SOL"), (0.012, -0.004, 0.021), (6.8e9, 4.5e9, 1.2e9)
            )
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "CHART_DIR", directory):
            path = chart_generator.generate_pulse_chart(derivatives_rows)
            image = mpimg.imread(path)
        self.assertGreater(image.shape[1] / image.shape[0], 1.2)


if __name__ == "__main__":
    unittest.main()
