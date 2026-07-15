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
            "15 20 * * 1-5": "macro_close",
            "15 21 * * 1-5": "macro_close",
        }
        self.assertEqual(main.CRON_MODE_MAP, expected)
        for cron, mode in expected.items():
            with patch.dict(os.environ, {"CRON_SCHEDULE": cron}):
                self.assertEqual(main._automatic_mode(), mode)


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
        self.assertIn("BTC funding", result)
        self.assertTrue(result.endswith(config.DISCLAIMER))

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
