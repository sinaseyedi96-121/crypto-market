# Crypto Market Update Channel

Posts a balanced, automated market publication to Telegram. It covers the top
ten non-stablecoin assets, BTC dominance, the S&P 500, the Federal Reserve
broad U.S. dollar index, derivatives positioning, and crypto-wide liquidity.

## What it posts

- A daily market-map chart covering all ten assets in one post
- A daily derivatives pulse: perpetual funding rates and open interest for
  BTC/ETH/SOL, plus CoinGecko's trending coins
- A rotating two-asset deep-dive album using confirmed closed candles,
  with EMA20/EMA50, Bollinger Bands, volume, and long-history pivot levels
- A weekday macro-close chart for the S&P 500, Fed broad dollar index, current
  BTC dominance, VIX, the 10Y-2Y Treasury spread, and BTC's rolling
  correlation with the S&P 500 and the dollar index
- A Sunday weekly digest album: a 7-day performance ranking, an RSI/ATR%
  technical scoreboard across the universe, a BTC-dominance + Fear & Greed
  trend chart, and a total/stablecoin market-cap trend chart
- A spaced DeepSeek V4 Pro overview: trend, momentum (RSI/MACD), volatility
  (ATR/Bollinger), and the Fear & Greed Index
- If price later breaks a zone flagged in an earlier post, a factual
  follow-up reply is posted under that original message

## Goals

All goal of these posts are educational only, and not financial advice.

What you get instead is the same underlying analysis (support/resistance
zones, indicators, real charts) framed as descriptive commentary rather than
instructions — genuinely useful, without the regulatory and credibility-fraud
risk. If you want to revisit this later, worth a proper conversation with
someone who knows financial-services regulation first.

## Setup

1. **Create the Telegram bot** (if you don't already have one for this) via
   [@BotFather](https://t.me/BotFather), and add it as admin to your channel.

2. **Get a DeepSeek API key** at [platform.deepseek.com](https://platform.deepseek.com).

3. **Push this to a new GitHub repo.**

4. **Add required repo secrets** — Settings → Secrets and variables → Actions:
   - `DEEPSEEK_KEY`
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHANNEL` (e.g. `@your_channel` or the numeric chat ID)

5. **Optional data key:** `COINGECKO_API_KEY` — a free Demo API key is
   recommended for reliable market-map and fallback data.

6. **Test manually** before waiting for the cron: Actions tab → "Crypto
   Market Update" → "Run workflow". Select one slot first; use `all` only when
   you intentionally want all three publications.

7. The workflow then publishes automatically (`.github/workflows/post_update.yml`),
   spaced roughly across the day so the channel stays active on weekdays and
   weekends alike:
   - 04:15 UTC daily — market map (also records the day's total market cap,
     BTC dominance, and stablecoin market cap for the weekly digest)
   - 08:15 UTC daily — derivatives pulse (funding rates, open interest,
     trending coins)
   - 12:15 UTC daily — rotating two-asset deep dive
   - 16:15 UTC Sundays — weekly digest
   - 20:15 and 21:15 UTC weekdays — daylight-safe macro checks; only the run
     after 16:10 New York time can publish, and the second is deduplicated
   - Every four hours at :10 — a silent confirmed-candle scan. It publishes
     only a support/resistance break or bullish/bearish regime change, capped
     at two alerts per day

   The paired macro checks keep that post close to 22:15 Rome across daylight-
   saving changes. Each slot is recorded in `post_history.json` (daily slots
   by date, the weekly digest by ISO week), so workflow retries and the
   second macro check do not duplicate posts.

## Local testing

```bash
cp .env.example .env      # fill in real values
pip install -r requirements.txt
python -m dotenv run python main.py --mode market_map
python -m dotenv run python main.py --mode daily_pulse
python -m dotenv run python main.py --mode deep_dive
python -m dotenv run python main.py --mode macro_close
python -m dotenv run python main.py --mode weekly_digest --force
```

The weekly digest needs at least one prior `market_map` run to have recorded
a market-structure snapshot (`--force` skips the Sunday-only day check, not
that requirement) — run `market_map` first when testing locally.

## Adjusting things

Everything tunable lives at the top of `config.py`: assets, weekly rotation,
timeframe, indicator periods, the follow-up threshold, chart settings, the
daily-pulse asset list (`PULSE_ASSETS`), and the weekly digest's day of week
(`WEEKLY_DIGEST_WEEKDAY`).

The configured universe is BTC, ETH, BNB, XRP, SOL, TRX, HYPE, DOGE, LEO, and
ZEC. Recheck the market-cap ranking monthly rather than changing the editorial
universe every day. Each asset has a Binance pair and a CoinGecko ID; unsupported
Binance pairs automatically fall back to free hourly CoinGecko history that is
aggregated locally into confirmed 4-hour or daily OHLC candles.

The dollar chart deliberately uses FRED's broad U.S. dollar index, not ICE DXY.
That keeps the source free and correctly labelled.

## File overview

| File | Purpose |
|---|---|
| `config.py` | All constants — edit this first |
| `data_fetcher.py` | Binance/CoinGecko OHLC, CoinGecko global/stablecoin/trending/derivatives data, FRED macro data, and sentiment |
| `indicators.py` | RSI/EMA/MACD/Bollinger/ATR/rolling correlation, computed manually with pandas (not `pandas_ta`, which currently breaks on numpy ≥2.0) |
| `chart_generator.py` | Renders every chart via `mplfinance`/`matplotlib` — candlesticks, rankings, scoreboards, and trend lines |
| `narrative_generator.py` | DeepSeek calls for each post's write-up + the follow-up template |
| `telegram_publisher.py` | Posts photo+caption, and reply-to-message for follow-ups |
| `state_manager.py` | Persists last-posted levels/message IDs to `post_history.json` |
| `main.py` | Orchestrates all of the above |

## Posting volume

The normal cadence is five guaranteed posts on weekdays (market map, daily
pulse, deep dive, macro close) plus a weekly digest on Sunday, and four on the
rest of the weekend — plus up to two more opportunistic alert posts on any
day when a confirmed technical event happens. Ten separate asset posts are
intentionally avoided: the market map batches breadth into one chart, the
deep dive uses one two-chart album, and the weekly digest batches four charts
into one album. Publishing is Telegram-only; no X, Bluesky, Discord, or other
social integration is active.
