# Crypto Market Update Channel

Posts a balanced, automated market publication to Telegram. It covers the top
ten non-stablecoin assets, BTC dominance, the S&P 500, and the Federal Reserve
broad U.S. dollar index.

## What it posts

- A daily market-map chart covering all ten assets in one post
- A rotating two-asset deep-dive album using confirmed closed candles,
  with EMA20/EMA50, Bollinger Bands, volume, and long-history pivot levels
- A weekday macro-close chart for the S&P 500, Fed broad dollar index, and
  current BTC dominance
- A spaced DeepSeek V4 Pro overview: trend, momentum (RSI/MACD), volatility
  (ATR/Bollinger), and the Fear & Greed Index
- If price later breaks a zone flagged in an earlier post, a factual
  follow-up reply is posted under that original message

## What it deliberately does NOT post

No entry price, take-profit, or stop-loss, and no "signal hit" tracking.

Reasoning: publishing specific trade levels to the public, with a mechanism
that shows when they "worked," is what actually defines an investment
recommendation / signal service under EU rules (MiFID II; CONSOB in Italy
specifically) — regardless of an "educational only" disclaimer attached to
it. It's also the exact pattern used by non-legitimate signal sellers to
manufacture a track record. The disclaimer alone doesn't change that.

What you get instead is the same underlying analysis (support/resistance
zones, indicators, real charts) framed as descriptive commentary rather than
instructions — genuinely useful, without the regulatory and credibility-fraud
risk. If you want to revisit this later, worth a proper conversation with
someone who knows Italian/EU financial-services regulation first.

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

7. The workflow then publishes automatically (`.github/workflows/post_update.yml`):
   - 06:15 UTC daily — market map
   - 12:00 UTC daily — rotating two-asset deep dive
   - 20:15 and 21:15 UTC weekdays — daylight-safe macro checks; only the run
     after 16:10 New York time can publish, and the second is deduplicated
   - Every four hours at :10 — a silent confirmed-candle scan. It publishes
     only a support/resistance break or bullish/bearish regime change, capped
     at two alerts per day

   The paired macro checks keep that post close to 22:15 Rome across daylight-
   saving changes. Each slot is recorded in `post_history.json`, so workflow
   retries and the second macro check do not duplicate posts.

## Local testing

```bash
cp .env.example .env      # fill in real values
pip install -r requirements.txt
python -m dotenv run python main.py --mode market_map
python -m dotenv run python main.py --mode deep_dive
python -m dotenv run python main.py --mode macro_close
```

## Adjusting things

Everything tunable lives at the top of `config.py`: assets, weekly rotation,
timeframe, indicator periods, the follow-up threshold, and chart settings.

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
| `data_fetcher.py` | Binance/CoinGecko OHLC, CoinGecko global data, FRED macro data, and sentiment |
| `indicators.py` | RSI/EMA/MACD/Bollinger/ATR, computed manually with pandas (not `pandas_ta`, which currently breaks on numpy ≥2.0) |
| `chart_generator.py` | Renders the real candlestick chart via `mplfinance` |
| `narrative_generator.py` | DeepSeek call for the write-up + the follow-up template |
| `telegram_publisher.py` | Posts photo+caption, and reply-to-message for follow-ups |
| `state_manager.py` | Persists last-posted levels/message IDs to `post_history.json` |
| `main.py` | Orchestrates all of the above |

## Posting volume

The normal cadence is three posts on weekdays and two on weekends. Ten separate
asset posts are intentionally avoided: the market map batches breadth into one
chart, and the deep dive uses one two-chart album. Publishing is Telegram-only;
no X, Bluesky, Discord, or other social integration is active.
