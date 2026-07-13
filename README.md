# Crypto Market Update Channel

Posts an automated, educational market overview for BTC and ETH (4h and 1D
timeframes) to a Telegram channel — real charts from real data, plus a short
AI-written narrative. Runs on a GitHub Actions cron, same pattern as the
Laplace Apply pipeline.

## What it posts

- A candlestick chart (Binance data) with EMA20/EMA50 and a recent
  support/resistance zone drawn on it
- A short DeepSeek-written overview: trend, momentum (RSI/MACD), volatility
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

4. **Add repo secrets** — Settings → Secrets and variables → Actions:
   - `DEEPSEEK_KEY`
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHANNEL` (e.g. `@your_channel` or the numeric chat ID)

5. **Test manually** before waiting for the cron: Actions tab → "Crypto
   Market Update" → "Run workflow".

6. It then runs automatically every 4 hours (`.github/workflows/post_update.yml`).

## Local testing

```bash
cp .env.example .env      # fill in real values
pip install -r requirements.txt
python -m dotenv run python main.py   # or just export the vars manually
```

## Adjusting things

Everything tunable lives at the top of `config.py`: symbols, timeframes,
indicator periods, the follow-up threshold, chart styling colors. No need to
touch the other files for day-to-day changes.

To add a symbol: add it to `SYMBOLS` in `config.py` (must be a valid Binance
pair, e.g. `"SOLUSDT"`). To add a timeframe: add it to `TIMEFRAMES` (valid
Binance intervals: `1m 5m 15m 1h 4h 1d 1w` etc.).

## File overview

| File | Purpose |
|---|---|
| `config.py` | All constants — edit this first |
| `data_fetcher.py` | Binance OHLC data + Fear & Greed index |
| `indicators.py` | RSI/EMA/MACD/Bollinger/ATR, computed manually with pandas (not `pandas_ta`, which currently breaks on numpy ≥2.0) |
| `chart_generator.py` | Renders the real candlestick chart via `mplfinance` |
| `narrative_generator.py` | DeepSeek call for the write-up + the follow-up template |
| `telegram_publisher.py` | Posts photo+caption, and reply-to-message for follow-ups |
| `state_manager.py` | Persists last-posted levels/message IDs to `post_history.json` |
| `main.py` | Orchestrates all of the above |

## X (Twitter) — not yet included

You mentioned cross-posting to X later — holding off on that until this is
running well on Telegram. It's a separate publisher module following the
same pattern as `telegram_publisher.py`, so it's a small add whenever you're
ready.
