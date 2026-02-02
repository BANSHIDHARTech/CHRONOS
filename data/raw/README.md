# Raw Data Directory

> Cached market data from yfinance

## Overview

This directory stores raw OHLCV (Open, High, Low, Close, Volume) data downloaded from Yahoo Finance via the `yfinance` library.

## Expected Files

After running the data pipeline, you should see:

| File | Description | Size |
|------|-------------|------|
| `GSPC_2019-01-01_2024-12-31.csv` | S&P 500 Index | ~150KB |
| `VIX_2019-01-01_2024-12-31.csv` | CBOE Volatility Index | ~150KB |
| `TNX_2019-01-01_2024-12-31.csv` | 10-Year Treasury Yield | ~150KB |
| `GC=F_2019-01-01_2024-12-31.csv` | Gold Futures | ~150KB |
| `SPY_2019-01-01_2024-12-31.csv` | S&P 500 ETF | ~150KB |
| `TLT_2019-01-01_2024-12-31.csv` | 20+ Year Treasury ETF | ~150KB |
| `GLD_2019-01-01_2024-12-31.csv` | Gold ETF | ~150KB |

## Downloading Data

Data is automatically downloaded on first pipeline run:

```powershell
python -m src.data.data_loader
```

Or as part of the full pipeline:

```powershell
python -m src.data.pipeline
```

## Not Committed to Version Control

This directory is **ignored by git** because:

1. Files are auto-generated from public yfinance API
2. Data changes as markets update
3. Reduces repository size
4. Ensures users get fresh data

## Fallback: Manual Download

If the yfinance API fails (rate limits, network issues):

1. Go to [Yahoo Finance](https://finance.yahoo.com/)
2. Search for each ticker (e.g., ^GSPC for S&P 500)
3. Click "Historical Data"
4. Set date range: 2019-01-01 to 2024-12-31
5. Download CSV
6. Rename to format: `{TICKER}_{START}_{END}.csv`
7. Place in this directory

### Ticker Mapping

| Symbol | Yahoo Finance Ticker |
|--------|---------------------|
| S&P 500 | ^GSPC |
| VIX | ^VIX |
| 10-Year Treasury | ^TNX |
| Gold Futures | GC=F |
| SPY ETF | SPY |
| TLT ETF | TLT |
| GLD ETF | GLD |

## Cache Behavior

- Data is cached after first download
- Use `--no-cache` flag to force re-download:

```powershell
python -m src.data.pipeline --no-cache
```

## Troubleshooting

### "yfinance download failed"

1. Check internet connection
2. Try again (API may be rate-limited)
3. Use `--no-cache` to retry fresh
4. Fall back to manual download

### "Data file not found"

Ensure you've run the data loader:
```powershell
python -m src.data.data_loader
```

### "Date range mismatch"

Update `config.py` dates to match your needs, then re-download.

---

*This directory is generated - do not add files manually.*
