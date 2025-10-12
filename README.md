# Portfolio Analysis - Downside Protection Strategy

A Python-based portfolio analysis tool designed to evaluate downside protection while maintaining upside participation across multiple market benchmarks.

## Overview

This tool analyzes a portfolio's performance against major market indices (S&P 500, Nasdaq 100, DJIA, Russell 2000) to measure:
- **Downside capture**: How much of benchmark losses the portfolio experiences
- **Upside capture**: How much of benchmark gains the portfolio captures
- **Hit rates**: Frequency of positive returns when benchmarks are down
- **Performance statistics**: Daily, cumulative, and comparative metrics

## Features

- Fetches historical price data via yfinance
- Handles ticker fallbacks and aliases
- Calculates capture ratios vs 4 major benchmarks
- Generates day-by-day analysis for up/down market days
- Exports comprehensive CSV reports

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Create a `weights.csv` file with your portfolio:
```csv
Symbol,Weight
PHYS,0.1499
CEF,0.0789
TPL,0.0782
...
```

2. Run the analysis:
```bash
python portfolio_analysis.py
```

3. Results are saved to the `output/` directory:
   - `capture_stats.csv` - Upside/downside capture ratios
   - `summary_stats.csv` - Performance by market conditions
   - `day_by_day_*.csv` - Daily return breakdowns
   - `cumulative_returns.csv` - Cumulative performance vs benchmarks

## Configuration

Edit `portfolio_analysis.py` to customize:
- `START_DATE` / `END_DATE` - Analysis period (default: 3 years)
- `BENCH` - Benchmark tickers (default: SPY, QQQ, DIA, IWM)
- `MIN_COVERAGE` - Required data coverage (default: 90%)
- `ALIASES` - Ticker aliases for special cases

## Key Metrics

### Downside Capture
Measures portfolio loss as a percentage of benchmark loss on down days. Lower is better.
- Example: 62.6% vs SPY = portfolio loses only 62.6% of SPY's losses

### Upside Capture
Measures portfolio gain as a percentage of benchmark gain on up days. Higher is better.
- Example: 82.7% vs SPY = portfolio captures 82.7% of SPY's gains

### Hit Rate
Percentage of days the portfolio is non-negative when benchmark is down.
- Example: 37.5% on SPY-down days = portfolio is flat or positive 37.5% of the time

## Output Files

| File | Description |
|------|-------------|
| `capture_stats.csv` | Downside/upside capture ratios and hit rates |
| `summary_stats.csv` | Aggregated statistics by market condition |
| `day_by_day_all.csv` | Complete daily returns for portfolio and benchmarks |
| `day_by_day_*_down.csv` | Returns filtered by benchmark down days |
| `cumulative_returns.csv` | Cumulative performance over time |
| `weights_used.csv` | Actual weights used after filtering |
| `coverage_report.csv` | Data coverage for each ticker |
| `dropped_tickers.txt` | Tickers excluded from analysis |

## Requirements

- Python 3.7+
- yfinance
- pandas

## License

Private repository for personal use.
