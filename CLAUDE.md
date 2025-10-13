# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Portfolio analysis tool for evaluating downside protection strategies. Analyzes a portfolio's performance against 4 major market benchmarks (SPY, QQQ, DIA, IWM) to measure downside/upside capture ratios, hit rates on down days, and comprehensive risk metrics.

**Goal:** Protect downside while participating in upside - measure how well the portfolio achieves this.

## Two-Script Architecture

The codebase is split into **analysis** (data) and **visualization** (presentation):

### 1. analyze.py - Data Pipeline
- Fetches historical prices via yfinance (3-year default lookback)
- Calculates portfolio returns from weighted holdings
- Computes capture ratios, hit rates, drawdown metrics
- Outputs CSV files to `html/` directory
- **Does NOT generate charts or HTML** - that's visualize.py's job

Key outputs:
- `capture_stats.csv` - Downside/upside capture ratios vs benchmarks
- `summary_stats.csv` - Performance by market condition (up/down days)
- `cumulative_returns.csv` - Time series of returns
- `weights_used.csv` - Final normalized weights after filtering
- `dropped_tickers.txt` - Tickers excluded (download failures, low coverage)

### 2. visualize.py - Presentation Layer
- Reads CSV files from analyze.py (does NOT fetch data)
- Generates 7 matplotlib charts as PNG files
- Builds HTML dashboard with embedded charts and metrics tables
- Can be run multiple times without re-fetching data (for styling iterations)

**Workflow:** Always run `analyze.py` first, then `visualize.py`. The separation allows iterating on visualizations without expensive data re-fetching.

## Running the Analysis

```bash
# Setup (once)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full workflow
python analyze.py        # Fetches data, calculates metrics (~2-5 minutes)
python visualize.py      # Generates charts and HTML dashboard (~10 seconds)
open html/index.html     # View results
```

## Key Configuration (analyze.py)

```python
START_DATE / END_DATE     # Analysis period (default: 3 years)
BENCH                     # ["SPY", "QQQ", "DIA", "IWM"]
MIN_COVERAGE = 0.90       # 90% data coverage requirement
ALIASES = {}              # Ticker alias mapping (Yahoo Finance variants)
DROP_SET                  # Non-equity identifiers to filter out
```

## Input File: weights.csv

Portfolio holdings with normalized weights:
```csv
Symbol,Weight
PHYS,0.1499
CEF,0.0789
...
```

**Important:** Weights must sum to ~1.0. The script auto-normalizes after filtering.

Column detection is flexible - supports `Symbol`/`Ticker` and `Weight`/`Pct`/`Percent`.

## Data Quality & Filtering

The analysis has robust filtering to handle real-world data issues:

1. **Ticker Sanitization**: Filters out cash/sweep accounts (`DROP_SET`)
2. **Download Failures**: Reports in `dropped_tickers.txt`, continues without them
3. **Coverage Threshold**: Requires 90% overlap with benchmark calendar (handles IPOs, delistings)
4. **Share Class Fallback**: Auto-tries BRK.B → BRK-B variants
5. **NaN Handling**: Drops tickers with >5% missing data after alignment
6. **Benchmark Calendar**: Uses intersection of all 4 benchmark dates as baseline

Result: Clean, aligned daily returns for all portfolio positions + benchmarks.

## Key Metrics Explained

### Downside/Upside Capture Ratio
- **Downside Capture**: Portfolio loss / Benchmark loss on down days (lower is better)
  - Example: 62.6% = portfolio loses only 62.6% of SPY's losses
- **Upside Capture**: Portfolio gain / Benchmark gain on up days (higher is better)
  - Example: 82.7% = portfolio captures 82.7% of SPY's gains

### Hit Rate
Percentage of days portfolio is non-negative when benchmark is down.
- Example: 37.5% on SPY-down days = portfolio stayed flat or positive 37.5% of the time

### Benchmark Conditions
- `{BENCH}_down_days`: Days when benchmark return < 0
- `{BENCH}_up_or_flat_days`: Days when benchmark return >= 0
- `Both_down_days`: SPY AND QQQ both down (severe market stress)

## Output Structure

```
html/
├── index.html              # Main dashboard (open this)
├── capture_stats.csv
├── summary_stats.csv
├── cumulative_returns.csv
├── weights_used.csv
├── dropped_tickers.txt
└── charts/
    ├── 1_cumulative_returns.png
    ├── 2_capture_ratios.png
    ├── 3_hit_rates.png
    ├── 4_return_distributions.png
    ├── 5_avg_returns_up_down.png
    ├── 6_drawdown.png
    └── 7_portfolio_composition.png
```

## Color Scheme (Consistent Across Charts)

```python
Portfolio: #2E86AB (Blue)
SPY:       #A23B72 (Purple)
QQQ:       #F18F01 (Orange)
DIA:       #06A77D (Green)
IWM:       #9D4EDD (Violet)
```

## Common Modifications

### Changing Analysis Period
Edit `analyze.py`:
```python
START_DATE = date(2020, 1, 1)  # Fixed start date
END_DATE = date.today()         # Or fixed end date
```

### Adding/Removing Benchmarks
Edit `BENCH` list in `analyze.py`:
```python
BENCH = ["SPY", "QQQ", "DIA", "IWM", "VTI"]  # Add VTI
```
**Note:** This changes metrics calculations. Both scripts reference these benchmarks.

### Ticker Aliases (Yahoo Finance variants)
Some tickers need aliases for Yahoo Finance. Edit `ALIASES` in `analyze.py`:
```python
ALIASES = {
    "BRK.B": "BRK-B",      # Berkshire B shares
    "BF.B": "BF-B",        # Brown-Forman B shares
}
```

### Adjusting Coverage Threshold
If many tickers are dropped for low coverage:
```python
MIN_COVERAGE = 0.80  # Reduce from 0.90 to 0.80
```
Trade-off: More holdings included vs. data quality.

## Gotchas & Known Issues

1. **matplotlib not installed**: `visualize.py` will fail gracefully with clear error
2. **Ticker download failures**: Common for international/OTC stocks - check `dropped_tickers.txt`
3. **Share class tickers**: Use `.` (BRK.B) in weights.csv, fallback logic auto-converts to `-`
4. **Cash/sweep positions**: Filtered out automatically (can't download prices)
5. **Recent IPOs**: May fail coverage check if less than 90% of 3-year history
6. **Delisted stocks**: Will fail downloads - remove from weights.csv
7. **Git ignores html/**: Output directory is gitignored (regenerated each run)

## When to Re-run

- **analyze.py**: When data is stale (daily/weekly) OR weights changed OR config changed
- **visualize.py**: When tweaking chart styles/colors OR HTML layout (no data refresh needed)

## Dependencies

```
yfinance   # Yahoo Finance API for historical prices
pandas     # Data processing and alignment
matplotlib # Chart generation (visualize.py only)
```

No API keys required. Uses public Yahoo Finance data.
