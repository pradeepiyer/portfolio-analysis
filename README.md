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

3. **Open the HTML dashboard:**
```bash
open html/index.html
```

The dashboard provides a comprehensive view of all metrics and visualizations in one place.

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

All results are saved to the `html/` directory:

### 📊 HTML Dashboard (PRIMARY OUTPUT)

**`html/index.html`** - Interactive dashboard with:
- Analysis metadata and date range
- Capture ratios table with interpretations
- Hit rates on down days
- Performance summary across market conditions
- All 7 visualization charts embedded
- Professional styling, print-friendly

**Simply open this file in your browser to view all results!**

### 📈 Visualization Charts (`html/charts/`)

The analysis automatically generates 7 charts:

1. **1_cumulative_returns.png** - Line chart showing portfolio vs benchmark performance over time
2. **2_capture_ratios.png** - Bar chart comparing upside/downside capture across all benchmarks
3. **3_hit_rates.png** - Bar chart showing portfolio positive hit rates on benchmark down days
4. **4_return_distributions.png** - Box plots comparing daily return distributions
5. **5_avg_returns_up_down.png** - Portfolio average returns on benchmark up vs down days
6. **6_drawdown.png** - Drawdown comparison showing maximum declines from peaks
7. **7_portfolio_composition.png** - Pie chart of top 10 holdings and weights

All charts use a consistent color scheme:
- **Portfolio**: Blue (#2E86AB)
- **SPY**: Purple (#A23B72)
- **QQQ**: Orange (#F18F01)
- **DIA**: Green (#06A77D)
- **IWM**: Violet (#9D4EDD)

### 📁 CSV Data Files (`html/`)

For further analysis, the following CSV files are generated:
- **capture_stats.csv** - Capture ratios and hit rates
- **summary_stats.csv** - Performance statistics by market condition
- **cumulative_returns.csv** - Time series of cumulative returns
- **weights_used.csv** - Final portfolio weights after filtering
- **dropped_tickers.txt** - Tickers excluded from analysis

## Requirements

- Python 3.7+
- yfinance
- pandas
- matplotlib

## License

Private repository for personal use.
