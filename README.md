# Portfolio Downside Protection Analyzer

A comprehensive Python toolkit for evaluating portfolio downside protection strategies. Analyzes portfolio performance against 4 major market benchmarks to measure risk-adjusted returns, capture ratios, and stress-day performance.

**Core Philosophy:** Protect capital on down days while participating in market upside.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Multi-Portfolio Analysis](#multi-portfolio-analysis)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Understanding the Metrics](#understanding-the-metrics)
- [Output Reference](#output-reference)
- [Example Insights](#example-insights)
- [Troubleshooting](#troubleshooting)

## Overview

This tool performs deep comparative analysis of portfolios against 4 market benchmarks:

- **SPY** - S&P 500 (large cap)
- **QQQ** - Nasdaq 100 (tech-heavy)
- **DIA** - Dow Jones Industrial Average (blue chip)
- **IWM** - Russell 2000 (small cap)

### What It Measures

- **Downside Capture Ratio**: Portfolio losses vs benchmark losses on down days (lower = better protection)
- **Upside Capture Ratio**: Portfolio gains vs benchmark gains on up days (higher = better participation)
- **Hit Rates**: % of days portfolio stays flat/positive when benchmark is down
- **Stress Performance**: Behavior when both SPY and QQQ decline (severe market stress)
- **Drawdowns**: Maximum peak-to-trough declines
- **Return Distributions**: Statistical spread of daily returns

### Key Features

✓ **Multi-Portfolio Support**: Analyze and compare multiple portfolios simultaneously
✓ **7 Professional Charts**: Publication-ready visualizations with colorblind-safe palette
✓ **Comprehensive CSV Exports**: Day-by-day analysis, capture ratios, coverage reports
✓ **Interactive HTML Dashboard**: All metrics and charts in one browser-viewable report
✓ **Robust Data Handling**: Automatic ticker aliases, fallback logic, coverage filtering
✓ **Separated Analysis & Visualization**: Iterate on chart styling without re-fetching data

## Quick Start

### Installation

```bash
# Clone or download this repository
cd stock

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

1. **Create your portfolio file** (or use the example `portfolio.csv`):

```csv
Symbol,Weight
PHYS,0.1499
CEF,0.0789
TPL,0.0782
FNV,0.0425
SPOT,0.0402
```

**Important:** Weights should sum to ~1.0. The script auto-normalizes after filtering.

2. **Run the analysis pipeline**:

```bash
# Step 1: Fetch data and calculate metrics (~2-5 minutes)
python analyze.py

# Step 2: Generate charts and HTML dashboard (~10 seconds)
python visualize.py

# Step 3: View the results
open html/index.html
```

That's it! The dashboard contains all metrics, charts, and interpretations.

## Multi-Portfolio Analysis

This repo supports analyzing multiple portfolios with organized output directories.

### Analyzing a Different Portfolio

```bash
# Create a portfolio file (e.g., faang.csv)
cat > faang.csv << EOF
Symbol,Weight
META,0.20
AAPL,0.20
AMZN,0.20
NFLX,0.20
GOOGL,0.20
EOF

# Run analysis with custom output directory
python analyze.py --output faang faang.csv
python visualize.py faang
open faang/index.html
```

### Current Portfolio Examples

This repository includes two analyzed portfolios:

**`portfolio/`** - Diversified defensive portfolio
- 31 holdings: precious metals, REITs, energy MLPs, growth stocks
- **Downside capture vs SPY**: 63.4% (strong protection)
- **Upside capture vs SPY**: 81.4% (good participation)
- **Hit rate on stress days**: 33.4%

**`faang/`** - Equal-weight mega-cap tech
- 5 holdings: META, AAPL, AMZN, NFLX, GOOGL (20% each)
- **Downside capture vs SPY**: 134.0% (amplifies losses)
- **Upside capture vs SPY**: 135.5% (amplifies gains)
- **Hit rate on stress days**: 14.2%

**Strategic Comparison:** These portfolios represent opposite risk profiles—defensive vs aggressive growth. See [Example Insights](#example-insights) for detailed comparison.

## Architecture

### Two-Script Workflow

The analysis is intentionally split for efficiency:

#### 1. `analyze.py` - Data Pipeline (Slow: ~2-5 minutes)

**What it does:**
- Downloads 3+ years of historical prices via yfinance API
- Aligns data across portfolio + 4 benchmarks
- Handles ticker aliases, fallbacks, and data quality filtering
- Calculates all metrics: capture ratios, hit rates, drawdowns
- Exports CSV files to output directory

**When to run:**
- Fresh data needed (daily/weekly updates)
- Portfolio weights changed
- Date range or benchmark configuration changed

**Command-line options:**
```bash
python analyze.py [--output DIR] [portfolio_file.csv]

# Examples:
python analyze.py                          # Uses portfolio.csv → html/
python analyze.py faang.csv                # Uses faang.csv → html/
python analyze.py --output custom faang.csv  # Custom output directory
```

#### 2. `visualize.py` - Presentation Layer (Fast: ~10 seconds)

**What it does:**
- Reads CSV files from analyze.py output
- Generates 7 matplotlib charts as PNG files
- Builds HTML dashboard with embedded charts and tables
- No network calls or heavy computation

**When to run:**
- After every analyze.py run
- When tweaking chart styles, colors, or layout
- Multiple times to experiment with visualization options

**Command-line usage:**
```bash
python visualize.py [directory]

# Examples:
python visualize.py          # Reads from html/, outputs to html/
python visualize.py faang    # Reads from faang/, outputs to faang/
```

**Why this separation?** You can iterate on visualizations (colors, labels, chart types) without waiting 5 minutes to re-download market data. Run `analyze.py` once, then `visualize.py` as many times as needed.

## Configuration

### Analysis Period

Edit `analyze.py` (lines 13-14):

```python
END_DATE = date.today()                    # Today (default)
START_DATE = END_DATE - timedelta(days=365*5)  # 5 years back

# Or use fixed dates:
START_DATE = date(2020, 1, 1)
END_DATE = date(2024, 12, 31)
```

### Benchmark Selection

Edit `analyze.py` (line 16):

```python
BENCH = ["SPY", "QQQ", "DIA", "IWM"]  # Default 4 benchmarks

# Add more:
BENCH = ["SPY", "QQQ", "DIA", "IWM", "VTI", "EFA"]  # + Total Market, Int'l
```

**Note:** Changing benchmarks affects all metrics. Re-run both scripts after changing.

### Ticker Aliases (Yahoo Finance Variants)

Some tickers need special handling for Yahoo Finance. Edit `ALIASES` dict in `analyze.py` (line 24):

```python
ALIASES = {
    "BRK.B": "BRK-B",    # Berkshire Hathaway B shares
    "BF.B": "BF-B",      # Brown-Forman B shares
}
```

**Auto-fallback:** The script automatically tries `.` → `-` conversion for share classes (e.g., `BRK.B` → `BRK-B`).

### Data Quality Settings

Edit `analyze.py`:

```python
MIN_COVERAGE = 0.90  # Require 90% data overlap with benchmarks (default)

# Relax for more recent holdings or IPOs:
MIN_COVERAGE = 0.80  # 80% coverage (includes stocks with shorter history)
```

**Trade-off:** Lower coverage = more tickers included, but data quality decreases.

## Understanding the Metrics

### Downside Capture Ratio

**Formula:** `(Portfolio avg return on benchmark down days) / (Benchmark avg return on down days)`

**Interpretation:**
- **< 100%** - Portfolio loses *less* than benchmark (good protection)
- **= 100%** - Portfolio loses *same* as benchmark (neutral)
- **> 100%** - Portfolio loses *more* than benchmark (amplifies losses)

**Example:** 63.4% vs SPY = portfolio experiences only 63.4% of SPY's losses

### Upside Capture Ratio

**Formula:** `(Portfolio avg return on benchmark up days) / (Benchmark avg return on up days)`

**Interpretation:**
- **> 100%** - Portfolio gains *more* than benchmark (amplifies upside)
- **= 100%** - Portfolio gains *same* as benchmark (neutral)
- **< 100%** - Portfolio gains *less* than benchmark (conservative)

**Example:** 135.5% vs SPY = portfolio captures 135.5% of SPY's gains

### Hit Rate

**Formula:** `% of days portfolio return >= 0% when benchmark < 0%`

**Interpretation:**
- **Higher is better** - More frequent "defense" when market declines
- **33.4% on BOTH-down days** - Portfolio stayed flat/positive 1 in 3 stress days

**Example:** FAANG 14.2% vs Portfolio 33.4% on SPY+QQQ down days → Portfolio has 2.4x better stress protection

### Benchmark Conditions

The analysis categorizes every trading day into conditions:

- **`SPY_down_days`**: Days when SPY < 0%
- **`QQQ_down_days`**: Days when QQQ < 0%
- **`DIA_down_days`**: Days when DIA < 0%
- **`IWM_down_days`**: Days when IWM < 0%
- **`Both_down_days`**: Days when *both* SPY AND QQQ < 0% (severe market stress)
- **`{BENCH}_up_or_flat_days`**: Days when benchmark >= 0%

## Output Reference

All outputs are saved to the specified directory (`html/` by default, or `--output` argument).

### HTML Dashboard

**`index.html`** - Main interactive report

**Contents:**
- Analysis metadata (date range, portfolio name, generated timestamp)
- Capture ratios table with color-coded interpretations
- Hit rates bar chart
- Performance summary table (up/down day statistics)
- All 7 visualization charts embedded inline
- Clean, print-friendly styling

**Just open this file in any browser!**

### Visualization Charts (`charts/` subdirectory)

All charts use a **colorblind-safe palette** (Okabe-Ito):

| Asset | Color | Hex |
|-------|-------|-----|
| Portfolio | Blue | #0173B2 |
| SPY | Orange | #DE8F05 |
| QQQ | Pink/Magenta | #CC78BC |
| DIA | Bluish Green | #029E73 |
| IWM | Yellow | #ECE133 |

#### Chart 1: Cumulative Returns
**File:** `1_cumulative_returns.png`
**Type:** Line chart
**Shows:** Portfolio vs all 4 benchmarks over time (starting at 0%)
**Use:** Identify periods of outperformance/underperformance

#### Chart 2: Capture Ratios
**File:** `2_capture_ratios.png`
**Type:** Grouped bar chart
**Shows:** Upside and downside capture for each benchmark
**Use:** Quick assessment of risk/reward profile

#### Chart 3: Hit Rates
**File:** `3_hit_rates.png`
**Type:** Bar chart
**Shows:** % of days portfolio is non-negative when benchmark is down
**Use:** Measure defensive capability on down days

#### Chart 4: Return Distributions
**File:** `4_return_distributions.png`
**Type:** Box plots
**Shows:** Statistical distribution of daily returns (median, quartiles, outliers)
**Use:** Understand volatility and tail risk

#### Chart 5: Average Returns (Up vs Down Days)
**File:** `5_avg_returns_up_down.png`
**Type:** Grouped bar chart
**Shows:** Portfolio's avg return on benchmark up vs down days
**Use:** Compare asymmetric performance

#### Chart 6: Drawdown Analysis
**File:** `6_drawdown.png`
**Type:** Area chart
**Shows:** Cumulative peak-to-trough declines over time
**Use:** Identify maximum loss periods and recovery times

#### Chart 7: Portfolio Composition
**File:** `7_portfolio_composition.png`
**Type:** Pie chart (top 10 + "Others")
**Shows:** Weight distribution of portfolio holdings
**Use:** Verify portfolio construction and concentration

### CSV Data Files

#### Core Metrics

**`capture_stats.csv`**
Capture ratios and hit rates:
```csv
Metric,Value
Downside capture vs SPY (avg),0.634156265558055
Upside capture vs SPY (avg),0.8143500269094379
Hit-rate on SPY-down days (>=0%),0.3274956217162872
...
```

**`summary_stats.csv`**
Performance by market condition:
```csv
Category,Days,Portfolio_hit_rate_(>=0%),Portfolio_avg,Portfolio_median,...
SPY_down_days,571.0,0.3275,-0.00492,-0.00433,-0.0699,...
Both_down_days,485.0,0.3340,-0.00506,-0.00442,-0.0699,...
```

#### Time Series

**`cumulative_returns.csv`**
Daily cumulative returns for portfolio + benchmarks:
```csv
Date,Portfolio,SPY,QQQ,DIA,IWM
2019-10-14,0.0,0.0,0.0,0.0,0.0
2019-10-15,0.0021,0.0032,0.0041,0.0028,0.0015
...
```

**`day_by_day_all.csv`**
Every trading day with returns for portfolio + benchmarks

**`day_by_day_{BENCH}_down.csv`**
Filtered to only days when specific benchmark declined (e.g., `day_by_day_SPY_down.csv`)

**`day_by_day_both_down.csv`**
Filtered to severe stress days (SPY AND QQQ both down)

#### Portfolio Management

**`weights_used.csv`**
Final normalized portfolio weights after filtering:
```csv
Ticker,Weight
PHYS,0.16507887584296832
CEF,0.08688095858453329
...
```

**`dropped_tickers.txt`**
Tickers excluded from analysis with reasons:
```
Dropped for insufficient coverage (<90% of baseline):
- TFPM
- UROY
- NE
- VAL
```

**`coverage_report.csv`** *(if generated)*
Data coverage statistics per ticker

**`adj_close_all_aligned.csv`** *(if generated)*
Raw aligned adjusted close prices (debugging)

## Example Insights

### Real Comparison: Defensive vs Aggressive

Using the included portfolios (`portfolio/` vs `faang/`):

#### Downside Protection

| Metric | Portfolio | FAANG | Winner |
|--------|-----------|-------|--------|
| Downside capture vs SPY | **63.4%** | 134.0% | Portfolio |
| Downside capture vs QQQ | **34.5%** | 107.7% | Portfolio |
| Hit rate (SPY+QQQ down) | **33.4%** | 14.2% | Portfolio |
| Avg loss (stress days) | **-0.51%** | -1.30% | Portfolio |

**Insight:** Portfolio loses 2.5x less than FAANG on severe down days.

#### Upside Participation

| Metric | Portfolio | FAANG | Winner |
|--------|-----------|-------|--------|
| Upside capture vs SPY | 81.4% | **135.5%** | FAANG |
| Upside capture vs QQQ | 50.7% | **110.3%** | FAANG |

**Insight:** FAANG amplifies gains 1.35x vs SPY, while Portfolio takes 81% of upside.

#### Strategic Profiles

**Portfolio = Defensive**
- Goal: Capital preservation with growth
- Holdings: Gold (PHYS), REITs (TPL), miners (FNV), diversified
- Result: 63% downside capture, 81% upside capture

**FAANG = Aggressive Growth**
- Goal: Maximum tech exposure
- Holdings: Equal-weight mega-cap tech
- Result: 134% downside capture, 135% upside capture (high beta)

**The trade-off:** Portfolio sacrifices 19% of SPY's upside to get 37% better downside protection.

## Troubleshooting

### Ticker Download Failures

**Symptom:** `dropped_tickers.txt` shows download failures

**Causes:**
- International/OTC stocks not on Yahoo Finance
- Delisted companies
- Ticker symbol mismatch

**Solutions:**
1. Check if ticker needs an alias (add to `ALIASES` dict)
2. Verify ticker on finance.yahoo.com
3. For share classes, try both `.` and `-` (e.g., BRK.B vs BRK-B)
4. Remove delisted stocks from portfolio CSV

### Insufficient Coverage

**Symptom:** Tickers in `dropped_tickers.txt` with "insufficient coverage" message

**Cause:** Stock has < 90% data overlap with benchmark calendar (e.g., recent IPO, delisted, trading halt)

**Solutions:**
1. Lower `MIN_COVERAGE` threshold (e.g., 0.80 for 80%)
2. Shorten analysis period (`START_DATE` closer to present)
3. Accept the exclusion (analysis continues with remaining tickers)

### Missing matplotlib

**Symptom:** `visualize.py` fails with ImportError

**Solution:**
```bash
pip install matplotlib
```

### Wrong Weights Sum

**Symptom:** Warning about weights not summing to 1.0

**Solution:** The script auto-normalizes weights. This is informational only. To fix manually:
```python
# Check your CSV - weights should sum to 1.0
# The script will normalize automatically after filtering
```

### Empty Results

**Symptom:** No charts or empty HTML

**Causes:**
1. All tickers failed download
2. No overlapping dates between portfolio and benchmarks
3. Wrong file paths

**Debug:**
```bash
# Check if CSV files exist
ls -la html/*.csv

# Check dropped_tickers.txt
cat html/dropped_tickers.txt

# Verify date range overlap
# (recent IPOs may have no data in 3-year lookback)
```

## Advanced Usage

### Custom Benchmark Analysis

Want to compare against different indices?

```python
# In analyze.py, line 16:
BENCH = ["VTI", "VXUS", "BND", "GLD"]  # Total market, international, bonds, gold
```

### Historical Snapshots

Track portfolio evolution:

```bash
# Save dated snapshots
python analyze.py --output snapshots/2024-10-12 portfolio.csv
python analyze.py --output snapshots/2024-11-01 portfolio.csv
python analyze.py --output snapshots/2024-12-01 portfolio.csv

# Compare dashboards
open snapshots/*/index.html
```

### Exporting for Excel

All CSV files can be imported directly into Excel/Google Sheets for custom analysis:

```bash
# Example: Import day_by_day_both_down.csv to analyze worst stress days
open html/day_by_day_both_down.csv
```

## Requirements

- **Python**: 3.7+
- **Dependencies**:
  - `yfinance` - Yahoo Finance API wrapper
  - `pandas` - Data processing and alignment
  - `matplotlib` - Chart generation

No API keys required. Uses free Yahoo Finance data.

## Repository Structure

```
stock/
├── analyze.py              # Data pipeline (fetch + calculate)
├── visualize.py            # Visualization generator
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── CLAUDE.md               # AI assistant instructions
│
├── portfolio.csv           # Example: Diversified defensive portfolio
├── faang.csv              # Example: Equal-weight mega-cap tech
│
├── html/                  # Default output directory
│   ├── index.html         # Main dashboard
│   ├── *.csv              # Analysis data
│   └── charts/            # PNG visualizations
│
├── portfolio/             # Analyzed: diversified portfolio
│   ├── index.html
│   ├── charts/
│   └── *.csv
│
├── faang/                 # Analyzed: FAANG portfolio
│   ├── index.html
│   ├── charts/
│   └── *.csv
│
└── output/                # Historical analysis output
    └── *.csv
```

## License

Private repository for personal portfolio analysis.

## Credits

Built with:
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance market data
- [pandas](https://pandas.pydata.org/) - Data analysis library
- [matplotlib](https://matplotlib.org/) - Visualization library
- [Okabe-Ito color palette](https://jfly.uni-koeln.de/color/) - Colorblind-safe colors
