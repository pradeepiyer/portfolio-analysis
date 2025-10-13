# Portfolio Downside Protection Analyzer

Analyzes portfolio performance against market benchmarks (SPY, QQQ, DIA, IWM) to measure downside protection and upside participation.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run analysis (example portfolios included in portfolios/ directory)
python analyze.py portfolios/faang.csv    # Fetch data, outputs to faang/ (~2-5 min)
python visualize.py faang                 # Generate charts in faang/ (~10 sec)
open faang/index.html                     # View dashboard
```

**Note:** Output directory is automatically named from the CSV filename (e.g., `portfolios/faang.csv` → `faang/`, `my_portfolio.csv` → `my_portfolio/`).

## Portfolio Input

Create a CSV file with your holdings:

```csv
Symbol,Weight
PHYS,0.1499
CEF,0.0789
TPL,0.0782
```

Weights should sum to ~1.0 (auto-normalized after filtering).

## Example Portfolios

This repo includes 6 pre-configured portfolio templates in `portfolios/`:

**1. FAANG** (`portfolios/faang.csv`) - Equal-weight mega-cap tech
```csv
Symbol,Weight
META,0.20
AAPL,0.20
AMZN,0.20
NFLX,0.20
GOOGL,0.20
```

**2. 60/40** (`portfolios/60_40.csv`) - Classic balanced portfolio
- 60% SPY, 40% AGG

**3. All Weather** (`portfolios/all_weather.csv`) - Ray Dalio's risk parity
- 30% VTI, 40% TLT, 15% IEI, 7.5% GLD, 7.5% DBC

**4. Permanent Portfolio** (`portfolios/permanent_portfolio.csv`) - Harry Browne's equal-weight
- 25% VTI, 25% TLT, 25% BIL, 25% GLD

**5. Bogleheads Three-Fund** (`portfolios/bogleheads_three_fund.csv`) - Diversified index approach
- 54% VTI, 18% VXUS, 28% BND

**6. Golden Butterfly** (`portfolios/golden_butterfly.csv`) - Tyler's 5-asset risk parity
- 20% VTI, 20% VBR, 20% TLT, 20% SHY, 20% GLD

Run any example:
```bash
python analyze.py portfolios/60_40.csv
python visualize.py 60_40
open 60_40/index.html
```

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Downside Capture** | Portfolio loss / Benchmark loss on down days | < 100% (lower is better) |
| **Upside Capture** | Portfolio gain / Benchmark gain on up days | > 100% (higher is better) |
| **Hit Rate** | % days portfolio ≥ 0% when benchmark < 0% | Higher is better |

### Example Results: FAANG Portfolio

The included FAANG portfolio (`portfolios/faang.csv`) demonstrates benchmark correlation analysis:

**Holdings:** 5 equal-weight mega-cap tech stocks
- META, AAPL, AMZN, NFLX, GOOGL (20% each)

**Correlation Profile:**
- Downside capture vs SPY: **134.0%** (amplifies losses)
- Upside capture vs SPY: **135.5%** (amplifies gains)
- Stress-day hit rate: **14.2%** (low independence)

**Interpretation:** High-beta portfolio with strong correlation to benchmarks. Amplifies movements in both directions—excellent for growth-focused strategies but offers minimal downside protection.

## Output Files

All files are saved to a directory named after your input CSV (e.g., `portfolio.csv` → `portfolio/` directory).

**Main dashboard:** `index.html` - All metrics and 7 charts in one page

**Key CSV files:**
- `capture_stats.csv` - Capture ratios and hit rates
- `summary_stats.csv` - Performance by market condition
- `cumulative_returns.csv` - Daily time series
- `day_by_day_both_down.csv` - Severe stress days (SPY+QQQ down)
- `weights_used.csv` - Final portfolio weights
- `dropped_tickers.txt` - Excluded tickers

**Charts:**
1. Cumulative returns over time
2. Capture ratios (up/down)
3. Hit rates on down days
4. Return distributions (box plots)
5. Average returns by condition
6. Drawdown analysis
7. Portfolio composition

## Configuration

Edit `analyze.py`:

```python
# Analysis period
START_DATE = END_DATE - timedelta(days=365*5)  # 5 years (default)

# Benchmarks
BENCH = ["SPY", "QQQ", "DIA", "IWM"]

# Data quality
MIN_COVERAGE = 0.90  # 90% data overlap required

# Ticker aliases
ALIASES = {
    "BRK.B": "BRK-B",
}
```

## Two-Script Workflow

**analyze.py portfolio.csv** - Downloads data, calculates metrics, outputs CSVs to `portfolio/`
- Run when: data is stale, weights changed, config changed
- Time: 2-5 minutes
- Usage: `python analyze.py <file.csv>` (required argument)

**visualize.py portfolio** - Reads CSVs, generates charts and HTML
- Run when: after analyze.py, or tweaking chart styles
- Time: ~10 seconds
- Usage: `python visualize.py [directory]` (defaults to `current`)

**Why separate?** Iterate on visualizations without re-downloading market data.

## Troubleshooting

**Ticker download failures:**
- Check `dropped_tickers.txt`
- Add aliases to `ALIASES` dict (e.g., `BRK.B → BRK-B`)
- Verify ticker on finance.yahoo.com

**Insufficient coverage:**
- Lower `MIN_COVERAGE` (e.g., 0.80)
- Shorten analysis period
- Accept the exclusion

**Missing matplotlib:**
```bash
pip install matplotlib
```

## Advanced Usage

**Custom benchmarks:**
```python
BENCH = ["VTI", "VXUS", "BND", "GLD"]
```

**Historical snapshots:**
```bash
# Create dated portfolio files
cp my_portfolio.csv my_portfolio_2024-10-12.csv
cp my_portfolio.csv my_portfolio_2024-11-01.csv

# Analyze each (creates my_portfolio_YYYY-MM-DD/ directories)
python analyze.py my_portfolio_2024-10-12.csv
python analyze.py my_portfolio_2024-11-01.csv

# Generate visualizations
python visualize.py my_portfolio_2024-10-12
python visualize.py my_portfolio_2024-11-01
```

## Requirements

- Python 3.7+
- yfinance, pandas, matplotlib
- No API keys required

## Repository Structure

```
stock/
├── analyze.py                # Data pipeline (requires CSV argument)
├── visualize.py              # Chart generator (optional directory arg)
├── portfolios/               # Example portfolio templates
│   ├── faang.csv             # Equal-weight mega-cap tech
│   ├── 60_40.csv             # Classic 60/40 balanced
│   ├── all_weather.csv       # Ray Dalio's risk parity
│   ├── permanent_portfolio.csv  # Harry Browne's equal-weight
│   ├── bogleheads_three_fund.csv  # Diversified index
│   └── golden_butterfly.csv  # Tyler's 5-asset risk parity
├── faang/, 60_40/, etc.      # Auto-created output directories
│   ├── index.html            # Dashboard
│   ├── *.csv                 # Analysis data
│   └── charts/               # PNG visualizations
└── output/                   # Historical analysis data
```
