# Portfolio Downside Protection Analyzer

Analyzes portfolio performance against market benchmarks (SPY, QQQ, DIA, IWM) to measure downside protection and upside participation.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run analysis (requires portfolio CSV file)
python analyze.py portfolio.csv    # Fetch data, outputs to portfolio/ (~2-5 min)
python visualize.py portfolio      # Generate charts in portfolio/ (~10 sec)
open portfolio/index.html          # View dashboard
```

**Note:** Output directory is automatically named from the CSV filename (e.g., `portfolio.csv` → `portfolio/`, `faang.csv` → `faang/`).

## Portfolio Input

Create a CSV file with your holdings:

```csv
Symbol,Weight
PHYS,0.1499
CEF,0.0789
TPL,0.0782
```

Weights should sum to ~1.0 (auto-normalized after filtering).

## Multi-Portfolio Analysis

```bash
# Analyze different portfolios (auto-creates faang/ directory)
python analyze.py faang.csv
python visualize.py faang
open faang/index.html
```

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Downside Capture** | Portfolio loss / Benchmark loss on down days | < 100% (lower is better) |
| **Upside Capture** | Portfolio gain / Benchmark gain on up days | > 100% (higher is better) |
| **Hit Rate** | % days portfolio ≥ 0% when benchmark < 0% | Higher is better |

### Example: Portfolio vs FAANG

This repo includes two analyzed portfolios:

**Portfolio** (31 holdings: metals, REITs, energy)
- Downside capture vs SPY: **63.4%** ✓
- Upside capture vs SPY: 81.4%
- Stress-day hit rate: 33.4%

**FAANG** (5 equal-weight tech)
- Downside capture vs SPY: 134.0% (amplifies losses)
- Upside capture vs SPY: **135.5%** ✓
- Stress-day hit rate: 14.2%

**Trade-off:** Portfolio sacrifices 19% upside for 37% better downside protection.

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
cp portfolio.csv portfolio_2024-10-12.csv
cp portfolio.csv portfolio_2024-11-01.csv

# Analyze each (creates portfolio_YYYY-MM-DD/ directories)
python analyze.py portfolio_2024-10-12.csv
python analyze.py portfolio_2024-11-01.csv

# Generate visualizations
python visualize.py portfolio_2024-10-12
python visualize.py portfolio_2024-11-01
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
├── portfolio.csv, faang.csv  # Example portfolio input files
├── portfolio/, faang/        # Auto-created output directories
│   ├── index.html            # Dashboard
│   ├── *.csv                 # Analysis data
│   └── charts/               # PNG visualizations
└── output/                   # Historical analysis data
```
