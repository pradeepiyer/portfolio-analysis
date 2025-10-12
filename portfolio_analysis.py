#!/usr/bin/env python3
import os, time, re
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please install dependencies first:\n  pip install yfinance pandas\n" + str(e))

# ---- Config ----
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365*3)

WEIGHTS_CSV = "weights.csv"   # expected columns: Symbol,Weight (or Ticker,Weight)
BENCH = ["SPY", "QQQ", "DIA", "IWM"]  # SPY=S&P500, QQQ=Nasdaq100, DIA=DJIA, IWM=Russell2000
OUTDIR = "html"
os.makedirs(OUTDIR, exist_ok=True)

# Ticker sanitation
ALLOWED = re.compile(r'^[A-Z][A-Z0-9\.\-]*$')
DROP_SET = {"", "-", "CASH", "USD", "SWEEP", "MARGIN", "FX", "CUR", "CURRENCY", "MONEY MARKET", "MMF"}

# Simple alias mapping to Yahoo variants (edit as needed)
ALIASES = {
    # "LB": "BBWI",   # Removed - user's LB is Landbridge, not Bath & Body Works
}

def sanitize_ticker(t: str) -> Optional[str]:
    if t is None:
        return None
    t = str(t).strip().upper()
    if t in DROP_SET:
        return None
    if not ALLOWED.match(t):
        return None
    return t

def share_class_fallback(t: str) -> Optional[str]:
    # e.g., BRK.B -> BRK-B
    if "." in t and t.count(".") == 1:
        base, suffix = t.split(".")
        if suffix.isalpha() and len(suffix) <= 2 and base and base[0].isalpha():
            return f"{base}-{suffix}"
    return None

def extract_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        raise RuntimeError("Empty dataframe for %s" % ticker)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            s = df.xs("Adj Close", axis=1, level=0)
        elif "Close" in lvl0:
            s = df.xs("Close", axis=1, level=0)
        else:
            raise RuntimeError("Missing Adj Close/Close for %s" % ticker)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = ticker
        return s.dropna()
    cols = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}
    col = cols.get("adjclose") or cols.get("close")
    if not col:
        raise RuntimeError("Missing Adj Close/Close columns for %s: %s" % (ticker, df.columns))
    s = df[col].copy()
    s.name = ticker
    return s.dropna()

def try_download(t: str, start, end, retries: int = 4) -> Optional[pd.Series]:
    last = None
    for i in range(1, retries + 1):
        try:
            d = yf.download(t, start=start, end=end, progress=False, auto_adjust=False, group_by="column")
            s = extract_series(d, t)
            s.index = pd.to_datetime(s.index)
            if s.empty:
                raise RuntimeError("Empty series after extraction")
            return s
        except Exception as e:
            last = e
            sleep = 0.6 * (2 ** (i - 1)) + 0.2 * (i - 1)
            print(f"[{t}] attempt {i}/{retries} failed: {e}. retrying in {sleep:.1f}s...")
            time.sleep(sleep)
    print(f"[{t}] failed after {retries} attempts. Last error: {last}")
    return None

def fetch_with_fallbacks(t: str, start, end) -> Optional[pd.Series]:
    # Alias
    t_eff = ALIASES.get(t, t)
    s = try_download(t_eff, start, end)
    if s is not None:
        s.name = t  # keep original label
        return s
    # Share-class fallback
    alt = share_class_fallback(t_eff)
    if alt and alt != t_eff:
        print(f"[{t}] trying share-class fallback variant: {alt}")
        s = try_download(alt, start, end)
        if s is not None:
            s.name = t
            return s
    return None

def load_weights(path: str) -> pd.Series:
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Place it next to this script.")
    wdf = pd.read_csv(path)
    # detect columns
    sym_col = None; wt_col = None
    lower = {c.lower(): c for c in wdf.columns}
    for key in lower:
        if key in ("symbol","ticker"):
            sym_col = lower[key]; break
    for key in lower:
        if key in ("weight","pct","percent"):
            wt_col = lower[key]; break
    if sym_col is None or wt_col is None:
        raise SystemExit(f"Could not detect Symbol/Weight columns in {path}. Columns: {list(wdf.columns)}")
    # sanitize
    wdf = wdf[[sym_col, wt_col]].rename(columns={sym_col: "Ticker", wt_col: "Weight"})
    wdf["Ticker"] = wdf["Ticker"].apply(sanitize_ticker)
    wdf = wdf.dropna(subset=["Ticker", "Weight"])
    wdf = wdf[wdf["Weight"] > 0]
    if wdf.empty:
        raise SystemExit("All tickers filtered out as invalid.")
    wdf["Weight"] = wdf["Weight"] / wdf["Weight"].sum()
    return wdf.set_index("Ticker")["Weight"]

def main():
    # Load initial weights and build download list
    weights = load_weights(WEIGHTS_CSV)
    tickers = list(weights.index) + BENCH
    print(f"Fetching {len(tickers)} tickers from {START_DATE} to {END_DATE}")

    # Download benchmarks first to establish baseline index
    bench_series = {}
    for b in BENCH:
        s = fetch_with_fallbacks(b, START_DATE, END_DATE)
        if s is None:
            raise SystemExit(f"Benchmark {b} could not be downloaded.")
        bench_series[b] = s
        time.sleep(0.3)
    # Baseline index = intersection of all benchmark calendars
    base_index = bench_series[BENCH[0]].index
    for b in BENCH[1:]:
        base_index = base_index.intersection(bench_series[b].index)
    if base_index.size < 200:
        raise SystemExit("Baseline index too small. Check date range.")

    # Download holdings
    series_map = {}
    failed = []
    for t in weights.index:
        s = fetch_with_fallbacks(t, START_DATE, END_DATE)
        if s is None:
            failed.append((t, "download_failed"))
        else:
            series_map[t] = s
        time.sleep(0.3)

    # Coverage check vs baseline
    coverage_rows = []
    MIN_COVERAGE = 0.90  # require at least 90% of base_index dates
    for t, s in series_map.items():
        overlap = s.index.intersection(base_index)
        cov = len(overlap) / len(base_index)
        coverage_rows.append({"Ticker": t, "Coverage_vs_baseline": cov, "Series_rows": len(s)})

    # Keep only tickers with enough coverage
    keep = [row["Ticker"] for row in coverage_rows if row["Coverage_vs_baseline"] >= MIN_COVERAGE]
    drop_cov = [row["Ticker"] for row in coverage_rows if row["Coverage_vs_baseline"] < MIN_COVERAGE]

    # Report dropped symbols
    if failed or drop_cov:
        with open(os.path.join(OUTDIR, "dropped_tickers.txt"), "w") as f:
            if failed:
                f.write("Failed downloads:\n")
                for t, reason in failed:
                    f.write(f"- {t}: {reason}\n")
            if drop_cov:
                f.write("\nDropped for insufficient coverage (<90% of baseline):\n")
                for t in drop_cov:
                    f.write(f"- {t}\n")

    # Filter weights to kept tickers and renormalize
    weights_used = weights[weights.index.isin(keep)]
    if weights_used.empty:
        raise SystemExit("All holdings were dropped due to low coverage or download failure.")
    weights_used = weights_used / weights_used.sum()
    weights_used.to_csv(os.path.join(OUTDIR, "weights_used.csv"), header=["Weight"])

    # Build aligned adjusted-close table on the baseline index
    # Include benchmarks as well
    aligned = pd.DataFrame(index=base_index)
    for b in BENCH:
        aligned[b] = bench_series[b].reindex(base_index)
    for t in weights_used.index:
        aligned[t] = series_map[t].reindex(base_index)

    # If any column has >5% NaN after reindexing, drop it to keep data quality high
    nan_ratio = aligned.isna().mean()
    drop_nan_cols = nan_ratio[nan_ratio > 0.05].index.tolist()
    # Never drop benchmarks
    drop_nan_cols = [c for c in drop_nan_cols if c not in BENCH]
    if drop_nan_cols:
        aligned = aligned.drop(columns=drop_nan_cols)
        weights_used = weights_used.drop(index=[c for c in drop_nan_cols if c in weights_used.index], errors="ignore")
        weights_used = weights_used / weights_used.sum()
        weights_used.to_csv(os.path.join(OUTDIR, "weights_used.csv"), header=["Weight"])

    # Final sanity
    if aligned.shape[1] < 3:
        raise SystemExit("Not enough valid series after alignment.")
    aligned = aligned.dropna(how="any")
    if len(aligned) < 200:
        raise SystemExit("Aligned price table still too small after cleaning. Consider lowering coverage threshold.")

    # Compute returns
    ret = aligned.pct_change().dropna()
    w = weights_used.copy()
    missing = [t for t in w.index if t not in ret.columns]
    if missing:
        print("WARNING: missing returns for holdings:", ", ".join(missing))
        w = w.drop(missing, errors="ignore")
        w = w / w.sum()

    port = (ret[w.index] @ w.values)
    out = pd.DataFrame({
        "SPY_ret": ret["SPY"],
        "QQQ_ret": ret["QQQ"],
        "DIA_ret": ret["DIA"],
        "IWM_ret": ret["IWM"],
        "Portfolio_ret": port
    }, index=ret.index).sort_index()

    out["SPY_down"] = out["SPY_ret"] < 0
    out["QQQ_down"] = out["QQQ_ret"] < 0
    out["DIA_down"] = out["DIA_ret"] < 0
    out["IWM_down"] = out["IWM_ret"] < 0
    out["Both_down"] = out["SPY_down"] & out["QQQ_down"]

    def summarize(mask, label):
        sub = out.loc[mask, ["SPY_ret","QQQ_ret","DIA_ret","IWM_ret","Portfolio_ret"]]
        if sub.empty:
            return pd.Series({"Days":0,"Portfolio_hit_rate_(>=0%)":float("nan"),
                              "Portfolio_avg":float("nan"),"Portfolio_median":float("nan"),
                              "Portfolio_worst":float("nan"),"SPY_avg":float("nan"),"QQQ_avg":float("nan"),
                              "DIA_avg":float("nan"),"IWM_avg":float("nan")}, name=label)
        return pd.Series({"Days":int(len(sub)),
                          "Portfolio_hit_rate_(>=0%)":float((sub["Portfolio_ret"]>=0).mean()),
                          "Portfolio_avg":float(sub["Portfolio_ret"].mean()),
                          "Portfolio_median":float(sub["Portfolio_ret"].median()),
                          "Portfolio_worst":float(sub["Portfolio_ret"].min()),
                          "SPY_avg":float(sub["SPY_ret"].mean()),"QQQ_avg":float(sub["QQQ_ret"].mean()),
                          "DIA_avg":float(sub["DIA_ret"].mean()),"IWM_avg":float(sub["IWM_ret"].mean())}, name=label)

    summary = pd.concat([
        summarize(out["SPY_down"], "SPY_down_days"),
        summarize(out["QQQ_down"], "QQQ_down_days"),
        summarize(out["DIA_down"], "DIA_down_days"),
        summarize(out["IWM_down"], "IWM_down_days"),
        summarize(out["Both_down"], "Both_down_days"),
        summarize(~out["SPY_down"], "SPY_up_or_flat_days"),
        summarize(~out["QQQ_down"], "QQQ_up_or_flat_days"),
        summarize(~out["DIA_down"], "DIA_up_or_flat_days"),
        summarize(~out["IWM_down"], "IWM_up_or_flat_days"),
    ], axis=1).T

    # Capture stats
    spy_down = out.loc[out["SPY_down"]]
    qqq_down = out.loc[out["QQQ_down"]]
    dia_down = out.loc[out["DIA_down"]]
    iwm_down = out.loc[out["IWM_down"]]
    spy_up = out.loc[~out["SPY_down"]]
    qqq_up = out.loc[~out["QQQ_down"]]
    dia_up = out.loc[~out["DIA_down"]]
    iwm_up = out.loc[~out["IWM_down"]]
    def safe_div(a, b): return float(a/b) if b and not pd.isna(b) and b != 0 else float("nan")
    capture = pd.DataFrame([
        ["Downside capture vs SPY (avg)", safe_div(spy_down["Portfolio_ret"].mean(), spy_down["SPY_ret"].mean())],
        ["Downside capture vs QQQ (avg)", safe_div(qqq_down["Portfolio_ret"].mean(), qqq_down["QQQ_ret"].mean())],
        ["Downside capture vs DIA (avg)", safe_div(dia_down["Portfolio_ret"].mean(), dia_down["DIA_ret"].mean())],
        ["Downside capture vs IWM (avg)", safe_div(iwm_down["Portfolio_ret"].mean(), iwm_down["IWM_ret"].mean())],
        ["Upside capture vs SPY (avg)",   safe_div(spy_up["Portfolio_ret"].mean(), spy_up["SPY_ret"].mean())],
        ["Upside capture vs QQQ (avg)",   safe_div(qqq_up["Portfolio_ret"].mean(), qqq_up["QQQ_ret"].mean())],
        ["Upside capture vs DIA (avg)",   safe_div(dia_up["Portfolio_ret"].mean(), dia_up["DIA_ret"].mean())],
        ["Upside capture vs IWM (avg)",   safe_div(iwm_up["Portfolio_ret"].mean(), iwm_up["IWM_ret"].mean())],
        ["Hit-rate on SPY-down days (>=0%)", float((spy_down["Portfolio_ret"] >= 0).mean()) if len(spy_down) else float("nan")],
        ["Hit-rate on QQQ-down days (>=0%)", float((qqq_down["Portfolio_ret"] >= 0).mean()) if len(qqq_down) else float("nan")],
        ["Hit-rate on DIA-down days (>=0%)", float((dia_down["Portfolio_ret"] >= 0).mean()) if len(dia_down) else float("nan")],
        ["Hit-rate on IWM-down days (>=0%)", float((iwm_down["Portfolio_ret"] >= 0).mean()) if len(iwm_down) else float("nan")],
        ["Hit-rate on BOTH-down days (>=0%)", float((out.loc[out["Both_down"], "Portfolio_ret"] >= 0).mean()) if out["Both_down"].any() else float("nan")],
    ], columns=["Metric","Value"])

    # Exports
    summary.to_csv(os.path.join(OUTDIR, "summary_stats.csv"), index_label="Category")
    capture.to_csv(os.path.join(OUTDIR, "capture_stats.csv"), index=False)

    cum = (1 + out[["Portfolio_ret","SPY_ret","QQQ_ret","DIA_ret","IWM_ret"]]).cumprod() - 1
    cum.to_csv(os.path.join(OUTDIR, "cumulative_returns.csv"), index_label="Date")

    # ---- Generate Visualizations ----
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        CHART_DIR = os.path.join(OUTDIR, "charts")
        os.makedirs(CHART_DIR, exist_ok=True)

        # Color scheme
        COLORS = {
            'Portfolio': '#2E86AB',
            'SPY': '#A23B72',
            'QQQ': '#F18F01',
            'DIA': '#06A77D',
            'IWM': '#9D4EDD'
        }

        # 1. Cumulative Returns Chart
        fig, ax = plt.subplots(figsize=(14, 8))
        for col in cum.columns:
            label = col.replace('_ret', '')
            ax.plot(cum.index, cum[col] * 100, label=label, linewidth=2, color=COLORS.get(label, 'gray'))
        ax.set_title('Cumulative Returns: Portfolio vs Benchmarks', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '1_cumulative_returns.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Capture Ratios Comparison
        capture_data = capture[capture['Metric'].str.contains('capture')].copy()
        capture_data['Type'] = capture_data['Metric'].apply(lambda x: 'Downside' if 'Downside' in x else 'Upside')
        capture_data['Benchmark'] = capture_data['Metric'].str.extract(r'vs (\w+)')[0]

        fig, ax = plt.subplots(figsize=(12, 7))
        benchmarks = ['SPY', 'QQQ', 'DIA', 'IWM']
        x = range(len(benchmarks))
        width = 0.35
        downside_vals = [capture_data[(capture_data['Type']=='Downside') & (capture_data['Benchmark']==b)]['Value'].values[0] * 100 for b in benchmarks]
        upside_vals = [capture_data[(capture_data['Type']=='Upside') & (capture_data['Benchmark']==b)]['Value'].values[0] * 100 for b in benchmarks]

        ax.bar([i - width/2 for i in x], downside_vals, width, label='Downside Capture', color='#E63946', alpha=0.8)
        ax.bar([i + width/2 for i in x], upside_vals, width, label='Upside Capture', color='#06A77D', alpha=0.8)
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Capture Ratio (%)', fontsize=12)
        ax.set_title('Portfolio Capture Ratios vs Benchmarks', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (d, u) in enumerate(zip(downside_vals, upside_vals)):
            ax.text(i - width/2, d + 2, f'{d:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, u + 2, f'{u:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '2_capture_ratios.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Hit Rates on Down Days
        hit_rate_data = capture[capture['Metric'].str.contains('Hit-rate')].copy()
        hit_rate_data['Benchmark'] = hit_rate_data['Metric'].str.extract(r'on (\w+)-down')[0]
        hit_rate_data = hit_rate_data[hit_rate_data['Benchmark'].notna()]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(hit_rate_data['Benchmark'], hit_rate_data['Value'] * 100, color='#2E86AB', alpha=0.8)
        ax.set_ylabel('Hit Rate (%)', fontsize=12)
        ax.set_title('Portfolio Hit Rate on Benchmark Down Days\n(% of days portfolio ≥ 0%)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '3_hit_rates.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Return Distribution Box Plots
        fig, ax = plt.subplots(figsize=(12, 7))
        data_to_plot = [out['Portfolio_ret'] * 100, out['SPY_ret'] * 100, out['QQQ_ret'] * 100,
                       out['DIA_ret'] * 100, out['IWM_ret'] * 100]
        labels = ['Portfolio', 'SPY', 'QQQ', 'DIA', 'IWM']
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(COLORS.get(label, 'lightgray'))
            patch.set_alpha(0.7)
        ax.set_ylabel('Daily Return (%)', fontsize=12)
        ax.set_title('Return Distribution Comparison', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '4_return_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 5. Average Returns: Up vs Down Days
        fig, ax = plt.subplots(figsize=(14, 8))
        bench_list = ['SPY', 'QQQ', 'DIA', 'IWM']
        x = range(len(bench_list))
        width = 0.35

        avg_on_down = []
        avg_on_up = []
        for bench in bench_list:
            bench_down = out[out[f'{bench}_down']]
            bench_up = out[~out[f'{bench}_down']]
            avg_on_down.append(bench_down['Portfolio_ret'].mean() * 100)
            avg_on_up.append(bench_up['Portfolio_ret'].mean() * 100)

        ax.bar([i - width/2 for i in x], avg_on_down, width, label=f'Portfolio on Down Days', color='#E63946', alpha=0.8)
        ax.bar([i + width/2 for i in x], avg_on_up, width, label=f'Portfolio on Up Days', color='#06A77D', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Average Daily Return (%)', fontsize=12)
        ax.set_title('Portfolio Average Returns on Benchmark Up vs Down Days', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b} Days' for b in bench_list], fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (d, u) in enumerate(zip(avg_on_down, avg_on_up)):
            ax.text(i - width/2, d - 0.02 if d < 0 else d + 0.02, f'{d:.2f}%',
                   ha='center', va='top' if d < 0 else 'bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, u + 0.02, f'{u:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '5_avg_returns_up_down.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 6. Drawdown Chart
        dd_data = pd.DataFrame(index=out.index)
        for col in ["Portfolio_ret", "SPY_ret", "QQQ_ret", "DIA_ret", "IWM_ret"]:
            cum_ret = (1 + out[col]).cumprod()
            running_max = cum_ret.expanding().max()
            dd_data[col.replace('_ret', '')] = (cum_ret - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=(14, 8))
        for col in dd_data.columns:
            ax.plot(dd_data.index, dd_data[col], label=col, linewidth=2, color=COLORS.get(col, 'gray'))
        ax.fill_between(dd_data.index, 0, dd_data['Portfolio'], alpha=0.3, color=COLORS['Portfolio'])
        ax.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '6_drawdown.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 7. Portfolio Composition
        weights_top = weights_used.sort_values(ascending=False).head(10)
        weights_other = weights_used.sort_values(ascending=False)[10:].sum()

        fig, ax = plt.subplots(figsize=(10, 10))
        if weights_other > 0:
            labels = list(weights_top.index) + ['Other']
            sizes = list(weights_top.values) + [weights_other]
        else:
            labels = list(weights_top.index)
            sizes = list(weights_top.values)

        colors_pie = plt.cm.Set3(range(len(labels)))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors_pie, textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        ax.set_title('Portfolio Composition (Top 10 Holdings)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_DIR, '7_portfolio_composition.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved to {CHART_DIR}")

    except ImportError:
        print("matplotlib not installed. Skipping visualizations.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    # ---- Generate HTML Dashboard ----
    try:
        from datetime import datetime

        # Read metrics
        capture_df = pd.read_csv(os.path.join(OUTDIR, "capture_stats.csv"))
        summary_df = pd.read_csv(os.path.join(OUTDIR, "summary_stats.csv"))

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Analysis Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 30px;
        }}
        .metadata {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            border-left: 4px solid #2E86AB;
        }}
        .metadata p {{
            margin: 5px 0;
            color: #555;
        }}
        h2 {{
            color: #2E86AB;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2E86AB;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: #2E86AB;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2E86AB;
        }}
        .good {{
            color: #06A77D;
            font-weight: bold;
        }}
        .moderate {{
            color: #F18F01;
            font-weight: bold;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-top: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-container h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Portfolio Analysis Dashboard</h1>
        <p class="subtitle">Downside Protection Strategy Analysis</p>

        <div class="metadata">
            <p><strong>Analysis Period:</strong> {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}</p>
            <p><strong>Portfolio Positions:</strong> {len(weights_used)} equities</p>
            <p><strong>Benchmarks:</strong> SPY (S&P 500), QQQ (Nasdaq 100), DIA (DJIA), IWM (Russell 2000)</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <h2>🎯 Capture Ratios</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add capture ratios to table
        for _, row in capture_df[capture_df['Metric'].str.contains('capture')].iterrows():
            metric = row['Metric']
            value = row['Value'] * 100

            if 'Downside' in metric:
                interpretation = "Lower is better - less downside captured"
                color_class = 'good' if value < 70 else 'moderate'
            else:
                interpretation = "Higher is better - more upside captured"
                color_class = 'good' if value > 80 else 'moderate'

            html_content += f"""
                <tr>
                    <td>{metric}</td>
                    <td class="{color_class}">{value:.1f}%</td>
                    <td>{interpretation}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>🛡️ Hit Rates on Down Days</h2>
        <p style="margin-bottom: 15px; color: #666;">Percentage of days portfolio is non-negative when benchmark is down</p>
        <table>
            <thead>
                <tr>
                    <th>Benchmark Down Days</th>
                    <th>Portfolio Hit Rate</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add hit rates to table
        for _, row in capture_df[capture_df['Metric'].str.contains('Hit-rate')].iterrows():
            metric = row['Metric']
            value = row['Value'] * 100
            benchmark = metric.split('on ')[1].split('-down')[0]
            color_class = 'good' if value > 35 else 'moderate'
            html_content += f"""
                <tr>
                    <td>{benchmark} down days</td>
                    <td class="{color_class}">{value:.1f}%</td>
                    <td>Portfolio positive {value:.1f}% of the time</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>📈 Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Market Condition</th>
                    <th>Days</th>
                    <th>Portfolio Hit Rate</th>
                    <th>Portfolio Avg Return</th>
                    <th>Portfolio Worst Day</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add summary stats
        for _, row in summary_df.iterrows():
            category = row['Category'].replace('_', ' ').title()
            days = int(row['Days'])
            hit_rate = row['Portfolio_hit_rate_(>=0%)'] * 100
            avg_return = row['Portfolio_avg'] * 100
            worst = row['Portfolio_worst'] * 100

            html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{days}</td>
                    <td class="metric-value">{hit_rate:.1f}%</td>
                    <td class="metric-value">{avg_return:+.2f}%</td>
                    <td class="metric-value">{worst:+.2f}%</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>📊 Visualizations</h2>

        <div class="charts-grid">
            <div class="chart-container">
                <h3>1. Cumulative Returns</h3>
                <img src="charts/1_cumulative_returns.png" alt="Cumulative Returns">
            </div>

            <div class="chart-container">
                <h3>2. Capture Ratios Comparison</h3>
                <img src="charts/2_capture_ratios.png" alt="Capture Ratios">
            </div>

            <div class="chart-container">
                <h3>3. Hit Rates on Down Days</h3>
                <img src="charts/3_hit_rates.png" alt="Hit Rates">
            </div>

            <div class="chart-container">
                <h3>4. Return Distribution Comparison</h3>
                <img src="charts/4_return_distributions.png" alt="Return Distributions">
            </div>

            <div class="chart-container">
                <h3>5. Average Returns: Up vs Down Days</h3>
                <img src="charts/5_avg_returns_up_down.png" alt="Average Returns">
            </div>

            <div class="chart-container">
                <h3>6. Drawdown Comparison</h3>
                <img src="charts/6_drawdown.png" alt="Drawdown">
            </div>

            <div class="chart-container">
                <h3>7. Portfolio Composition</h3>
                <img src="charts/7_portfolio_composition.png" alt="Portfolio Composition">
            </div>
        </div>

        <div class="footer">
            <p>Portfolio Analysis Tool | Generated from portfolio_analysis.py</p>
            <p>Analysis Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}</p>
        </div>
    </div>
</body>
</html>
"""

        # Write HTML dashboard
        html_path = os.path.join(OUTDIR, "index.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML dashboard saved to {html_path}")

    except Exception as e:
        print(f"Error generating HTML dashboard: {e}")

    print("Success. Files in", OUTDIR)

if __name__ == "__main__":
    main()
