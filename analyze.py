#!/usr/bin/env python3
import os, sys, time, re
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please install dependencies first:\n  pip install yfinance pandas\n" + str(e))

# ---- Config ----
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365*5)

BENCH = ["SPY", "QQQ", "DIA", "IWM"]  # SPY=S&P500, QQQ=Nasdaq100, DIA=DJIA, IWM=Russell2000

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

def calculate_drawdowns(cumulative_returns: pd.Series) -> pd.DataFrame:
    """Calculate all drawdown periods for a cumulative return series."""
    cum_ret = cumulative_returns + 1
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max

    # Find drawdown periods (consecutive negative drawdowns)
    is_underwater = drawdown < 0

    drawdowns = []
    in_drawdown = False
    peak_idx = None
    peak_val = None
    trough_idx = None
    trough_val = None

    for i, (idx, val) in enumerate(zip(cum_ret.index, cum_ret.values)):
        if not in_drawdown and is_underwater.iloc[i]:
            # Start of drawdown
            in_drawdown = True
            peak_idx = cum_ret.index[i-1] if i > 0 else idx
            peak_val = running_max.iloc[i]
            trough_idx = idx
            trough_val = val
        elif in_drawdown and is_underwater.iloc[i]:
            # Continue drawdown - update trough if deeper
            if val < trough_val:
                trough_idx = idx
                trough_val = val
        elif in_drawdown and not is_underwater.iloc[i]:
            # End of drawdown (recovery to previous peak)
            recovery_idx = idx
            duration = (trough_idx - peak_idx).days if hasattr(trough_idx - peak_idx, 'days') else len(cum_ret[peak_idx:trough_idx])
            recovery_time = (recovery_idx - trough_idx).days if hasattr(recovery_idx - trough_idx, 'days') else len(cum_ret[trough_idx:recovery_idx])

            drawdowns.append({
                'peak_date': peak_idx,
                'trough_date': trough_idx,
                'recovery_date': recovery_idx,
                'peak_value': peak_val,
                'trough_value': trough_val,
                'drawdown_pct': (trough_val - peak_val) / peak_val,
                'duration_days': duration,
                'recovery_days': recovery_time,
                'total_underwater_days': duration + recovery_time
            })

            in_drawdown = False

    # Handle ongoing drawdown (not recovered yet)
    if in_drawdown:
        duration = (trough_idx - peak_idx).days if hasattr(trough_idx - peak_idx, 'days') else len(cum_ret[peak_idx:trough_idx])
        drawdowns.append({
            'peak_date': peak_idx,
            'trough_date': trough_idx,
            'recovery_date': None,
            'peak_value': peak_val,
            'trough_value': trough_val,
            'drawdown_pct': (trough_val - peak_val) / peak_val,
            'duration_days': duration,
            'recovery_days': None,
            'total_underwater_days': None
        })

    return pd.DataFrame(drawdowns) if drawdowns else pd.DataFrame()

def calculate_resilience_metrics(cum_returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate comprehensive resilience metrics for portfolio and benchmarks."""

    assets = cum_returns_df.columns.tolist()

    # 1. Drawdown stats per asset
    drawdown_stats = []
    # 2. Resilience stats per asset
    resilience_stats = []

    all_drawdowns = {}  # Store all drawdowns for later analysis

    for asset in assets:
        cum_ret = cum_returns_df[asset]
        dds = calculate_drawdowns(cum_ret)
        all_drawdowns[asset] = dds

        if dds.empty:
            # No drawdowns - perfect performance
            drawdown_stats.append({
                'Asset': asset.replace('_ret', ''),
                'Max_Drawdown_Pct': 0.0,
                'Max_Drawdown_Date': None,
                'Max_DD_Duration_Days': 0,
                'Num_Drawdowns_Over_10pct': 0,
                'Avg_Drawdown_Pct': 0.0
            })
            resilience_stats.append({
                'Asset': asset.replace('_ret', ''),
                'Avg_Recovery_Days': 0.0,
                'Avg_Recovery_Speed_Pct_Per_Day': 0.0,
                'Longest_Underwater_Period_Days': 0
            })
            continue

        # Max drawdown
        max_dd_idx = dds['drawdown_pct'].idxmin()
        max_dd = dds.loc[max_dd_idx]

        # Drawdowns > 10%
        significant_dds = dds[dds['drawdown_pct'] < -0.10]

        # Average drawdown
        avg_dd = dds['drawdown_pct'].mean()

        drawdown_stats.append({
            'Asset': asset.replace('_ret', ''),
            'Max_Drawdown_Pct': max_dd['drawdown_pct'],
            'Max_Drawdown_Date': max_dd['trough_date'],
            'Max_DD_Duration_Days': max_dd['duration_days'],
            'Num_Drawdowns_Over_10pct': len(significant_dds),
            'Avg_Drawdown_Pct': avg_dd
        })

        # Recovery metrics (only for completed drawdowns)
        completed_dds = dds[dds['recovery_date'].notna()]

        if not completed_dds.empty:
            avg_recovery_days = completed_dds['recovery_days'].mean()
            # Recovery speed = drawdown depth / recovery days
            completed_dds_copy = completed_dds.copy()
            completed_dds_copy['recovery_speed'] = completed_dds_copy['drawdown_pct'].abs() / completed_dds_copy['recovery_days']
            avg_recovery_speed = completed_dds_copy['recovery_speed'].mean()
        else:
            avg_recovery_days = float('nan')
            avg_recovery_speed = float('nan')

        # Longest underwater period
        longest_underwater = dds['total_underwater_days'].max() if not dds['total_underwater_days'].isna().all() else 0
        if pd.isna(longest_underwater):
            longest_underwater = 0

        resilience_stats.append({
            'Asset': asset.replace('_ret', ''),
            'Avg_Recovery_Days': avg_recovery_days,
            'Avg_Recovery_Speed_Pct_Per_Day': avg_recovery_speed,
            'Longest_Underwater_Period_Days': int(longest_underwater)
        })

    drawdown_df = pd.DataFrame(drawdown_stats)
    resilience_df = pd.DataFrame(resilience_stats)

    # 3. Comparative resilience (Portfolio vs each benchmark)
    portfolio_resilience = resilience_df[resilience_df['Asset'] == 'Portfolio'].iloc[0] if 'Portfolio' in resilience_df['Asset'].values else None
    portfolio_drawdown = drawdown_df[drawdown_df['Asset'] == 'Portfolio'].iloc[0] if 'Portfolio' in drawdown_df['Asset'].values else None

    comparison_stats = []

    if portfolio_resilience is not None and portfolio_drawdown is not None:
        benchmarks = ['SPY', 'QQQ', 'DIA', 'IWM']
        for bench in benchmarks:
            bench_resilience = resilience_df[resilience_df['Asset'] == bench]
            bench_drawdown = drawdown_df[drawdown_df['Asset'] == bench]

            if not bench_resilience.empty and not bench_drawdown.empty:
                bench_r = bench_resilience.iloc[0]
                bench_d = bench_drawdown.iloc[0]

                # Calculate ratios (Portfolio / Benchmark)
                # Lower recovery time ratio = better (portfolio recovers faster)
                # Higher recovery speed ratio = better (portfolio recovers faster per day)
                # Lower max drawdown ratio = better (portfolio has shallower drawdowns)

                recovery_time_ratio = portfolio_resilience['Avg_Recovery_Days'] / bench_r['Avg_Recovery_Days'] if bench_r['Avg_Recovery_Days'] > 0 and not pd.isna(bench_r['Avg_Recovery_Days']) else float('nan')
                recovery_speed_ratio = portfolio_resilience['Avg_Recovery_Speed_Pct_Per_Day'] / bench_r['Avg_Recovery_Speed_Pct_Per_Day'] if bench_r['Avg_Recovery_Speed_Pct_Per_Day'] > 0 and not pd.isna(bench_r['Avg_Recovery_Speed_Pct_Per_Day']) else float('nan')
                max_dd_ratio = abs(portfolio_drawdown['Max_Drawdown_Pct']) / abs(bench_d['Max_Drawdown_Pct']) if bench_d['Max_Drawdown_Pct'] != 0 else float('nan')

                comparison_stats.append({
                    'Benchmark': bench,
                    'Recovery_Time_Ratio': recovery_time_ratio,
                    'Recovery_Speed_Ratio': recovery_speed_ratio,
                    'Max_Drawdown_Ratio': max_dd_ratio
                })

    comparison_df = pd.DataFrame(comparison_stats)

    return drawdown_df, resilience_df, comparison_df

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <portfolio.csv>")
        print("\nExample:")
        print("  python analyze.py portfolio.csv  # Creates portfolio/ directory")
        print("  python analyze.py faang.csv      # Creates faang/ directory")
        sys.exit(1)

    weights_csv = sys.argv[1]
    if not os.path.exists(weights_csv):
        raise SystemExit(f"Error: {weights_csv} not found")

    # Derive output directory from CSV filename
    # e.g., "faang.csv" -> "faang", "portfolio.csv" -> "portfolio"
    outdir = os.path.splitext(os.path.basename(weights_csv))[0]
    os.makedirs(outdir, exist_ok=True)

    # Load initial weights and build download list
    weights = load_weights(weights_csv)
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
        with open(os.path.join(outdir, "dropped_tickers.txt"), "w") as f:
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
    weights_used.to_csv(os.path.join(outdir, "weights_used.csv"), header=["Weight"])

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
        weights_used.to_csv(os.path.join(outdir, "weights_used.csv"), header=["Weight"])

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
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"), index_label="Category")
    capture.to_csv(os.path.join(outdir, "capture_stats.csv"), index=False)

    cum = (1 + out[["Portfolio_ret","SPY_ret","QQQ_ret","DIA_ret","IWM_ret"]]).cumprod() - 1
    cum.to_csv(os.path.join(outdir, "cumulative_returns.csv"), index_label="Date")

    # Calculate resilience metrics
    print("\nCalculating resilience metrics...")
    drawdown_df, resilience_df, comparison_df = calculate_resilience_metrics(cum)

    # Export resilience CSVs
    drawdown_df.to_csv(os.path.join(outdir, "drawdown_stats.csv"), index=False)
    resilience_df.to_csv(os.path.join(outdir, "resilience_stats.csv"), index=False)
    comparison_df.to_csv(os.path.join(outdir, "recovery_comparison.csv"), index=False)

    # Print resilience summary to terminal
    print("\n" + "="*60)
    print("RESILIENCE ANALYSIS")
    print("="*60)

    port_dd = drawdown_df[drawdown_df['Asset'] == 'Portfolio']
    port_res = resilience_df[resilience_df['Asset'] == 'Portfolio']

    if not port_dd.empty and not port_res.empty:
        max_dd = port_dd.iloc[0]['Max_Drawdown_Pct'] * 100
        avg_recovery = port_res.iloc[0]['Avg_Recovery_Days']
        recovery_speed = port_res.iloc[0]['Avg_Recovery_Speed_Pct_Per_Day'] * 100

        print(f"Portfolio Max Drawdown: {max_dd:.2f}%")
        print(f"Avg Recovery Time: {avg_recovery:.0f} days")
        print(f"Recovery Speed: {recovery_speed:.3f}% per day")

        # Show comparisons vs SPY
        spy_comparison = comparison_df[comparison_df['Benchmark'] == 'SPY']
        if not spy_comparison.empty:
            spy_comp = spy_comparison.iloc[0]
            spy_dd = drawdown_df[drawdown_df['Asset'] == 'SPY'].iloc[0]
            spy_res = resilience_df[resilience_df['Asset'] == 'SPY'].iloc[0]

            print(f"\nVs SPY:")
            print(f"  Max Drawdown: Portfolio {max_dd:.2f}% vs SPY {spy_dd['Max_Drawdown_Pct']*100:.2f}% ({spy_comp['Max_Drawdown_Ratio']:.2f}x)")
            if not pd.isna(spy_comp['Recovery_Time_Ratio']):
                print(f"  Recovery Time: {spy_comp['Recovery_Time_Ratio']:.2f}x SPY ({avg_recovery:.0f} vs {spy_res['Avg_Recovery_Days']:.0f} days)")
            if not pd.isna(spy_comp['Recovery_Speed_Ratio']):
                print(f"  Recovery Speed: {spy_comp['Recovery_Speed_Ratio']:.2f}x SPY")

    print("\n" + "="*60)
    print("CORRELATION METRICS")
    print("="*60)

    # Show key correlation metrics
    spy_down = out.loc[out["SPY_down"]]
    if len(spy_down) > 0:
        port_avg_spy_down = spy_down["Portfolio_ret"].mean() * 100
        spy_avg_spy_down = spy_down["SPY_ret"].mean() * 100
        downside_capture = (port_avg_spy_down / spy_avg_spy_down) * 100 if spy_avg_spy_down != 0 else float('nan')
        print(f"Downside Capture vs SPY: {downside_capture:.1f}%")

    spy_up = out.loc[~out["SPY_down"]]
    if len(spy_up) > 0:
        port_avg_spy_up = spy_up["Portfolio_ret"].mean() * 100
        spy_avg_spy_up = spy_up["SPY_ret"].mean() * 100
        upside_capture = (port_avg_spy_up / spy_avg_spy_up) * 100 if spy_avg_spy_up != 0 else float('nan')
        print(f"Upside Capture vs SPY: {upside_capture:.1f}%")

    if len(spy_down) > 0:
        hit_rate = (spy_down["Portfolio_ret"] >= 0).mean() * 100
        print(f"Hit Rate on SPY Down Days: {hit_rate:.1f}%")

    print("\n" + "="*60)
    print(f"Analysis complete. CSV data saved to {outdir}/")
    print(f"Run: python visualize.py {outdir}")
    print("="*60)

if __name__ == "__main__":
    main()
