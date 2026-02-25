"""
AQR Fund Comparison Tool

Downloads daily prices via yfinance (using ISINs) for funds in tickerlist.csv,
generates an HTML report with:
- Performance table (1M, 3M, 1Y, Max returns)
- Indexed performance chart (base = 1.0)
- Correlation matrix (daily returns)
- Relative performance of AQR funds vs FTSE All World
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import yfinance as yf


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def read_tickers(csv_path):
    """Parse tickerlist.csv -> list of (isin, name, bbg_ticker)."""
    tickers = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip("\t")
            if not line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 3:
                tickers.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                # Handle "Name, TICKER Equity" format
                sub = parts[1].rsplit(",", 1)
                if len(sub) == 2:
                    tickers.append((parts[0], sub[0].strip(), sub[1].strip()))
    return tickers


# Short display names for charts
SHORT_NAMES = {
    "APEX": "AQR Apex",
    "ADAPTIVE EQUITY MARKET NEUTRAL": "AQR Eq Mkt Neutral",
    "ALTERNATIVE TRENDS": "AQR Alt Trends",
    "STYLE PREMIA": "AQR Style Premia",
    "MANAGED FUTURES": "AQR Managed Futures",
    "Delphi Long-Short": "AQR Delphi L/S",
    "Vanguard FTSE All World": "FTSE All World",
}


def short_name(full_name):
    for key, val in SHORT_NAMES.items():
        if key.lower() in full_name.lower():
            return val
    return full_name[:30]


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_prices(tickers):
    """Download daily close prices via yfinance using ISINs."""
    all_prices = {}
    for isin, name, _ in tickers:
        print(f"  {isin}  {short_name(name):25s}", end="", flush=True)
        try:
            tk = yf.Ticker(isin)
            hist = tk.history(period="max")
            if not hist.empty and "Close" in hist.columns:
                series = hist["Close"].dropna()
                series.index = series.index.tz_localize(None)  # remove timezone
                all_prices[name] = series
                print(f"  {len(series):5d} days  ({series.index.min().date()} -> {series.index.max().date()})")
            else:
                print("  NO DATA")
        except Exception as e:
            print(f"  ERROR: {e}")
    return pd.DataFrame(all_prices)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def compute_returns_table(prices):
    """Compute 1M, 3M, 1Y, Max total and annualized returns."""
    latest = prices.index.max()
    cutoffs = {
        "1M": latest - pd.DateOffset(months=1),
        "3M": latest - pd.DateOffset(months=3),
        "1Y": latest - pd.DateOffset(years=1),
    }

    rows = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            continue
        row = {"Start": s.index[0].strftime("%Y-%m-%d")}

        for label, cutoff in cutoffs.items():
            sub = s[s.index >= cutoff]
            if len(sub) >= 2:
                row[label] = sub.iloc[-1] / sub.iloc[0] - 1
            else:
                row[label] = None

        # Max total and annualized
        row["Max"] = s.iloc[-1] / s.iloc[0] - 1
        years = (s.index[-1] - s.index[0]).days / 365.25
        if years > 0:
            row["Max (p.a.)"] = (1 + row["Max"]) ** (1 / years) - 1

        # Annualized volatility and Sharpe ratio
        daily_rets = s.pct_change().dropna()
        if len(daily_rets) > 20:
            ann_vol = daily_rets.std() * np.sqrt(252)
            row["Vol (ann.)"] = ann_vol
            if ann_vol > 0 and years > 0:
                row["Sharpe"] = row["Max (p.a.)"] / ann_vol

        rows[col] = row

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def performance_chart(prices):
    """All funds indexed to 1.0 from common start date."""
    starts = prices.apply(lambda s: s.dropna().index.min())
    common_start = starts.max()
    trimmed = prices[prices.index >= common_start].copy()

    fig = go.Figure()
    for i, col in enumerate(trimmed.columns):
        s = trimmed[col].dropna()
        if s.empty:
            continue
        indexed = s / s.iloc[0]
        fig.add_trace(go.Scatter(
            x=indexed.index, y=indexed.values,
            mode="lines", name=short_name(col),
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig.update_layout(
        title=f"Indexed Performance (common start: {common_start.date()})",
        yaxis_title="Growth of 1.0",
        template="plotly_white", height=520,
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def rolling_correlation_chart(prices, aqr_names, benchmark_name, window=60):
    """60-day rolling correlation of AQR funds vs benchmark."""
    bench_rets = prices[benchmark_name].pct_change()
    fig = go.Figure()
    for i, name in enumerate(aqr_names):
        fund_rets = prices[name].pct_change()
        combined = pd.concat(
            [fund_rets.rename("fund"), bench_rets.rename("bench")],
            axis=1, sort=True,
        ).dropna()
        if len(combined) < window:
            continue
        rolling_corr = combined["fund"].rolling(window).corr(combined["bench"]).dropna()
        # Only show last year
        one_year_ago = rolling_corr.index.max() - pd.DateOffset(months=6)
        rolling_corr = rolling_corr[rolling_corr.index >= one_year_ago]
        if rolling_corr.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr.values,
            mode="lines", name=short_name(name),
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"{window}-Day Rolling Correlation with {short_name(benchmark_name)}",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        template="plotly_white", height=520,
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def return_dendrogram(prices):
    """Hierarchical clustering dendrogram based on correlation distance."""
    rets = prices.pct_change().dropna(how="all")
    corr = rets.corr()
    # Correlation distance: d = sqrt(0.5 * (1 - corr))
    dist = np.sqrt(0.5 * (1 - corr)).values.copy()
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    link = linkage(condensed, method="ward")
    labels = [short_name(c) for c in corr.columns]

    fig = ff.create_dendrogram(
        dist,
        labels=labels,
        linkagefun=lambda x: link,
        color_threshold=0.7 * max(link[:, 2]),
    )
    fig.update_layout(
        title="Fund Clustering (Correlation Distance, Ward Linkage)",
        yaxis_title="Distance",
        template="plotly_white", height=450,
        xaxis=dict(tickangle=-30),
    )
    return fig


def correlation_heatmap(prices):
    """Correlation matrix heatmap of daily returns."""
    rets = prices.pct_change().dropna(how="all")
    corr = rets.corr()
    labels = [short_name(c) for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Correlation Matrix (Daily Returns)",
        template="plotly_white", height=550, width=750,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def returns_table_html(returns_table):
    """Render performance table as styled HTML."""
    display_cols = ["Start", "1M", "3M", "1Y", "Max", "Max (p.a.)", "Vol (ann.)", "Sharpe"]
    col_labels = {
        "Start": "Start Date", "1M": "1 Month", "3M": "3 Months",
        "1Y": "1 Year", "Max": "Max (total)", "Max (p.a.)": "Max (p.a.)",
        "Vol (ann.)": "Vol (ann.)", "Sharpe": "Sharpe",
    }
    cols = [c for c in display_cols if c in returns_table.columns]

    header = "<th>Fund</th>" + "".join(
        f"<th>{col_labels.get(c, c)}</th>" for c in cols
    )
    body = ""
    for fund, row in returns_table.iterrows():
        cells = f"<td class='fund-name'>{short_name(fund)}</td>"
        for c in cols:
            val = row.get(c)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells += "<td>—</td>"
            elif c == "Start":
                cells += f"<td>{val}</td>"
            elif c == "Sharpe":
                cells += f"<td>{val:.2f}</td>"
            elif c == "Vol (ann.)":
                cells += f"<td>{val * 100:.1f}%</td>"
            else:
                pct = val * 100
                cls = "pos" if pct >= 0 else "neg"
                cells += f"<td class='{cls}'>{pct:+.2f}%</td>"
        body += f"<tr>{cells}</tr>\n"

    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def generate_report(prices, returns_table, tickers):
    aqr_names = [name for isin, name, _ in tickers if isin.startswith("LU")]
    benchmark_name = next(
        (name for isin, name, _ in tickers if isin.startswith("IE")), None
    )

    fig1 = performance_chart(prices)
    fig2 = rolling_correlation_chart(prices, aqr_names, benchmark_name) if benchmark_name else None
    fig3 = correlation_heatmap(prices)
    fig4 = return_dendrogram(prices)

    tbl = returns_table_html(returns_table)
    c1 = fig1.to_html(full_html=False, include_plotlyjs=False)
    c2 = fig2.to_html(full_html=False, include_plotlyjs=False) if fig2 else ""
    c3 = fig3.to_html(full_html=False, include_plotlyjs=False)
    c4 = fig4.to_html(full_html=False, include_plotlyjs=False)

    rolling_corr_section = ""
    if c2:
        rolling_corr_section = (
            "<h2>Rolling Correlation with FTSE All World</h2>\n"
            "<p class='note'>60-day rolling correlation of daily returns. "
            "Values near 0 indicate low correlation (diversification benefit).</p>\n"
            f"<div class='chart-box'>{c2}</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AQR Fund Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 40px auto; max-width: 1100px; background: #f8f9fa; color: #333;
  }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 40px; }}
  .subtitle {{ color: #666; font-size: 14px; }}
  .note {{ color: #888; font-size: 13px; margin-top: -5px; }}
  table {{
    border-collapse: collapse; width: 100%; margin: 20px 0; background: #fff;
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.12);
  }}
  th {{
    background: #1a1a2e; color: #fff; padding: 12px 16px;
    text-align: center; font-size: 13px;
  }}
  td {{ padding: 10px 16px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }}
  .fund-name {{ font-weight: 600; text-align: left !important; white-space: nowrap; }}
  .pos {{ color: #2ca02c; font-weight: 600; }}
  .neg {{ color: #d62728; font-weight: 600; }}
  tr:hover {{ background: #f5f5f5; }}
  .chart-box {{
    background: #fff; border-radius: 8px; padding: 15px; margin: 20px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.12);
  }}
</style>
</head>
<body>
<h1>AQR Fund Comparison</h1>
<p class="subtitle">Generated {dt.datetime.now().strftime("%Y-%m-%d %H:%M")} | Data source: Yahoo Finance</p>

<h2>Performance Summary</h2>
{tbl}

<h2>Indexed Performance</h2>
<div class="chart-box">{c1}</div>

<h2>Correlation Matrix</h2>
<div class="chart-box">{c3}</div>

<h2>Fund Clustering</h2>
<p class="note">Hierarchical clustering using correlation distance. Funds that merge at lower heights have more similar return profiles.</p>
<div class="chart-box">{c4}</div>

{rolling_corr_section}

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_dir, "tickerlist.csv")
    output_dir = os.path.join(project_dir, "public")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "index.html")

    print("=" * 60)
    print("  AQR Fund Comparison")
    print("=" * 60)

    tickers = read_tickers(csv_path)
    print(f"\n  {len(tickers)} funds:\n")

    # Download via yfinance using ISINs
    prices = download_prices(tickers)

    if prices.empty:
        print("\nERROR: No data downloaded")
        sys.exit(1)

    # Forward-fill gaps (different trading calendars)
    prices = prices.ffill()
    print(f"\n  Combined: {prices.shape[1]} funds, {prices.shape[0]} trading days")

    # Performance table
    returns_table = compute_returns_table(prices)
    print(f"\n{returns_table.to_string()}\n")

    # Generate HTML report
    html = generate_report(prices, returns_table, tickers)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
