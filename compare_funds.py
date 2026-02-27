"""
AQR Fund Comparison Tool

Downloads daily prices via yfinance (using ISINs) for funds in tickerlist.csv
and ETFs in etfs.csv, generates an HTML report with two tabs:

Tab 1 – AQR Funds:
- Performance table (1M, 3M, 1Y, Max returns)
- Indexed performance chart
- Correlation matrix (daily returns)
- Hierarchical clustering dendrogram
- Rolling correlation with FTSE All World
- Stress test (worst weeks)
- Portfolio optimization (≥50% FTSE constraint)

Tab 2 – Global Equity ETFs:
- Same sections as Tab 1
- Correlation relative to FTSE All World
- Portfolio optimization with no minimum-weight constraint
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
import yfinance as yf


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def read_tickers(csv_path):
    """Parse a ticker CSV -> list of (isin, name, bbg_ticker)."""
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
                sub = parts[1].rsplit(",", 1)
                if len(sub) == 2:
                    tickers.append((parts[0], sub[0].strip(), sub[1].strip()))
    return tickers


# Short display names for charts
SHORT_NAMES = {
    # AQR funds
    "APEX": "AQR Apex",
    "ADAPTIVE EQUITY MARKET NEUTRAL": "AQR Eq Mkt Neutral",
    "ALTERNATIVE TRENDS": "AQR Alt Trends",
    "STYLE PREMIA": "AQR Style Premia",
    "MANAGED FUTURES": "AQR Managed Futures",
    "Delphi Long-Short": "AQR Delphi L/S",
    # Benchmark
    "Vanguard FTSE All World": "FTSE All World",
    # Global equity ETFs
    "iShares MSCI World": "MSCI World",
    "Xtrackers MSCI USA": "MSCI USA",
    "iShares MSCI Europe": "MSCI Europe",
    "iShares MSCI Emerging Markets": "EM",
    "ishares MSCI Japan": "MSCI Japan",
    "iShares Pacific": "Pacific ex-JP",
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
                series.index = series.index.tz_localize(None)
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

        row["Max"] = s.iloc[-1] / s.iloc[0] - 1
        years = (s.index[-1] - s.index[0]).days / 365.25
        if years > 0:
            row["Max (p.a.)"] = (1 + row["Max"]) ** (1 / years) - 1

        daily_rets = s.pct_change().dropna()
        if len(daily_rets) > 20:
            ann_vol = daily_rets.std() * np.sqrt(252)
            row["Vol (ann.)"] = ann_vol

            one_year_ago = latest - pd.DateOffset(years=1)
            rets_1y = daily_rets[daily_rets.index >= one_year_ago]
            if len(rets_1y) > 20:
                ann_ret_1y = rets_1y.mean() * 252
                ann_vol_1y = rets_1y.std() * np.sqrt(252)
                if ann_vol_1y > 0:
                    row["Sharpe (1Y)"] = ann_ret_1y / ann_vol_1y

            prices_1y = s[s.index >= one_year_ago]
            if len(prices_1y) > 20:
                running_max = prices_1y.cummax()
                drawdown_pct = ((prices_1y - running_max) / running_max) * 100
                ulcer_index = np.sqrt((drawdown_pct ** 2).mean())
                if ulcer_index > 0:
                    ann_ret_1y_ul = (prices_1y.iloc[-1] / prices_1y.iloc[0] - 1)
                    years_1y = (prices_1y.index[-1] - prices_1y.index[0]).days / 365.25
                    if years_1y > 0:
                        ann_ret_1y_ul = (1 + ann_ret_1y_ul) ** (1 / years_1y) - 1
                    row["UPI (1Y)"] = (ann_ret_1y_ul * 100) / ulcer_index

        rows[col] = row

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def performance_chart(prices, chart_id_prefix="perf"):
    """All funds indexed to 1.0 with timeframe selector buttons."""
    latest = prices.index.max()
    starts = prices.apply(lambda s: s.dropna().index.min())
    common_start = starts.max()

    timeframes = {
        "1M": latest - pd.DateOffset(months=1),
        "3M": latest - pd.DateOffset(months=3),
        "1Y": latest - pd.DateOffset(years=1),
        "All": common_start,
    }

    fig = go.Figure()
    buttons = []
    n_funds = len(prices.columns)
    for tf_idx, (tf_label, tf_start) in enumerate(timeframes.items()):
        start = max(tf_start, common_start)
        trimmed = prices[prices.index >= start].copy()
        for i, col in enumerate(trimmed.columns):
            s = trimmed[col].dropna()
            if s.empty:
                continue
            indexed = s / s.iloc[0]
            fig.add_trace(go.Scatter(
                x=indexed.index, y=indexed.values,
                mode="lines", name=short_name(col),
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                visible=(tf_label == "All"),
                showlegend=(tf_label == "All"),
            ))

        vis = [False] * (n_funds * len(timeframes))
        for j in range(n_funds):
            vis[tf_idx * n_funds + j] = True
        buttons.append(dict(
            label=tf_label,
            method="update",
            args=[
                {"visible": vis},
                {"title": f"Indexed Performance ({tf_label})"},
            ],
        ))

    fig.update_layout(
        title="Indexed Performance (All)",
        yaxis_title="Growth of 1.0",
        template="plotly_white", height=520,
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
        hovermode="x unified",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.0, xanchor="right",
            y=1.18, yanchor="top",
            buttons=buttons,
            bgcolor="#e8e8e8",
            font=dict(size=12),
        )],
    )
    return fig


def rolling_correlation_chart(prices, fund_names, benchmark_name, window=60):
    """60-day rolling correlation of funds vs benchmark."""
    bench_rets = prices[benchmark_name].pct_change()
    fig = go.Figure()
    for i, name in enumerate(fund_names):
        fund_rets = prices[name].pct_change()
        combined = pd.concat(
            [fund_rets.rename("fund"), bench_rets.rename("bench")],
            axis=1, sort=True,
        ).dropna()
        if len(combined) < window:
            continue
        rolling_corr = combined["fund"].rolling(window).corr(combined["bench"]).dropna()
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


def stress_test_table(prices, benchmark_name, n_worst=5):
    """Find the worst n weeks for the benchmark and show all funds' returns."""
    weekly = prices.resample("W-FRI").last()
    weekly_rets = weekly.pct_change().dropna(how="all")

    if benchmark_name not in weekly_rets.columns:
        return None

    bench_weekly = weekly_rets[benchmark_name].dropna()
    bench_weekly = bench_weekly[bench_weekly.index >= "2024-01-01"]
    worst_weeks = bench_weekly.nsmallest(n_worst)

    rows = []
    for date, bench_ret in worst_weeks.items():
        row = {"Week ending": date.strftime("%Y-%m-%d")}
        for col in weekly_rets.columns:
            val = weekly_rets.loc[date, col] if date in weekly_rets.index else None
            row[short_name(col)] = val
        rows.append(row)

    return pd.DataFrame(rows)


def stress_test_html(stress_df, benchmark_short):
    """Render the stress test table as styled HTML."""
    if stress_df is None or stress_df.empty:
        return ""

    fund_cols = [c for c in stress_df.columns if c != "Week ending"]
    header = "<th>Week ending</th>" + "".join(f"<th>{c}</th>" for c in fund_cols)

    body = ""
    for _, row in stress_df.iterrows():
        cells = f"<td style='font-weight:600'>{row['Week ending']}</td>"
        for c in fund_cols:
            val = row[c]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells += "<td>—</td>"
            else:
                pct = val * 100
                cls = "pos" if pct >= 0 else "neg"
                cells += f"<td class='{cls}'>{pct:+.2f}%</td>"
        body += f"<tr>{cells}</tr>\n"

    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


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


def correlation_vs_benchmark_chart(prices, fund_names, benchmark_name):
    """Bar chart of full-period correlation of each fund vs the benchmark."""
    rets = prices.pct_change().dropna(how="all")
    bench_rets = rets[benchmark_name]
    corrs = []
    labels = []
    for name in fund_names:
        if name not in rets.columns:
            continue
        combined = pd.concat(
            [rets[name].rename("fund"), bench_rets.rename("bench")],
            axis=1,
        ).dropna()
        if len(combined) < 20:
            continue
        c = combined["fund"].corr(combined["bench"])
        corrs.append(c)
        labels.append(short_name(name))

    colors = ["#2ca02c" if c >= 0 else "#d62728" for c in corrs]
    fig = go.Figure(go.Bar(
        x=labels, y=corrs,
        marker_color=colors,
        text=[f"{c:.2f}" for c in corrs],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Correlation with {short_name(benchmark_name)} (Full Period, Daily Returns)",
        yaxis_title="Correlation",
        yaxis=dict(range=[-0.1, 1.1]),
        template="plotly_white", height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio optimization
# ---------------------------------------------------------------------------

def optimize_portfolios(prices, benchmark_name, aqr_names):
    """
    AQR tab: FTSE All World + AQR funds.
    Constraint: FTSE weight >= 50%, all weights >= 0, sum = 1.
    """
    cols = [benchmark_name] + aqr_names
    overlap = prices[cols].dropna()
    daily_rets = overlap.pct_change().dropna()

    n = len(cols)
    cov = daily_rets.cov().values * 252
    vols = np.sqrt(np.diag(cov))

    bench_idx = 0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.5 if i == bench_idx else 0.0, 1.0) for i in range(n)]

    n_aqr = len(aqr_names)
    ew = np.zeros(n)
    ew[bench_idx] = 0.5
    ew[1:] = 0.5 / n_aqr

    def port_var(w):
        return w @ cov @ w

    res_mv = minimize(port_var, ew.copy(), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    w_mv = res_mv.x if res_mv.success else ew.copy()

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ cov @ w)
        weighted_vol = w @ vols
        return 0.0 if port_vol < 1e-10 else -weighted_vol / port_vol

    res_md = minimize(neg_div_ratio, ew.copy(), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    w_md = res_md.x if res_md.success else ew.copy()

    return _build_portfolio_results(cols, daily_rets, cov, vols, {
        "Equal Weight": ew,
        "Min Variance": w_mv,
        "Max Diversification": w_md,
    }, benchmark_name)


def optimize_portfolios_free(prices, benchmark_name, fund_names):
    """
    ETF tab: no minimum weight constraint — optimizer picks freely.
    Constraints: all weights >= 0, sum = 1.
    """
    cols = [benchmark_name] + fund_names
    overlap = prices[cols].dropna()
    daily_rets = overlap.pct_change().dropna()

    n = len(cols)
    cov = daily_rets.cov().values * 252
    vols = np.sqrt(np.diag(cov))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]

    ew = np.ones(n) / n  # equal weight across all ETFs (incl. benchmark)

    def port_var(w):
        return w @ cov @ w

    res_mv = minimize(port_var, ew.copy(), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    w_mv = res_mv.x if res_mv.success else ew.copy()

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ cov @ w)
        weighted_vol = w @ vols
        return 0.0 if port_vol < 1e-10 else -weighted_vol / port_vol

    res_md = minimize(neg_div_ratio, ew.copy(), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    w_md = res_md.x if res_md.success else ew.copy()

    return _build_portfolio_results(cols, daily_rets, cov, vols, {
        "Equal Weight": ew,
        "Min Variance": w_mv,
        "Max Diversification": w_md,
    }, benchmark_name)


def _build_portfolio_results(cols, daily_rets, cov, vols, portfolios, benchmark_name):
    """Shared logic: build equity curves and stats for a set of portfolios."""
    equity_curves = pd.DataFrame(index=daily_rets.index)
    for name, w in portfolios.items():
        port_rets = daily_rets.values @ w
        equity_curves[name] = (1 + pd.Series(port_rets, index=daily_rets.index)).cumprod()

    bench_rets = daily_rets[benchmark_name]
    equity_curves["FTSE All World (100%)"] = (1 + bench_rets).cumprod()

    stats_rows = {}
    for name, w in portfolios.items():
        port_daily = pd.Series(daily_rets.values @ w, index=daily_rets.index)
        ann_ret = port_daily.mean() * 252
        ann_vol = port_daily.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + port_daily).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd = drawdown.min()
        dd_pct = drawdown * 100
        ulcer = np.sqrt((dd_pct ** 2).mean())
        upi = (ann_ret * 100) / ulcer if ulcer > 0 else 0
        stats_rows[name] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "UPI": upi,
        }

    ann_ret_f = bench_rets.mean() * 252
    ann_vol_f = bench_rets.std() * np.sqrt(252)
    cum_f = (1 + bench_rets).cumprod()
    dd_f = cum_f / cum_f.cummax() - 1
    dd_f_pct = dd_f * 100
    ulcer_f = np.sqrt((dd_f_pct ** 2).mean())
    upi_f = (ann_ret_f * 100) / ulcer_f if ulcer_f > 0 else 0
    stats_rows["FTSE All World (100%)"] = {
        "Ann. Return": ann_ret_f,
        "Ann. Vol": ann_vol_f,
        "Sharpe": ann_ret_f / ann_vol_f if ann_vol_f > 0 else 0,
        "Max Drawdown": dd_f.min(),
        "UPI": upi_f,
    }

    stats = pd.DataFrame(stats_rows).T
    weights = {
        name: {short_name(cols[i]): w[i] for i in range(len(cols))}
        for name, w in portfolios.items()
    }
    return weights, equity_curves, stats


def portfolio_chart(equity_curves):
    """Performance chart for optimized portfolios."""
    port_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#999999"]
    fig = go.Figure()
    for i, col in enumerate(equity_curves.columns):
        dash = "dash" if col == "FTSE All World (100%)" else None
        fig.add_trace(go.Scatter(
            x=equity_curves.index, y=equity_curves[col],
            mode="lines", name=col,
            line=dict(color=port_colors[i % len(port_colors)], width=2.5, dash=dash),
        ))
    fig.update_layout(
        title="Portfolio Performance (Indexed to 1.0)",
        yaxis_title="Growth of 1.0",
        template="plotly_white", height=520,
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def portfolio_stats_html(stats, weights):
    """Render portfolio statistics and weights as HTML tables."""
    stat_cols = ["Ann. Return", "Ann. Vol", "Sharpe", "Max Drawdown", "UPI"]
    header = "<th>Portfolio</th>" + "".join(f"<th>{c}</th>" for c in stat_cols)
    body = ""
    for portfolio in stats.index:
        cells = f"<td class='fund-name'>{portfolio}</td>"
        for c in stat_cols:
            val = stats.loc[portfolio, c]
            if c in ("Sharpe", "UPI"):
                cells += f"<td>{val:.2f}</td>"
            elif c == "Max Drawdown":
                pct = val * 100
                cells += f"<td class='neg'>{pct:.1f}%</td>"
            else:
                pct = val * 100
                cls = "pos" if pct >= 0 else "neg"
                cells += f"<td class='{cls}'>{pct:+.1f}%</td>"
        body += f"<tr>{cells}</tr>\n"
    stats_html = f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    fund_names = list(next(iter(weights.values())).keys())
    w_header = "<th>Portfolio</th>" + "".join(f"<th>{f}</th>" for f in fund_names)
    w_body = ""
    for portfolio, w in weights.items():
        cells = f"<td class='fund-name'>{portfolio}</td>"
        for f in fund_names:
            pct = w[f] * 100
            if pct < 0.5:
                cells += "<td style='color:#ccc'>0%</td>"
            else:
                cells += f"<td>{pct:.1f}%</td>"
        w_body += f"<tr>{cells}</tr>\n"
    weights_html = f"<table><thead><tr>{w_header}</tr></thead><tbody>{w_body}</tbody></table>"

    return stats_html, weights_html


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------

def returns_table_html(returns_table):
    """Render performance table as styled HTML."""
    display_cols = ["Start", "1M", "3M", "1Y", "Max", "Max (p.a.)", "Vol (ann.)", "Sharpe (1Y)", "UPI (1Y)"]
    col_labels = {
        "Start": "Start Date", "1M": "1 Month", "3M": "3 Months",
        "1Y": "1 Year", "Max": "Max (total)", "Max (p.a.)": "Max (p.a.)",
        "Vol (ann.)": "Vol (ann.)", "Sharpe (1Y)": "Sharpe (1Y)",
        "UPI (1Y)": "UPI (1Y)",
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
            elif c in ("Sharpe (1Y)", "UPI (1Y)"):
                cells += f"<td>{val:.2f}</td>"
            elif c == "Vol (ann.)":
                cells += f"<td>{val * 100:.1f}%</td>"
            else:
                pct = val * 100
                cls = "pos" if pct >= 0 else "neg"
                cells += f"<td class='{cls}'>{pct:+.2f}%</td>"
        body += f"<tr>{cells}</tr>\n"

    return f"<table class='sortable'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


# ---------------------------------------------------------------------------
# Tab section builders
# ---------------------------------------------------------------------------

def build_aqr_section(prices, returns_table, tickers):
    """Build the HTML content for the AQR Funds tab."""
    aqr_names = [name for isin, name, _ in tickers if isin.startswith("LU")]
    benchmark_name = next(
        (name for isin, name, _ in tickers if isin.startswith("IE")), None
    )

    fig1 = performance_chart(prices)
    fig2 = rolling_correlation_chart(prices, aqr_names, benchmark_name) if benchmark_name else None
    fig3 = correlation_heatmap(prices)
    fig4 = return_dendrogram(prices)
    stress_df = stress_test_table(prices, benchmark_name) if benchmark_name else None
    port_weights, port_curves, port_stats = optimize_portfolios(
        prices, benchmark_name, aqr_names,
    ) if benchmark_name else (None, None, None)
    fig5 = portfolio_chart(port_curves) if port_curves is not None else None

    tbl = returns_table_html(returns_table)
    stress_tbl = stress_test_html(stress_df, short_name(benchmark_name)) if stress_df is not None else ""
    port_stats_tbl, port_weights_tbl = portfolio_stats_html(port_stats, port_weights) if port_stats is not None else ("", "")
    c1 = fig1.to_html(full_html=False, include_plotlyjs=False)
    c2 = fig2.to_html(full_html=False, include_plotlyjs=False) if fig2 else ""
    c3 = fig3.to_html(full_html=False, include_plotlyjs=False)
    c4 = fig4.to_html(full_html=False, include_plotlyjs=False)
    c5 = fig5.to_html(full_html=False, include_plotlyjs=False) if fig5 else ""

    rolling_corr_section = (
        "<h2>Rolling Correlation with FTSE All World</h2>\n"
        "<p class='note'>60-day rolling correlation of daily returns. "
        "Values near 0 indicate low correlation (diversification benefit).</p>\n"
        f"<div class='chart-box'>{c2}</div>"
    ) if c2 else ""

    bench_label = short_name(benchmark_name) if benchmark_name else "benchmark"

    return f"""
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

<h2>Stress Test — Worst Weeks (FTSE All World)</h2>
<p class="note">Fund returns during the 5 worst weekly drawdowns of {bench_label}. Positive returns indicate diversification benefit.</p>
{stress_tbl}

<h2>Portfolio Optimization</h2>
<p class="note">Optimized portfolios using FTSE All World + AQR funds. Constraint: at least 50% in FTSE All World. Based on the common overlap period of all funds.</p>
<div class="chart-box">{c5}</div>

<h3>Portfolio Statistics</h3>
{port_stats_tbl}

<h3>Weight Allocation</h3>
{port_weights_tbl}
"""


def build_etf_section(prices, returns_table, etf_tickers):
    """Build the HTML content for the Global Equity ETFs tab."""
    benchmark_isin = "IE00BK5BQT80"
    benchmark_name = next(
        (name for isin, name, _ in etf_tickers if isin == benchmark_isin), None
    )
    etf_names = [name for isin, name, _ in etf_tickers if isin != benchmark_isin]

    fig1 = performance_chart(prices)
    fig_corr_bar = correlation_vs_benchmark_chart(prices, etf_names, benchmark_name) if benchmark_name else None
    fig2 = rolling_correlation_chart(prices, etf_names, benchmark_name) if benchmark_name else None
    fig3 = correlation_heatmap(prices)
    fig4 = return_dendrogram(prices)
    stress_df = stress_test_table(prices, benchmark_name) if benchmark_name else None
    port_weights, port_curves, port_stats = optimize_portfolios_free(
        prices, benchmark_name, etf_names,
    ) if benchmark_name else (None, None, None)
    fig5 = portfolio_chart(port_curves) if port_curves is not None else None

    tbl = returns_table_html(returns_table)
    stress_tbl = stress_test_html(stress_df, short_name(benchmark_name)) if stress_df is not None else ""
    port_stats_tbl, port_weights_tbl = portfolio_stats_html(port_stats, port_weights) if port_stats is not None else ("", "")
    c1 = fig1.to_html(full_html=False, include_plotlyjs=False)
    c_bar = fig_corr_bar.to_html(full_html=False, include_plotlyjs=False) if fig_corr_bar else ""
    c2 = fig2.to_html(full_html=False, include_plotlyjs=False) if fig2 else ""
    c3 = fig3.to_html(full_html=False, include_plotlyjs=False)
    c4 = fig4.to_html(full_html=False, include_plotlyjs=False)
    c5 = fig5.to_html(full_html=False, include_plotlyjs=False) if fig5 else ""

    bench_label = short_name(benchmark_name) if benchmark_name else "benchmark"

    corr_section = ""
    if c_bar:
        corr_section += (
            "<h2>Correlation with FTSE All World</h2>\n"
            "<p class='note'>Full-period Pearson correlation of each ETF's daily returns against the FTSE All World.</p>\n"
            f"<div class='chart-box'>{c_bar}</div>"
        )
    if c2:
        corr_section += (
            "<h2>Rolling Correlation with FTSE All World</h2>\n"
            "<p class='note'>60-day rolling correlation of daily returns against FTSE All World.</p>\n"
            f"<div class='chart-box'>{c2}</div>"
        )

    return f"""
<h2>Performance Summary</h2>
{tbl}

<h2>Indexed Performance</h2>
<div class="chart-box">{c1}</div>

<h2>Correlation Matrix</h2>
<div class="chart-box">{c3}</div>

<h2>Fund Clustering</h2>
<p class="note">Hierarchical clustering using correlation distance. ETFs that merge at lower heights have more similar return profiles.</p>
<div class="chart-box">{c4}</div>

{corr_section}

<h2>Stress Test — Worst Weeks (FTSE All World)</h2>
<p class="note">ETF returns during the 5 worst weekly drawdowns of {bench_label}.</p>
{stress_tbl}

<h2>Portfolio Optimization</h2>
<p class="note">Optimized portfolios from the global equity ETF universe. No minimum weight constraint — the optimizer picks freely. Based on the common overlap period of all ETFs.</p>
<div class="chart-box">{c5}</div>

<h3>Portfolio Statistics</h3>
{port_stats_tbl}

<h3>Weight Allocation</h3>
{port_weights_tbl}
"""


# ---------------------------------------------------------------------------
# Full HTML report (two tabs)
# ---------------------------------------------------------------------------

def generate_report(aqr_section, etf_section):
    """Wrap two tab sections into a complete HTML page."""
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fund Comparison</title>
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
  .sortable th {{
    cursor: pointer; user-select: none; position: relative;
  }}
  .sortable th:hover {{ background: #2a2a4e; }}
  .sortable th::after {{
    content: '\u2195'; opacity: 0.4; margin-left: 4px; font-size: 11px;
  }}
  .sortable th.sort-asc::after {{ content: '\u25B2'; opacity: 1; }}
  .sortable th.sort-desc::after {{ content: '\u25BC'; opacity: 1; }}
  td {{ padding: 10px 16px; text-align: center; border-bottom: 1px solid #eee; font-size: 13px; }}
  .fund-name {{ font-weight: 600; text-align: left !important; white-space: nowrap; }}
  .pos {{ color: #2ca02c; font-weight: 600; }}
  .neg {{ color: #d62728; font-weight: 600; }}
  tr:hover {{ background: #f5f5f5; }}
  .chart-box {{
    background: #fff; border-radius: 8px; padding: 15px; margin: 20px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.12);
  }}

  /* ── Tab navigation ── */
  .tab-nav {{
    display: flex; gap: 4px; margin: 24px 0 0; border-bottom: 2px solid #1a1a2e;
  }}
  .tab-btn {{
    padding: 10px 28px; font-size: 14px; font-weight: 600; border: none;
    border-radius: 6px 6px 0 0; cursor: pointer; background: #dde1ea; color: #555;
    transition: background 0.15s, color 0.15s;
  }}
  .tab-btn:hover {{ background: #c5cad8; color: #1a1a2e; }}
  .tab-btn.active {{ background: #1a1a2e; color: #fff; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
</style>
</head>
<body>
<h1>Fund Comparison</h1>
<p class="subtitle">Generated {generated} | Data source: Yahoo Finance</p>

<div class="tab-nav">
  <button class="tab-btn active" data-tab="aqr">AQR Funds</button>
  <button class="tab-btn" data-tab="etf">Global Equity ETFs</button>
</div>

<div class="tab-content active" id="tab-aqr">
{aqr_section}
</div>

<div class="tab-content" id="tab-etf">
{etf_section}
</div>

<script>
// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
    // Resize Plotly charts that were hidden during initial render
    document.querySelectorAll('#tab-' + tab + ' .js-plotly-plot').forEach(el => {{
      Plotly.Plots.resize(el);
    }});
  }});
}});

// ── Sortable tables ────────────────────────────────────────────────────────
document.querySelectorAll('table.sortable').forEach(table => {{
  const headers = table.querySelectorAll('th');
  headers.forEach((th, colIdx) => {{
    th.addEventListener('click', () => {{
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      const curDir = th.classList.contains('sort-asc') ? 'desc' : 'asc';
      headers.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
      th.classList.add('sort-' + curDir);
      rows.sort((a, b) => {{
        let aText = a.children[colIdx].textContent.trim();
        let bText = b.children[colIdx].textContent.trim();
        let aVal = parseFloat(aText.replace(/[%+,]/g, ''));
        let bVal = parseFloat(bText.replace(/[%+,]/g, ''));
        if (isNaN(aVal) || isNaN(bVal)) {{
          return curDir === 'asc' ? aText.localeCompare(bText) : bText.localeCompare(aText);
        }}
        return curDir === 'asc' ? aVal - bVal : bVal - aVal;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "public")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "index.html")

    print("=" * 60)
    print("  AQR Fund Comparison")
    print("=" * 60)

    # ── AQR funds ──────────────────────────────────────────────
    aqr_csv = os.path.join(project_dir, "tickerlist.csv")
    aqr_tickers = read_tickers(aqr_csv)
    print(f"\n  AQR funds ({len(aqr_tickers)}):\n")
    aqr_prices = download_prices(aqr_tickers)

    if aqr_prices.empty:
        print("\nERROR: No AQR data downloaded")
        sys.exit(1)

    aqr_prices = aqr_prices.ffill()
    print(f"\n  AQR combined: {aqr_prices.shape[1]} funds, {aqr_prices.shape[0]} trading days")
    aqr_returns = compute_returns_table(aqr_prices)

    # ── Global equity ETFs ──────────────────────────────────────
    etf_csv = os.path.join(project_dir, "etfs.csv")
    etf_tickers = read_tickers(etf_csv)
    print(f"\n  Global equity ETFs ({len(etf_tickers)}):\n")
    etf_prices = download_prices(etf_tickers)

    if etf_prices.empty:
        print("\nERROR: No ETF data downloaded")
        sys.exit(1)

    etf_prices = etf_prices.ffill()
    print(f"\n  ETF combined: {etf_prices.shape[1]} ETFs, {etf_prices.shape[0]} trading days")
    etf_returns = compute_returns_table(etf_prices)

    # ── Build report ────────────────────────────────────────────
    print("\n  Building AQR section...")
    aqr_section = build_aqr_section(aqr_prices, aqr_returns, aqr_tickers)
    print("  Building ETF section...")
    etf_section = build_etf_section(etf_prices, etf_returns, etf_tickers)

    html = generate_report(aqr_section, etf_section)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Report: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
