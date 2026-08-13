"""
AQR Fund Comparison Tool

Downloads daily prices via yfinance (using ISINs) for funds in tickerlist.csv
and ETFs in etfs.csv, generates an HTML report with three tabs:

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

Tab 3 – Sector Performance:
- US Sectors (SPDR Select Sector ETFs vs S&P 500 / Nasdaq 100)
- STOXX Europe 600 Sectors (Lyxor/Amundi/iShares vs broad STOXX 600)
- Sourced from CSV outputs of the equity-sector-performance scripts.
"""

import datetime as dt
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    # Other AQR-tab non-AQR funds
    "UBS Carry": "UBS Carry",
    "Invesco Physical Gold": "Gold",
    "Global Aggregate Bond UCITS EUR Hedged": "Global Agg Bond",
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


def empty_state_html(title, message):
    """Render a simple notice box when a section cannot be computed."""
    return (
        f"<div class='empty-state'><strong>{title}</strong>"
        f"<div>{message}</div></div>"
    )


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

DESPIKE_SIGMA = 10.0     # one-day excursion in robust daily sigmas of that fund
DESPIKE_MIN_REL = 0.03   # ...and at least 3% in absolute terms
DESPIKE_REVERT = 0.015   # ...and the two neighbours must agree within 1.5%


def despike(prices, n_sigma=DESPIKE_SIGMA, min_rel=DESPIKE_MIN_REL,
            revert=DESPIKE_REVERT, verbose=True):
    """Remove isolated bad prints from a price frame.

    Yahoo occasionally returns a stray quote that is wildly off and reverts the
    next day. A single such print creates a matched pair of huge returns (e.g.
    +25% / -20%) and wrecks every volatility, correlation and optimizer input
    computed from the full history.

    A point is treated as a bad print only when ALL THREE hold:

      1. EXCURSION vs its neighbours. The reference is the average of the day
         before and the day after, not a rolling median. Measured against that,
         the excursion must exceed n_sigma times the fund's own robust daily
         volatility (MAD of returns x 1.4826).
      2. It must also be at least min_rel in absolute terms, so a fund in a
         dead-calm stretch does not get shredded by rounding noise.
      3. The neighbours must agree with each other within revert -- a bad print
         leaves no trace, the market resumes exactly where it left off.

    Why the neighbour average rather than a rolling median: it separates the
    two cases by orders of magnitude. On this data the real bad prints are
    90-120 sigma excursions, while the worst genuine one-day move that also
    happens to look symmetric is under 3 sigma. There is no threshold worth
    arguing about in between.

    Earlier attempts using a Hampel filter on a rolling median flagged real
    history -- the COVID bottom on FTSE All World (2020-02-28) and the middle
    of a genuine two-day drawdown on AQR Style Premia (2022-04-05). Both were
    correct prices; the median-based reference simply could not tell a V-shaped
    market move from a data error.

    Outliers are replaced by linear interpolation between their neighbours, so
    the date grid stays intact. Every removal is printed -- silent data surgery
    is worse than the bad data.
    """
    if prices.empty:
        return prices
    cleaned = prices.copy()
    removed = []
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < 30:
            continue
        r = s.pct_change().dropna()
        sigma = 1.4826 * (r - r.median()).abs().median()      # robuste Tagesvola
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        prv, nxt = s.shift(1), s.shift(-1)
        ref = (prv + nxt) / 2.0
        excursion = (s / ref - 1).abs()
        neighbours_agree = (prv / nxt - 1).abs() < revert
        bad = ((excursion > n_sigma * sigma)
               & (excursion > min_rel)
               & neighbours_agree).fillna(False)
        if not bad.any():
            continue
        for d in s.index[bad]:
            removed.append((short_name(col), d, float(s.loc[d]), float(ref.loc[d]),
                            float(excursion.loc[d] / sigma)))
        fixed = s.mask(bad).interpolate(method="linear", limit_direction="both")
        cleaned.loc[fixed.index, col] = fixed
    if verbose and removed:
        print(f"\n  Despiking: {len(removed)} bad print(s) replaced")
        for name, d, was, ref_, sig in removed:
            print(f"    {name:25s} {d:%Y-%m-%d}  {was:>10.4f}  ->  {ref_:>9.4f}"
                  f"   ({sig:.0f} sigma)")
    elif verbose:
        print("\n  Despiking: no bad prints found")
    return cleaned


TRIM_BREAK = 0.15        # residual daily move that despiking could not repair
TRIM_MAX_FRACTION = 0.40  # only trim if the damage sits in the first 40% of the series


def trim_broken_history(prices, threshold=TRIM_BREAK,
                        max_fraction=TRIM_MAX_FRACTION, verbose=True):
    """Cut off an early stretch of history that despiking cannot repair.

    despike() fixes ISOLATED bad prints. It cannot fix a period where bad
    prints arrive every few days, because then a bad print has another bad
    print as its neighbour: the "neighbours agree" test fails, and worse, a
    GOOD price sandwiched between two bad ones gets flagged instead.

    MSCI Japan is the live example: 92 daily moves beyond 15% between
    2009-11 and 2010-12 (the quote alternates between roughly 17 and 25 --
    a currency mix-up in the source), then nothing at all. From 2011-01 the
    series is spotless at 17.6% vol; before that it is unusable at ~50%.

    So: after despiking, look for residual moves beyond `threshold`. If the
    LAST one still sits inside the first `max_fraction` of the series, treat
    everything up to it as unusable and start the series after it. The
    fraction guard matters -- a recent break is news, not a data artifact,
    and must not silently delete the fund's current history.
    """
    if prices.empty:
        return prices
    out = prices.copy()
    for col in prices.columns:
        s_ = prices[col].dropna()
        if len(s_) < 60:
            continue
        r = s_.pct_change()
        breaks = r.index[(r.abs() > threshold).fillna(False)]
        if len(breaks) == 0:
            continue
        last_break = breaks.max()
        pos = s_.index.get_loc(last_break)
        if pos > len(s_) * max_fraction:
            if verbose:
                print(f"    {short_name(col):25s} {len(breaks)} residual break(s), latest "
                      f"{last_break:%Y-%m-%d} -- too recent to trim, LEFT AS IS")
            continue
        out.loc[out.index <= last_break, col] = np.nan
        if verbose:
            kept = s_.index[pos + 1]
            print(f"    {short_name(col):25s} {len(breaks)} break(s) up to {last_break:%Y-%m-%d}"
                  f" -- dropped {pos + 1} obs, series now starts {kept:%Y-%m-%d}")
    return out


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
    px = despike(pd.DataFrame(all_prices))
    print("  Trimming unrepairable history:")
    return trim_broken_history(px)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def compute_returns_table(prices, last_valid=None):
    """Compute 1M, 3M, 1Y, Max total and annualized returns.

    last_valid: dict {col -> (last_date, last_price)} captured before ffill,
                so each fund's returns are calculated up to its own last real
                data point (not the ffill-padded end of the DataFrame).
    """
    rows = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            continue
        if last_valid and col in last_valid:
            last_date, last_price = last_valid[col]
            s = s[s.index <= last_date]  # strip ffill-padded tail
        else:
            last_date, last_price = s.index[-1], s.iloc[-1]
        if s.empty:
            continue

        # Cutoffs anchored to this fund's own last data date
        cutoffs = {
            "1M": last_date - pd.DateOffset(months=1),
            "3M": last_date - pd.DateOffset(months=3),
            "1Y": last_date - pd.DateOffset(years=1),
        }

        row = {
            "Start": s.index[0].strftime("%Y-%m-%d"),
            "Last Date": last_date.strftime("%Y-%m-%d"),
            "Last Price": last_price,
        }

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
            # Vol ueber das LETZTE JAHR, nicht ueber die ganze Historie. Zwei Gruende:
            #  1) einheitliches Fenster mit Sharpe (1Y) und UPI (1Y) -- sonst passen die
            #     Kennzahlen der Tabelle nicht zueinander
            #  2) eine Vol ueber 8+ Jahre mischt laengst vergangene Regime bei und reagiert
            #     kaum noch auf das aktuelle Risiko des Fonds
            one_year_ago = last_date - pd.DateOffset(years=1)
            rets_1y = daily_rets[daily_rets.index >= one_year_ago]
            if len(rets_1y) > 20:
                ann_vol_1y = rets_1y.std() * np.sqrt(252)
                row["Vol (ann.)"] = ann_vol_1y
                ann_ret_1y = rets_1y.mean() * 252
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
    if prices.empty:
        return None

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        return None

    latest = prices.index.max()
    starts = prices.apply(lambda s: s.dropna().index.min())
    common_start = starts.max()
    if pd.isna(common_start):
        return None

    timeframes = {
        "1M": latest - pd.DateOffset(months=1),
        "3M": latest - pd.DateOffset(months=3),
        "1Y": latest - pd.DateOffset(years=1),
        "All": common_start,
    }

    fig = go.Figure()
    buttons = []
    trace_groups = []
    for tf_label, tf_start in timeframes.items():
        start = max(tf_start, common_start)
        trimmed = prices[prices.index >= start].copy()
        trace_indices = []
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
            trace_indices.append(len(fig.data) - 1)
        trace_groups.append(trace_indices)

    if not fig.data:
        return None

    total_traces = len(fig.data)
    for tf_label, trace_indices in zip(timeframes.keys(), trace_groups):
        vis = [False] * total_traces
        for idx in trace_indices:
            vis[idx] = True
        buttons.append(dict(
            label=tf_label,
            method="update",
            args=[
                {"visible": vis},
                {"title": f"Indexed Performance ({tf_label})"},
            ],
        ))

    fig.update_layout(
        # t=130 schafft Platz fuer Titel UND die zweizeilige Legende darunter;
        # mit dem Default ueberlagerte die Legende den Titel.
        title=dict(text="Indexed Performance (All)", y=0.97, yanchor="top"),
        yaxis_title="Growth of 1.0",
        template="plotly_white", height=560, margin=dict(t=130, b=60),
        legend=dict(orientation="h", y=1.13, x=0.5, xanchor="center"),
        hovermode="x unified",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.0, xanchor="right",
            y=1.13, yanchor="bottom",
            buttons=buttons,
            bgcolor="#e8e8e8",
            font=dict(size=12),
        )],
    )
    return fig


TARGET_VOL = 0.10      # every fund is rescaled to this annualized volatility
ANNUAL_RF = 0.02       # assumed risk-free rate; only the EXCESS return is levered


def _funding_label(k, annual_rf):
    """Annual funding cost (k>1) or cash credit (k<1) implied by the scaling."""
    cost = (k - 1.0) * annual_rf
    if cost >= 0:
        return f"funding cost {cost*100:.2f}%/yr"
    return f"cash credit {-cost*100:.2f}%/yr"


def funding_range_note(prices, target_vol=TARGET_VOL, annual_rf=ANNUAL_RF):
    """Sentence describing how large the implied funding costs actually are.

    Computed on the full common window, i.e. the 'All' view.
    """
    if prices is None or prices.empty:
        return ""
    prices = prices.dropna(axis=1, how="all")
    starts = prices.apply(lambda s: s.dropna().index.min())
    common_start = starts.max()
    if pd.isna(common_start):
        return ""
    trimmed = prices[prices.index >= common_start]
    rf_daily = annual_rf / 252.0
    rows = []
    for col in trimmed.columns:
        s = trimmed[col].dropna()
        if len(s) < 3:
            continue
        ex = s.pct_change().dropna() - rf_daily
        vol = ex.std() * np.sqrt(252)
        if not np.isfinite(vol) or vol <= 0:
            continue
        rows.append((short_name(col), target_vol / vol))
    if not rows:
        return ""
    rows.sort(key=lambda x: x[1])
    lo_name, lo_k = rows[0]
    hi_name, hi_k = rows[-1]
    lo_c = (lo_k - 1) * annual_rf * 100
    hi_c = (hi_k - 1) * annual_rf * 100
    return (
        f"Funding costs are included: scaling the excess return is the same as holding the fund "
        f"at k times its size and financing the extra (k−1) at the {annual_rf*100:.1f}% "
        f"risk-free rate. Over the full window this ranges from {hi_c:+.2f}%/yr for "
        f"{hi_name} (k = {hi_k:.2f}×, the most levered) down to {lo_c:+.2f}%/yr for "
        f"{lo_name} (k = {lo_k:.2f}× — de-levered funds hold the unused cash and earn "
        f"the rate instead of paying it). Real financing is dearer than the risk-free rate, "
        f"so this is a floor."
    )


def vol_normalized_chart(prices, target_vol=TARGET_VOL, annual_rf=ANNUAL_RF):
    """All funds rescaled to a common volatility, then indexed to 1.0.

    Comparing raw performance mixes two things: how good a fund is and how much
    risk it took. A fund with twice the volatility should earn twice the return
    just for taking twice the risk -- that tells you nothing about skill.

    Here each fund's EXCESS return over cash is levered by a constant factor
    k = target_vol / realized_vol, and cash is added back:

        excess = r - rf_daily
        k      = target_vol / (std(excess) * sqrt(252))
        scaled = excess * k + rf_daily

    Only the excess return is scaled -- levering the total return would also
    lever the cash component, which is not a risk-bearing part of the return.

    FUNDING COSTS ARE INCLUDED. Scaling the excess return is algebraically the
    same as holding the fund at k times its size and financing the extra (k-1)
    at the risk-free rate:

        k*r_fund - (k-1)*rf  ==  k*(r_fund - rf) + rf  ==  k*excess + rf

    So a fund levered to k = 3 pays (3-1) * rf = 2 * rf per year in funding.
    Mirror image for k < 1: the unused (1-k) sits in cash and EARNS rf, which
    is why de-levered funds get a small credit rather than a cost. Real-world
    financing is of course dearer than the risk-free rate -- the funding drag
    shown here is a floor, not a quote.

    Every line therefore has the same ~target_vol volatility WITHIN the shown
    window, so the ending value is directly the risk-adjusted ranking: whoever
    ends highest delivered the most return per unit of risk.

    The scaling factor is recomputed PER TIMEFRAME, so each window is
    self-contained. Caveat: over 1M that is ~21 observations, which makes the
    volatility estimate noisy -- read the short windows with care.
    """
    if prices.empty:
        return None

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        return None

    latest = prices.index.max()
    starts = prices.apply(lambda s: s.dropna().index.min())
    common_start = starts.max()
    if pd.isna(common_start):
        return None

    timeframes = {
        "1M": latest - pd.DateOffset(months=1),
        "3M": latest - pd.DateOffset(months=3),
        "1Y": latest - pd.DateOffset(years=1),
        "All": common_start,
    }
    rf_daily = annual_rf / 252.0

    fig = go.Figure()
    buttons = []
    trace_groups = []
    for tf_label, tf_start in timeframes.items():
        start = max(tf_start, common_start)
        trimmed = prices[prices.index >= start]
        trace_indices = []
        for i, col in enumerate(trimmed.columns):
            s = trimmed[col].dropna()
            if len(s) < 3:
                continue
            r = s.pct_change().dropna()
            excess = r - rf_daily
            vol = excess.std() * np.sqrt(252)
            if not np.isfinite(vol) or vol <= 0:
                continue
            k = target_vol / vol
            scaled = excess * k + rf_daily
            nav = (1 + scaled).cumprod()
            # start the line at 1.0 on the window's first date
            nav = pd.concat([pd.Series([1.0], index=[s.index[0]]), nav])
            fig.add_trace(go.Scatter(
                x=nav.index, y=nav.values,
                mode="lines", name=short_name(col),
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                visible=(tf_label == "All"),
                showlegend=(tf_label == "All"),
                hovertemplate=(f"<b>{short_name(col)}</b><br>%{{y:.3f}}"
                               f"<br>vol {vol*100:.1f}% -> x{k:.2f}"
                               f"<br>{_funding_label(k, annual_rf)}<extra></extra>"),
            ))
            trace_indices.append(len(fig.data) - 1)
        trace_groups.append(trace_indices)

    if not fig.data:
        return None

    total_traces = len(fig.data)
    for tf_label, trace_indices in zip(timeframes.keys(), trace_groups):
        vis = [False] * total_traces
        for idx in trace_indices:
            vis[idx] = True
        buttons.append(dict(
            label=tf_label,
            method="update",
            args=[
                {"visible": vis},
                {"title": f"Indexed Vol Normalized Performance ({tf_label})"},
            ],
        ))

    fig.add_annotation(
        text=(f"Excess return over a {annual_rf*100:.1f}% risk-free rate levered to "
              f"{target_vol*100:.0f}% vol — funding of the levered part is charged at "
              f"that same rate"),
        xref="paper", yref="paper", x=0, y=-0.16, xanchor="left", yanchor="top",
        showarrow=False, font=dict(size=11, color="#666"),
    )
    fig.update_layout(
        title=dict(text="Indexed Vol Normalized Performance (All)", y=0.97, yanchor="top"),
        yaxis_title=f"Growth of 1.0 at {target_vol*100:.0f}% vol",
        template="plotly_white", height=600, margin=dict(t=130, b=90),
        legend=dict(orientation="h", y=1.13, x=0.5, xanchor="center"),
        hovermode="x unified",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.0, xanchor="right",
            y=1.13, yanchor="bottom",
            buttons=buttons,
            bgcolor="#e8e8e8",
            font=dict(size=12),
        )],
    )
    return fig


def rolling_correlation_chart(prices, fund_names, benchmark_name, window=60):
    """60-day rolling correlation of funds vs benchmark."""
    if prices.empty or benchmark_name not in prices.columns:
        return None

    bench_rets = prices[benchmark_name].pct_change()
    fig = go.Figure()
    for i, name in enumerate(fund_names):
        if name not in prices.columns:
            continue
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

    if not fig.data:
        return None

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
    if rets.empty:
        return None

    rets = rets.loc[:, rets.nunique(dropna=True) > 1]
    if rets.shape[1] < 2:
        return None

    corr = rets.corr()
    valid_cols = corr.columns[~corr.isna().any(axis=0)]
    corr = corr.loc[valid_cols, valid_cols]
    if corr.shape[0] < 2:
        return None

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
    if bench_weekly.empty:
        return None

    worst_weeks = bench_weekly.nsmallest(n_worst)
    if worst_weeks.empty:
        return None

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
    if rets.empty:
        return None

    rets = rets.loc[:, rets.nunique(dropna=True) > 1]
    if rets.empty:
        return None

    corr = rets.corr()
    valid_cols = corr.columns[~corr.isna().any(axis=0)]
    corr = corr.loc[valid_cols, valid_cols]
    if corr.empty:
        return None

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
    if benchmark_name not in rets.columns:
        return None

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

    if not labels:
        return None

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
    aqr_names = [name for name in aqr_names if name in prices.columns]
    if benchmark_name not in prices.columns or not aqr_names:
        return None, None, None

    cols = [benchmark_name] + aqr_names
    overlap = prices[cols].dropna()
    daily_rets = overlap.pct_change().dropna()
    if daily_rets.empty:
        return None, None, None

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
    ETF tab: no minimum weight constraint - optimizer picks freely.
    Constraints: all weights >= 0, sum = 1.
    """
    fund_names = [name for name in fund_names if name in prices.columns]
    if benchmark_name not in prices.columns or not fund_names:
        return None, None, None

    cols = [benchmark_name] + fund_names
    overlap = prices[cols].dropna()
    daily_rets = overlap.pct_change().dropna()
    if daily_rets.empty:
        return None, None, None

    n = len(cols)
    cov = daily_rets.cov().values * 252
    vols = np.sqrt(np.diag(cov))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]

    ew = np.ones(n) / n

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
    if equity_curves is None or equity_curves.empty:
        return None

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
    if stats is None or stats.empty or weights is None or not weights:
        return "", ""

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
    if returns_table is None or returns_table.empty:
        return empty_state_html(
            "No performance data available",
            "Yahoo Finance did not return enough usable price history for this group.",
        )

    display_cols = ["Start", "Last Date", "Last Price", "1M", "3M", "1Y", "Max", "Max (p.a.)", "Vol (ann.)", "Sharpe (1Y)", "UPI (1Y)"]
    col_labels = {
        "Start": "Start Date", "Last Date": "Last Price Date", "Last Price": "Last Price",
        "1M": "1 Month", "3M": "3 Months",
        "1Y": "1 Year", "Max": "Max (total)", "Max (p.a.)": "Max (p.a.)",
        "Vol (ann.)": "Vol (ann., 1Y)", "Sharpe (1Y)": "Sharpe (1Y)",
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
                cells += "<td>-</td>"
            elif c in ("Start", "Last Date"):
                cells += f"<td>{val}</td>"
            elif c == "Last Price":
                cells += f"<td>{val:,.2f}</td>"
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
# Sector performance (sourced from equity-sector-performance CSVs)
# ---------------------------------------------------------------------------

SECTOR_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sector_scripts", "output",
))


def _sector_color_for(name, broad_labels, palette_idx):
    """Black for the broad index, palette colors for sectors."""
    if name in broad_labels:
        return "#1a1a2e"
    return COLORS[palette_idx % len(COLORS)]


def sector_performance_chart(prices, broad_labels, title):
    """Indexed performance chart with timeframe buttons; broad index highlighted."""
    if prices is None or prices.empty:
        return None

    latest = prices.index.max()
    earliest = prices.index.min()
    timeframes = {
        "1M": latest - pd.DateOffset(months=1),
        "3M": latest - pd.DateOffset(months=3),
        "6M": latest - pd.DateOffset(months=6),
        "1Y": latest - pd.DateOffset(years=1),
        "All": earliest,
    }

    fig = go.Figure()
    trace_groups = []
    palette_idx = 0
    name_to_pidx = {}
    for name in prices.columns:
        if name in broad_labels:
            name_to_pidx[name] = -1
        else:
            name_to_pidx[name] = palette_idx
            palette_idx += 1

    for tf_label, tf_start in timeframes.items():
        start = max(tf_start, earliest)
        trimmed = prices[prices.index >= start]
        trace_indices = []
        for name in trimmed.columns:
            s = trimmed[name].dropna()
            if s.empty:
                continue
            indexed = s / s.iloc[0]
            is_broad = name in broad_labels
            color = _sector_color_for(name, broad_labels, name_to_pidx[name])
            fig.add_trace(go.Scatter(
                x=indexed.index, y=indexed.values,
                mode="lines", name=name,
                line=dict(color=color, width=3 if is_broad else 1.6),
                opacity=1.0 if is_broad else 0.85,
                visible=(tf_label == "1Y"),
                showlegend=(tf_label == "1Y"),
            ))
            trace_indices.append(len(fig.data) - 1)
        trace_groups.append(trace_indices)

    if not fig.data:
        return None

    total = len(fig.data)
    buttons = []
    for tf_label, indices in zip(timeframes.keys(), trace_groups):
        vis = [False] * total
        for idx in indices:
            vis[idx] = True
        buttons.append(dict(
            label=tf_label,
            method="update",
            args=[{"visible": vis}, {"title": f"{title} ({tf_label})"}],
        ))

    fig.update_layout(
        title=f"{title} (1Y)",
        yaxis_title="Growth of 1.0",
        template="plotly_white",
        height=540,
        legend=dict(orientation="v", y=1.0, x=1.02, xanchor="left", yanchor="top"),
        hovermode="x unified",
        margin=dict(r=200),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.0, xanchor="left", y=1.12, yanchor="top",
            buttons=buttons,
            bgcolor="#e8e8e8", font=dict(size=12),
        )],
    )
    return fig


def sector_metrics_table_html(metrics_df):
    """Render the sector metrics CSV in the AQR table style."""
    if metrics_df is None or metrics_df.empty:
        return ""
    cols = [c for c in ["1W", "1M", "3M", "1Y", "Ann Vol (1Y)", "Sharpe (rf=0)"] if c in metrics_df.columns]
    header = "<th>Sector</th>" + "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, row in metrics_df.iterrows():
        cells = f"<td class='fund-name'>{row['Series']}</td>"
        for c in cols:
            val = row.get(c)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cells += "<td>-</td>"
            elif c == "Sharpe (rf=0)":
                cells += f"<td>{val:.2f}</td>"
            elif c == "Ann Vol (1Y)":
                cells += f"<td>{val * 100:.1f}%</td>"
            else:
                pct = val * 100
                cls = "pos" if pct >= 0 else "neg"
                cells += f"<td class='{cls}'>{pct:+.2f}%</td>"
        body += f"<tr>{cells}</tr>\n"
    return f"<table class='sortable'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _sector_only_prices(prices, broad_labels):
    """Strip broad indices (S&P 500 / Nasdaq 100 / Stoxx 600) from a price frame."""
    cols = [c for c in prices.columns if c not in broad_labels]
    return prices[cols].dropna(how="all")


def compute_dispersion_summary(prices, broad_labels):
    """Top-bottom spreads at multiple horizons + cross-sectional dispersion regime."""
    sec = _sector_only_prices(prices, broad_labels)
    if sec.empty or sec.shape[1] < 2:
        return None

    last = sec.index.max()
    horizons = [("1W", 7), ("1M", 30), ("3M", 90), ("1Y", 365)]
    spreads = []
    for label, days in horizons:
        cutoff = last - pd.Timedelta(days=days)
        hist = sec.loc[:cutoff].dropna(how="all")
        if hist.empty:
            continue
        base = hist.iloc[-1]
        rets = (sec.iloc[-1] / base - 1.0).dropna()
        if rets.empty:
            continue
        spreads.append({
            "Horizon": label,
            "Best": rets.idxmax(), "Best Ret": float(rets.max()),
            "Worst": rets.idxmin(), "Worst Ret": float(rets.min()),
            "Spread (pp)": float(rets.max() - rets.min()) * 100,
        })

    daily_rets = sec.pct_change()
    cs_disp = daily_rets.std(axis=1).rolling(21).mean() * np.sqrt(252)
    cs_disp_clean = cs_disp.dropna()
    if cs_disp_clean.empty:
        return {"spreads": spreads, "current_disp": None, "avg_disp_1y": None,
                "percentile": None, "regime": None}

    current = float(cs_disp_clean.iloc[-1])
    one_year_ago = last - pd.DateOffset(years=1)
    last_year = cs_disp_clean[cs_disp_clean.index >= one_year_ago]
    avg_1y = float(last_year.mean()) if not last_year.empty else None
    if last_year.empty:
        pct_rank, regime = None, None
    else:
        pct_rank = float((last_year < current).mean())
        if pct_rank < 0.33:
            regime = "narrow"
        elif pct_rank < 0.66:
            regime = "normal"
        else:
            regime = "wide"

    return {"spreads": spreads, "current_disp": current, "avg_disp_1y": avg_1y,
            "percentile": pct_rank, "regime": regime}


def dispersion_chart(prices, broad_labels, title):
    """Dual-axis chart: 21d cross-sectional σ (left) + 63d avg pairwise corr (right)."""
    sec = _sector_only_prices(prices, broad_labels)
    if sec.empty or sec.shape[1] < 2:
        return None

    daily_rets = sec.pct_change().dropna(how="all")
    cs_disp = daily_rets.std(axis=1).rolling(21).mean() * np.sqrt(252)

    window = 63
    n = len(daily_rets)
    if n <= window:
        avg_corr_s = pd.Series(dtype=float)
    else:
        avg_corr_vals, idx = [], []
        for i in range(window, n):
            win = daily_rets.iloc[i - window:i]
            c = win.corr().values
            if c.shape[0] < 2:
                continue
            mask = ~np.eye(c.shape[0], dtype=bool)
            avg_corr_vals.append(np.nanmean(c[mask]))
            idx.append(daily_rets.index[i])
        avg_corr_s = pd.Series(avg_corr_vals, index=idx)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=cs_disp.index, y=cs_disp.values,
        name="21d cross-sectional σ (annualized)",
        line=dict(color="#1a1a2e", width=2),
    ), secondary_y=False)
    if not avg_corr_s.empty:
        fig.add_trace(go.Scatter(
            x=avg_corr_s.index, y=avg_corr_s.values,
            name="63d avg pairwise correlation",
            line=dict(color="#d62728", width=2),
        ), secondary_y=True)
    fig.update_layout(
        title=title,
        template="plotly_white", height=460,
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Cross-sectional σ (annualized)", secondary_y=False)
    fig.update_yaxes(title_text="Avg pairwise correlation", secondary_y=True, range=[0, 1])
    return fig


def dispersion_summary_html(summary):
    """Render the regime banner + top-bottom spread table."""
    if not summary:
        return ""

    parts = []
    if summary.get("regime") is not None:
        color = {"narrow": "#2ca02c", "normal": "#888", "wide": "#d62728"}[summary["regime"]]
        parts.append(
            "<p class='note'>Current 21d cross-sectional dispersion (annualized): "
            f"<strong>{summary['current_disp'] * 100:.1f}%</strong> "
            f"(1Y average: {summary['avg_disp_1y'] * 100:.1f}%). "
            "Percentile vs trailing 1Y: "
            f"<strong style='color:{color}'>{summary['percentile'] * 100:.0f}% — {summary['regime']}</strong>.</p>"
        )

    spreads = summary.get("spreads") or []
    if spreads:
        rows = ""
        for s in spreads:
            best_cls = "pos" if s["Best Ret"] >= 0 else "neg"
            worst_cls = "pos" if s["Worst Ret"] >= 0 else "neg"
            rows += (
                f"<tr><td class='fund-name'>{s['Horizon']}</td>"
                f"<td>{s['Best']}</td>"
                f"<td class='{best_cls}'>{s['Best Ret'] * 100:+.2f}%</td>"
                f"<td>{s['Worst']}</td>"
                f"<td class='{worst_cls}'>{s['Worst Ret'] * 100:+.2f}%</td>"
                f"<td><strong>{s['Spread (pp)']:.1f} pp</strong></td></tr>"
            )
        parts.append(
            "<table><thead><tr>"
            "<th>Horizon</th><th>Best Sector</th><th>Best Ret</th>"
            "<th>Worst Sector</th><th>Worst Ret</th><th>Spread</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )

    return "\n".join(parts)


def sector_mapping_table_html(mapping_df):
    """Render the WKN/ISIN/Ticker mapping in the AQR table style."""
    if mapping_df is None or mapping_df.empty:
        return ""
    cols = [c for c in mapping_df.columns if c != "Series"]
    header = "<th>Sector</th>" + "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for _, row in mapping_df.iterrows():
        cells = f"<td class='fund-name'>{row['Series']}</td>"
        for c in cols:
            val = row[c]
            if pd.isna(val):
                val = ""
            cells += f"<td style='text-align:left'>{val}</td>"
        body += f"<tr>{cells}</tr>\n"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def build_sector_group(group_id, group_title, intro, prices_csv, metrics_csv, mapping_csv,
                       broad_labels, chart_title):
    """Render one sector group (US or STOXX 600) as HTML."""
    prices_path = os.path.join(SECTOR_DATA_DIR, prices_csv)
    metrics_path = os.path.join(SECTOR_DATA_DIR, metrics_csv)
    mapping_path = os.path.join(SECTOR_DATA_DIR, mapping_csv)

    if not (os.path.exists(prices_path) and os.path.exists(metrics_path) and os.path.exists(mapping_path)):
        return empty_state_html(
            f"{group_title} data unavailable",
            f"Run the sector report script in sector_scripts/ to generate "
            f"{prices_csv}, {metrics_csv}, {mapping_csv}.",
        )

    prices = pd.read_csv(prices_path, parse_dates=["date"]).set_index("date").sort_index()
    metrics = pd.read_csv(metrics_path)
    mapping = pd.read_csv(mapping_path)

    fig = sector_performance_chart(prices, broad_labels=broad_labels, title=chart_title)
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False) if fig else ""

    metrics_html = sector_metrics_table_html(metrics)
    mapping_html = sector_mapping_table_html(mapping)
    last_date = prices.index.max().strftime("%Y-%m-%d") if not prices.empty else "—"

    disp_summary = compute_dispersion_summary(prices, broad_labels)
    disp_summary_html = dispersion_summary_html(disp_summary) if disp_summary else ""
    disp_fig = dispersion_chart(prices, broad_labels, f"{group_title} — Dispersion Regime")
    disp_chart_html = disp_fig.to_html(full_html=False, include_plotlyjs=False) if disp_fig else ""

    parts = [f"<h2 id='{group_id}'>{group_title}</h2>"]
    if intro:
        parts.append(f"<p class='note'>{intro} Last data point: {last_date}.</p>")
    if metrics_html:
        parts.append("<h3>Performance &amp; Risk</h3>")
        parts.append(metrics_html)
    if chart_html:
        parts.append("<h3>Indexed Performance</h3>")
        parts.append(f"<div class='chart-box'>{chart_html}</div>")
    if disp_summary_html or disp_chart_html:
        parts.append("<h3>Sector Dispersion</h3>")
        parts.append(
            "<p class='note'>Wide top-bottom spreads and high cross-sectional &sigma; indicate "
            "strong rotation; high average pairwise correlation indicates a single-factor regime "
            "where everything moves together.</p>"
        )
        if disp_summary_html:
            parts.append(disp_summary_html)
        if disp_chart_html:
            parts.append(f"<div class='chart-box'>{disp_chart_html}</div>")
    if mapping_html:
        parts.append("<h3>Instrument Mapping</h3>")
        parts.append(mapping_html)
    return "\n".join(parts)


def build_sector_section():
    """Build the full Sector Performance tab content."""
    us = build_sector_group(
        group_id="sector-us",
        group_title="US Sectors",
        intro=(
            "SPDR Select Sector ETFs vs the S&amp;P 500 and Nasdaq 100. "
            "Daily total-return prices from EODHD."
        ),
        prices_csv="us_sector_prices.csv",
        metrics_csv="us_sector_metrics.csv",
        mapping_csv="us_sector_mapping.csv",
        broad_labels={"S&P 500", "Nasdaq 100"},
        chart_title="US Sector Indexed Performance",
    )
    eu = build_sector_group(
        group_id="sector-eu",
        group_title="STOXX Europe 600 Sectors",
        intro=(
            "Lyxor / Amundi / iShares STOXX Europe 600 sector ETFs vs the broad STOXX 600. "
            "Daily total-return prices from EODHD."
        ),
        prices_csv="stoxx600_sector_prices.csv",
        metrics_csv="stoxx600_sector_metrics.csv",
        mapping_csv="stoxx600_sector_mapping.csv",
        broad_labels={"Stoxx 600"},
        chart_title="STOXX 600 Sector Indexed Performance",
    )
    return us + "\n" + eu


# ---------------------------------------------------------------------------
# Tab section builders
# ---------------------------------------------------------------------------
def build_aqr_section(prices, prices_raw, returns_table, tickers):
    """Build the HTML content for the AQR Funds tab.

    prices:     ffill-padded DataFrame, used only for the indexed performance chart.
    prices_raw: pre-ffill DataFrame, used for all analytics so that overlaps are
                based on real trading dates only.
    """
    if prices.empty or prices_raw.empty:
        return empty_state_html(
            "AQR data unavailable",
            "No AQR prices were downloaded in this run, so this tab could not be updated.",
        )

    benchmark_isin = "IE00BK5BQT80"
    requested_benchmark = next(
        (name for isin, name, _ in tickers if isin == benchmark_isin), None
    )
    benchmark_name = requested_benchmark if requested_benchmark in prices_raw.columns else None
    aqr_names = [
        name for isin, name, _ in tickers
        if isin != benchmark_isin and name in prices_raw.columns
    ]

    fig1 = performance_chart(prices)
    fig1n = vol_normalized_chart(prices)
    fig2 = rolling_correlation_chart(prices_raw, aqr_names, benchmark_name) if benchmark_name else None
    fig3 = correlation_heatmap(prices_raw)
    fig4 = return_dendrogram(prices_raw)
    stress_df = stress_test_table(prices_raw, benchmark_name) if benchmark_name else None
    port_weights, port_curves, port_stats = optimize_portfolios(
        prices_raw, benchmark_name, aqr_names,
    ) if benchmark_name else (None, None, None)
    fig5 = portfolio_chart(port_curves) if port_curves is not None else None

    tbl = returns_table_html(returns_table)
    stress_tbl = stress_test_html(stress_df, short_name(benchmark_name)) if stress_df is not None else ""
    port_stats_tbl, port_weights_tbl = portfolio_stats_html(port_stats, port_weights) if port_stats is not None else ("", "")
    c1 = fig1.to_html(full_html=False, include_plotlyjs=False) if fig1 else ""
    c1n = fig1n.to_html(full_html=False, include_plotlyjs=False) if fig1n else ""
    c2 = fig2.to_html(full_html=False, include_plotlyjs=False) if fig2 else ""
    c3 = fig3.to_html(full_html=False, include_plotlyjs=False) if fig3 else ""
    c4 = fig4.to_html(full_html=False, include_plotlyjs=False) if fig4 else ""
    c5 = fig5.to_html(full_html=False, include_plotlyjs=False) if fig5 else ""

    rolling_corr_section = (
        "<h2>Rolling Correlation with FTSE All World</h2>\n"
        "<p class='note'>60-day rolling correlation of daily returns. "
        "Values near 0 indicate low correlation (diversification benefit).</p>\n"
        f"<div class='chart-box'>{c2}</div>"
    ) if c2 else ""

    bench_label = short_name(benchmark_name) if benchmark_name else "benchmark"

    sections = [
        "<h2>Performance Summary</h2>",
        tbl,
    ]
    if c1:
        sections.extend([
            "<h2>Indexed Performance</h2>",
            f'<div class="chart-box">{c1}</div>',
        ])
    if c1n:
        sections.extend([
            "<h2>Indexed Vol Normalized Performance</h2>",
            f"<p class=\"note\">The same funds, each rescaled to {TARGET_VOL*100:.0f}% "
            "annualized volatility. Raw performance mixes skill with risk taken — a fund "
            "running twice the volatility should earn twice the return for that alone. Here "
            "every line carries the same risk, so the ending value <em>is</em> the "
            "risk-adjusted ranking. Scaling is recomputed per timeframe; hover shows each "
            "fund's realized vol, applied factor and funding cost.</p>",
            f"<p class=\"note\">{funding_range_note(prices)}</p>",
            f'<div class="chart-box">{c1n}</div>',
        ])
    if c3:
        sections.extend([
            "<h2>Correlation Matrix</h2>",
            f'<div class="chart-box">{c3}</div>',
        ])
    if c4:
        sections.extend([
            "<h2>Fund Clustering</h2>",
            "<p class=\"note\">Hierarchical clustering using correlation distance. Funds that merge at lower heights have more similar return profiles.</p>",
            f'<div class="chart-box">{c4}</div>',
        ])
    if rolling_corr_section:
        sections.append(rolling_corr_section)
    if stress_tbl:
        sections.extend([
            "<h2>Stress Test - Worst Weeks (FTSE All World)</h2>",
            f'<p class="note">Fund returns during the 5 worst weekly drawdowns of {bench_label}. Positive returns indicate diversification benefit.</p>',
            stress_tbl,
        ])
    if c5 and port_stats_tbl and port_weights_tbl:
        sections.extend([
            "<h2>Portfolio Optimization</h2>",
            "<p class=\"note\">Optimized portfolios using FTSE All World + AQR funds. Constraint: at least 50% in FTSE All World. Based on the common overlap period of all funds.</p>",
            f'<div class="chart-box">{c5}</div>',
            "<h3>Portfolio Statistics</h3>",
            port_stats_tbl,
            "<h3>Weight Allocation</h3>",
            port_weights_tbl,
        ])
    if not benchmark_name:
        sections.append(empty_state_html(
            "Benchmark unavailable",
            "FTSE All World did not download in this run, so benchmark-relative analytics were skipped.",
        ))

    return "\n".join(sections)


def build_etf_section(prices, prices_raw, returns_table, etf_tickers):
    """Build the HTML content for the Global Equity ETFs tab.

    prices:     ffill-padded DataFrame, used only for the indexed performance chart.
    prices_raw: pre-ffill DataFrame, used for all analytics so that overlaps are
                based on real trading dates only.
    """
    if prices.empty or prices_raw.empty:
        return empty_state_html(
            "ETF data unavailable",
            "No ETF prices were downloaded in this run, so this tab could not be updated.",
        )

    benchmark_isin = "IE00BK5BQT80"
    requested_benchmark = next(
        (name for isin, name, _ in etf_tickers if isin == benchmark_isin), None
    )
    benchmark_name = requested_benchmark if requested_benchmark in prices_raw.columns else None
    etf_names = [
        name for isin, name, _ in etf_tickers
        if isin != benchmark_isin and name in prices_raw.columns
    ]

    fig1 = performance_chart(prices)
    fig1n = vol_normalized_chart(prices)
    fig_corr_bar = correlation_vs_benchmark_chart(prices_raw, etf_names, benchmark_name) if benchmark_name else None
    fig2 = rolling_correlation_chart(prices_raw, etf_names, benchmark_name) if benchmark_name else None
    fig3 = correlation_heatmap(prices_raw)
    fig4 = return_dendrogram(prices_raw)
    stress_df = stress_test_table(prices_raw, benchmark_name) if benchmark_name else None
    port_weights, port_curves, port_stats = optimize_portfolios_free(
        prices_raw, benchmark_name, etf_names,
    ) if benchmark_name else (None, None, None)
    fig5 = portfolio_chart(port_curves) if port_curves is not None else None

    tbl = returns_table_html(returns_table)
    stress_tbl = stress_test_html(stress_df, short_name(benchmark_name)) if stress_df is not None else ""
    port_stats_tbl, port_weights_tbl = portfolio_stats_html(port_stats, port_weights) if port_stats is not None else ("", "")
    c1 = fig1.to_html(full_html=False, include_plotlyjs=False) if fig1 else ""
    c1n = fig1n.to_html(full_html=False, include_plotlyjs=False) if fig1n else ""
    c_bar = fig_corr_bar.to_html(full_html=False, include_plotlyjs=False) if fig_corr_bar else ""
    c2 = fig2.to_html(full_html=False, include_plotlyjs=False) if fig2 else ""
    c3 = fig3.to_html(full_html=False, include_plotlyjs=False) if fig3 else ""
    c4 = fig4.to_html(full_html=False, include_plotlyjs=False) if fig4 else ""
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

    sections = [
        "<h2>Performance Summary</h2>",
        tbl,
    ]
    if c1:
        sections.extend([
            "<h2>Indexed Performance</h2>",
            f'<div class="chart-box">{c1}</div>',
        ])
    if c1n:
        sections.extend([
            "<h2>Indexed Vol Normalized Performance</h2>",
            f"<p class=\"note\">The same funds, each rescaled to {TARGET_VOL*100:.0f}% "
            "annualized volatility. Raw performance mixes skill with risk taken — a fund "
            "running twice the volatility should earn twice the return for that alone. Here "
            "every line carries the same risk, so the ending value <em>is</em> the "
            "risk-adjusted ranking. Scaling is recomputed per timeframe; hover shows each "
            "fund's realized vol, applied factor and funding cost.</p>",
            f"<p class=\"note\">{funding_range_note(prices)}</p>",
            f'<div class="chart-box">{c1n}</div>',
        ])
    if c3:
        sections.extend([
            "<h2>Correlation Matrix</h2>",
            f'<div class="chart-box">{c3}</div>',
        ])
    if c4:
        sections.extend([
            "<h2>Fund Clustering</h2>",
            "<p class=\"note\">Hierarchical clustering using correlation distance. ETFs that merge at lower heights have more similar return profiles.</p>",
            f'<div class="chart-box">{c4}</div>',
        ])
    if corr_section:
        sections.append(corr_section)
    if stress_tbl:
        sections.extend([
            "<h2>Stress Test - Worst Weeks (FTSE All World)</h2>",
            f'<p class="note">ETF returns during the 5 worst weekly drawdowns of {bench_label}.</p>',
            stress_tbl,
        ])
    if c5 and port_stats_tbl and port_weights_tbl:
        sections.extend([
            "<h2>Portfolio Optimization</h2>",
            "<p class=\"note\">Optimized portfolios from the global equity ETF universe. No minimum weight constraint - the optimizer picks freely. Based on the common overlap period of all ETFs.</p>",
            f'<div class="chart-box">{c5}</div>',
            "<h3>Portfolio Statistics</h3>",
            port_stats_tbl,
            "<h3>Weight Allocation</h3>",
            port_weights_tbl,
        ])
    if not benchmark_name:
        sections.append(empty_state_html(
            "Benchmark unavailable",
            "FTSE All World did not download in this run, so benchmark-relative analytics were skipped.",
        ))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Tab 4 - Maximum Diversification (cardinality-constrained)
# ---------------------------------------------------------------------------

DIV_N_ASSETS = 5          # wanted portfolio size (hard cardinality constraint)
DIV_MIN_WEIGHT = 0.10     # floor so a "5-asset" portfolio really holds 5

# Zwei Fenster, weil sich Fondsauswahl und Krisen-Abdeckung direkt widersprechen:
# jeder junge Fonds, den man aufnimmt, schneidet das GEMEINSAME Fenster vorne ab.
# Gefiltert wird deshalb nach STARTDATUM, nicht nach Anzahl Handelstage - die
# Tageszahl sagt nichts darueber, wie weit eine Reihe zurueckreicht (AQR Alt Trends
# hat 787 Tage, beginnt aber erst 2023-03 und kostet damit die 2022-Korrektur).
DIV_PANELS = [
    (4.0, "Long window", "includes the 2022 equity/bond selloff - the only real "
                         "stress event in this data, so this is the window that "
                         "actually speaks to crisis behaviour"),
    (2.0, "Recent window", "more funds qualify, but the window contains no crisis - "
                           "treat the correlations as fair-weather estimates"),
]


def _mdp_weights(corr, vols):
    """Maximum-diversification weights, solved as a CONVEX problem.

    DR(w) = (w'sigma) / sqrt(w' Sigma w). Substituting y = w*sigma and
    Sigma = D C D gives DR = sum(y) / sqrt(y'Cy), so under sum(y)=1 maximising DR
    is exactly minimising y'Cy with y >= 0 -- convex, unique optimum. Recover
    w ~ y/sigma. Maximising DR directly in w is non-convex and can land in a
    local optimum, which is why this substitution is worth the extra step.
    """
    n = len(vols)
    res = minimize(lambda y: y @ corr @ y, np.ones(n) / n, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda y: y.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None) / vols
    return w / w.sum()


def _solve_weights(objective, n, lo, n_start=25, seed=0):
    """Multi-start SLSQP on the simplex for the non-convex objectives (ERC, ENB,
    and DR once weight floors break the convex substitution)."""
    rng = np.random.default_rng(seed)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(lo, 1.0)] * n
    starts = [np.full(n, 1.0 / n)]
    starts += [lo + (1 - lo * n) * rng.dirichlet(np.ones(n)) for _ in range(n_start - 1)]
    best, best_v = starts[0], np.inf
    for s in starts:
        res = minimize(objective, s, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        if res.success and res.fun < best_v:
            best_v, best = res.fun, res.x
    return best


def _div_ratio(w, cov, vols):
    pv = np.sqrt(w @ cov @ w)
    return float((w @ vols) / pv) if pv > 1e-12 else np.nan


def _risk_shares(w, cov):
    """Anteil jedes Assets an der Portfolio-Varianz (Euler-Zerlegung)."""
    pv = w @ cov @ w
    return (w * (cov @ w)) / pv if pv > 1e-12 else np.full(len(w), np.nan)


def _enb(w, cov):
    """Effective Number of Bets: Entropie der Risikoverteilung ueber die
    UNKORRELIERTEN Hauptkomponenten von Sigma.

    Das ist das Mass, das 'moeglichst orthogonal / breit' woertlich nimmt:
    ENB = 1 heisst, das ganze Risiko haengt an EINEM Faktor; ENB = N heisst,
    es verteilt sich gleichmaessig auf N unabhaengige Risikoquellen. Die
    Diversification Ratio misst das NICHT - sie belohnt niedrige Portfolio-Vol
    und kann dabei in einen einzigen Faktor laufen.

    Caveat: die Hauptkomponenten sind eine Basiswahl (Meuccis Minimum-Torsion
    waere die basis-unabhaengige Variante); als Vergleichsmass zwischen
    Portfolios auf DEMSELBEN Universum ist die PCA-Variante aussagekraeftig.
    """
    lam, vec = np.linalg.eigh(cov)
    lam = np.clip(lam, 1e-16, None)
    v = vec.T @ w
    c = (v ** 2) * lam
    tot = c.sum()
    if tot <= 1e-16:
        return np.nan
    p = c / tot
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def _schemes(cov, vols, corr, lo):
    """Die vier Gewichtungs-Logiken, die auf der Seite verglichen werden."""
    n = len(vols)
    ew = np.full(n, 1.0 / n)

    def erc_obj(w):
        return float(((_risk_shares(w, cov) - 1.0 / n) ** 2).sum())

    def neg_dr(w):
        return -_div_ratio(w, cov, vols)

    def neg_enb(w):
        return -_enb(w, cov)

    return {
        "Equal Weight": ew,
        "Max Diversification (DR)": (_mdp_weights(corr, vols) if lo <= 0
                                     else _solve_weights(neg_dr, n, lo)),
        "Equal Risk Contribution": _solve_weights(erc_obj, n, lo),
        "Max Effective Bets (ENB)": _solve_weights(neg_enb, n, lo),
    }


def _port_stats(w, rets):
    pr = rets.values @ w
    eq = (1 + pd.Series(pr, index=rets.index)).cumprod()
    ann = float(eq.iloc[-1] ** (252 / len(pr)) - 1)
    vol = float(pr.std() * np.sqrt(252))
    mdd = float((eq / eq.cummax() - 1).min())
    return ann, vol, mdd, eq


def _div_panel(prices, benchmark_name, min_years, label, blurb):
    """Ein Fenster-Panel: Vollenumeration + Gewichtungsvarianten + Krisen-Check."""
    # 1) Kandidaten nach STARTDATUM filtern (s. DIV_PANELS): ein Fonds ist nur
    #    dabei, wenn er min_years zurueckreicht - sonst kuerzt er allen anderen
    #    das gemeinsame Fenster weg.
    last = max(prices[c].dropna().index.max() for c in prices.columns)
    cutoff = last - pd.Timedelta(days=int(min_years * 365.25))
    starts = {c: prices[c].dropna().index.min() for c in prices.columns}
    keep = [c for c in prices.columns if starts[c] <= cutoff]
    dropped = sorted(((c, starts[c]) for c in prices.columns if c not in keep),
                     key=lambda t: t[1])
    if len(keep) < DIV_N_ASSETS:
        return empty_state_html(
            f"Maximum Diversification &mdash; {label}",
            f"Only {len(keep)} fund(s) reach back {min_years:g} years "
            f"(before {cutoff.date()}) - need {DIV_N_ASSETS}.")

    # 2) EIN gemeinsames Fenster fuer ALLE Kandidaten. Liesse man jede Teilmenge
    #    ihr eigenes Fenster nutzen, waeren die DR-Werte nicht vergleichbar
    #    (langes ruhiges Fenster vs. kurzes turbulentes).
    win = prices[keep].dropna()
    rets = win.pct_change().dropna()
    if len(rets) < 60:
        return empty_state_html(f"Maximum Diversification &mdash; {label}",
                                "Common window too short for a covariance estimate.")

    cov_all = rets.cov().values * 252
    vol_all = np.sqrt(np.diag(cov_all))
    idx = {c: i for i, c in enumerate(keep)}

    # 3) Vollenumeration aller Teilmengen - C(n,5) ist klein genug fuer exakt
    #    statt heuristisch.
    rows = []
    for sub in combinations(keep, DIV_N_ASSETS):
        ii = [idx[c] for c in sub]
        cov = cov_all[np.ix_(ii, ii)]
        vols = vol_all[ii]
        corr = cov / np.outer(vols, vols)
        w = _mdp_weights(corr, vols)
        ann, vol, mdd, _ = _port_stats(w, rets[list(sub)])
        rows.append(dict(sub=sub, w=w, dr=_div_ratio(w, cov, vols),
                         enb=_enb(w, cov), ann=ann, vol=vol, mdd=mdd,
                         corr=float(corr[np.triu_indices(len(sub), 1)].mean())))
    rows.sort(key=lambda d: -d["dr"])

    top_rows = "".join(
        f"<tr><td class='fund-name'>{' + '.join(r['sub'])}</td>"
        f"<td>{r['dr']:.2f}</td><td>{r['enb']:.2f}</td>"
        f"<td class='{'pos' if r['ann'] >= 0 else 'neg'}'>{r['ann']:+.1%}</td>"
        f"<td>{r['vol']:.1%}</td><td>{r['ann'] / r['vol'] if r['vol'] else float('nan'):.2f}</td>"
        f"<td class='neg'>{r['mdd']:.1%}</td><td>{r['corr']:+.2f}</td></tr>"
        for r in rows[:10])

    # 4) Gewichtungs-Varianten fuer die beste Kombination
    best = rows[0]
    sub = list(best["sub"])
    ii = [idx[c] for c in sub]
    cov = cov_all[np.ix_(ii, ii)]
    vols = vol_all[ii]
    corr = cov / np.outer(vols, vols)
    sub_rets = rets[sub]

    schemes = _schemes(cov, vols, corr, lo=0.0)
    schemes.update({f"{k} · min {DIV_MIN_WEIGHT:.0%}": v
                    for k, v in _schemes(cov, vols, corr, lo=DIV_MIN_WEIGHT).items()
                    if k.startswith("Max Diversification")})

    curves, scheme_rows = {}, ""
    for name, w in schemes.items():
        ann, vol, mdd, eq = _port_stats(w, sub_rets)
        curves[name] = eq / eq.iloc[0] * 100
        wt = "".join(f"<td>{x:.0%}</td>" for x in w)
        scheme_rows += (
            f"<tr><td class='fund-name'>{name}</td>{wt}"
            f"<td>{_div_ratio(w, cov, vols):.2f}</td><td>{_enb(w, cov):.2f}</td>"
            f"<td class='{'pos' if ann >= 0 else 'neg'}'>{ann:+.1%}</td>"
            f"<td>{vol:.1%}</td><td>{ann / vol if vol else float('nan'):.2f}</td>"
            f"<td class='neg'>{mdd:.1%}</td></tr>")

    fig = go.Figure()
    for name, eq in curves.items():
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))
    fig.update_layout(height=430, margin=dict(l=50, r=20, t=30, b=40),
                      yaxis_title="Indexed (100 = start)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18))
    slug = "".join(ch for ch in label.lower() if ch.isalnum())
    chart = fig.to_html(full_html=False, include_plotlyjs=False,
                        div_id=f"div-portfolio-chart-{slug}")

    # 5) Krisen-Check: in den schlechtesten Benchmark-Wochen - war etwas im Plus?
    crisis = ""
    if benchmark_name and (benchmark_name in rets.columns or benchmark_name in win.columns):
        bench_rets = (win[benchmark_name].pct_change().dropna()
                      if benchmark_name in win.columns else rets[benchmark_name])
        wk = (1 + rets[sub]).resample("W").prod() - 1
        bwk = (1 + bench_rets).resample("W").prod() - 1
        both = pd.concat([wk, bwk.rename("__bench")], axis=1).dropna()
        worst = both.nsmallest(10, "__bench")
        w_best = schemes["Max Diversification (DR)"]
        hits = 0
        crisis_rows = ""
        for ts, row in worst.iterrows():
            s = row[sub]
            pos = s[s > 0]
            hits += len(pos) > 0
            crisis_rows += (
                f"<tr><td class='fund-name'>{ts.date()}</td>"
                f"<td class='neg'>{row['__bench']:+.2%}</td>"
                f"<td class='{'pos' if (s @ w_best) >= 0 else 'neg'}'>{s @ w_best:+.2%}</td>"
                f"<td>{len(pos)} / {len(sub)}</td>"
                f"<td class='fund-name'>{s.idxmax()}</td>"
                f"<td class='{'pos' if s.max() >= 0 else 'neg'}'>{s.max():+.2%}</td></tr>")
        crisis = f"""
<h3>Crisis behaviour &mdash; the 10 worst weeks for {benchmark_name}</h3>
<p class="note">The goal is that <em>something</em> in the book is green and can be
 sold. That property comes from which assets you hold, not how you weight them.
 In <strong>{hits} of {len(worst)}</strong> of the worst weeks at least one holding
 was positive.</p>
<table class="sortable">
<thead><tr><th>Week</th><th>{benchmark_name}</th><th>Portfolio (MDP)</th>
  <th>Holdings positive</th><th>Best holding</th><th>Its return</th></tr></thead>
<tbody>{crisis_rows}</tbody></table>"""

    drop_note = (" Too short for this window: "
                 + ", ".join(f"{c} (from {d.date()})" for c, d in dropped) + "."
                 if dropped else "")
    head = "".join(f"<th>{c}</th>" for c in sub)

    return f"""
<h2>{label} &mdash; best {DIV_N_ASSETS}-asset portfolio</h2>
<p class="note">
 {blurb}.<br>
 All <strong>{len(rows)}</strong> possible {DIV_N_ASSETS}-asset combinations of
 {len(keep)} candidates solved exactly (full enumeration, no heuristic).
 Common window <strong>{rets.index.min().date()} &rarr; {rets.index.max().date()}</strong>
 ({len(rets)} days) &mdash; identical for every combination, so the numbers are
 comparable.{drop_note}
</p>

<h3>Top 10 combinations by diversification ratio</h3>
<table class="sortable">
<thead><tr><th>Portfolio</th><th>DR</th><th>ENB</th><th>Return p.a.</th><th>Vol</th>
  <th>Sharpe</th><th>Max DD</th><th>&#216; Corr</th></tr></thead>
<tbody>{top_rows}</tbody></table>

<h3>Weighting schemes for the winner</h3>
<p class="note">Same five assets, four definitions of "diversified". Note how the
 unconstrained DP optimum concentrates capital in the lowest-vol asset &mdash; the
 <em>min {DIV_MIN_WEIGHT:.0%}</em> row shows what a genuinely {DIV_N_ASSETS}-asset
 book costs in DR and buys in return.</p>
<table class="sortable">
<thead><tr><th>Scheme</th>{head}<th>DR</th><th>ENB</th><th>Return p.a.</th>
  <th>Vol</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{scheme_rows}</tbody></table>

<div class="chart-box">{chart}</div>
{crisis}
"""


def build_diversification_section(prices, tickers):
    """Tab 4: bestes N-Asset-Portfolio nach Diversifikation + Krisen-Diagnostik.

    prices: pre-ffill (prices_raw) - Korrelationen duerfen nur auf echten
            Handelstagen beruhen, ffill erzeugt kuenstliche Null-Renditen.
    """
    if prices is None or prices.empty:
        return empty_state_html("Maximum Diversification",
                                "No price data available.")
    prices = prices.rename(columns=short_name)
    benchmark_name = next((short_name(n) for i, n, _ in tickers
                           if i == "IE00BK5BQT80"), None)
    if benchmark_name not in prices.columns:
        benchmark_name = None

    intro = f"""
<h2>Maximum Diversification &mdash; {DIV_N_ASSETS} assets</h2>
<p class="note">
 Holding no more than {DIV_N_ASSETS} funds, which ones combine into the least
 concentrated book? Two measures are reported side by side because they answer
 different questions:
</p>
<p class="note">
 <strong>DR</strong> (diversification ratio) = weighted-average vol / portfolio vol.
 Maximising it is mathematically the same as a minimum-variance problem in
 <em>correlation</em> space, so it spreads <em>risk</em> rather than capital &mdash;
 and it will happily pile into the lowest-vol asset, which is why the unconstrained
 optimum tends to be bond-heavy.<br>
 <strong>ENB</strong> (effective number of bets) = entropy of risk across the
 uncorrelated principal components. This is the one that takes
 &ldquo;as orthogonal as possible&rdquo; literally: ENB&nbsp;=&nbsp;1 means all risk
 rides on a single factor, ENB&nbsp;=&nbsp;{DIV_N_ASSETS} means it is spread evenly
 over {DIV_N_ASSETS} independent sources. A high DR with a low ENB is low volatility
 concentrated in one risk source &mdash; not what you want.
</p>
<p class="note">
 Weight floors matter: without one, a &ldquo;{DIV_N_ASSETS}-asset&rdquo; optimum can
 put 2% in a holding and really be a 3-asset portfolio. The
 <em>min&nbsp;{DIV_MIN_WEIGHT:.0%}</em> row shows what insisting on a genuine
 {DIV_N_ASSETS}-asset book costs in DR and buys in return.
</p>
"""
    panels = [_div_panel(prices, benchmark_name, yrs, label, blurb)
              for yrs, label, blurb in DIV_PANELS]
    return intro + "<hr style='margin:38px 0;border:none;border-top:1px solid #ddd'>".join(panels)


def generate_report(aqr_section, etf_section, sector_section, diversification_section):
    """Wrap four tab sections into a complete HTML page."""
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
  h3 {{ color: #2a3a5a; margin-top: 24px; font-size: 16px; }}
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
  .empty-state {{
    background: #fff7e6; border: 1px solid #f1d28a; border-radius: 8px;
    padding: 16px 18px; margin: 20px 0; color: #6f4e0f;
  }}
  .empty-state strong {{ display: block; margin-bottom: 4px; color: #7a5200; }}
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
  <button class="tab-btn" data-tab="sector">Sector Performance</button>
  <button class="tab-btn" data-tab="div">Max Diversification</button>
</div>

<div class="tab-content active" id="tab-aqr">
{aqr_section}
</div>

<div class="tab-content" id="tab-etf">
{etf_section}
</div>

<div class="tab-content" id="tab-sector">
{sector_section}
</div>

<div class="tab-content" id="tab-div">
{diversification_section}
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

    aqr_csv = os.path.join(project_dir, "tickerlist.csv")
    aqr_tickers = read_tickers(aqr_csv)
    print(f"\n  AQR funds ({len(aqr_tickers)}):\n")
    aqr_prices = download_prices(aqr_tickers)
    aqr_prices_raw = aqr_prices.copy()
    if aqr_prices.empty:
        print("\nWARNING: No AQR data downloaded; the AQR tab will show a notice.")
        aqr_returns = pd.DataFrame()
    else:
        aqr_last_valid = {
            col: (aqr_prices[col].last_valid_index(), aqr_prices[col].dropna().iloc[-1])
            for col in aqr_prices.columns
            if aqr_prices[col].last_valid_index() is not None
        }
        aqr_prices = aqr_prices.ffill()
        print(f"\n  AQR combined: {aqr_prices.shape[1]} funds, {aqr_prices.shape[0]} trading days")
        aqr_returns = compute_returns_table(aqr_prices, aqr_last_valid)

    etf_csv = os.path.join(project_dir, "etfs.csv")
    etf_tickers = read_tickers(etf_csv)
    print(f"\n  Global equity ETFs ({len(etf_tickers)}):\n")
    etf_prices = download_prices(etf_tickers)
    etf_prices_raw = etf_prices.copy()

    if aqr_prices.empty and etf_prices.empty:
        print("\nERROR: No data downloaded for either AQR funds or ETFs")
        sys.exit(1)

    if etf_prices.empty:
        print("\nWARNING: No ETF data downloaded; the ETF tab will show a notice.")
        etf_returns = pd.DataFrame()
    else:
        etf_last_valid = {
            col: (etf_prices[col].last_valid_index(), etf_prices[col].dropna().iloc[-1])
            for col in etf_prices.columns
            if etf_prices[col].last_valid_index() is not None
        }
        etf_prices = etf_prices.ffill()
        print(f"\n  ETF combined: {etf_prices.shape[1]} ETFs, {etf_prices.shape[0]} trading days")
        etf_returns = compute_returns_table(etf_prices, etf_last_valid)

    print("\n  Building AQR section...")
    aqr_section = build_aqr_section(aqr_prices, aqr_prices_raw, aqr_returns, aqr_tickers)
    print("  Building ETF section...")
    etf_section = build_etf_section(etf_prices, etf_prices_raw, etf_returns, etf_tickers)
    print("  Building Sector section...")
    sector_section = build_sector_section()
    print("  Building Max Diversification section...")
    try:
        div_section = build_diversification_section(aqr_prices_raw, aqr_tickers)
    except Exception as exc:
        print(f"  ! Max Diversification skipped: {exc}")
        div_section = empty_state_html("Maximum Diversification", str(exc))

    html = generate_report(aqr_section, etf_section, sector_section, div_section)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Report: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()