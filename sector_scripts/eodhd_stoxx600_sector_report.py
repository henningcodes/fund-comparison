import os
import re
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN")
if not EODHD_API_TOKEN:
    raise SystemExit("EODHD_API_TOKEN environment variable is required.")

SERIES = [
    ("Stoxx 600", "LYX0Q0"),
    ("Banks", "LYX01W"),
    ("Health Care", "LYX02K"),
    ("Basic Resources", "LYX01X"),
    ("Industrials", "LYX02L"),
    ("Energy", "LYX02P"),
    ("Technology", "LYX02S"),
    ("Utilities", "LYX02V"),
    ("Insurance", "LYX02M"),
    ("Consumer Discretionary", "LYX02U"),
    ("Real Estate", "A2H58A"),
    ("Consumer Staples", "LYX02J"),
    ("Telecom", "A0RPSF"),
    ("Basic Materials", "LYX01Y"),
]


# Static WKN → ISIN mapping for the STOXX 600 sector ETFs used in SERIES.
# Avoids relying on ariva.de, which is slow and unreliable.
_WKN_ISIN_CACHE: dict[str, str] = {
    "LYX0Q0": "LU0908500753",  # Amundi/Lyxor Core STOXX Europe 600 (DR) Acc
    "LYX01W": "FR0010345371",  # Lyxor STOXX Europe 600 Banks
    "LYX02K": "FR0010344879",  # Lyxor STOXX Europe 600 Health Care
    "LYX01X": "LU1834983550",  # Amundi STOXX Europe 600 Basic Resources
    "LYX02L": "LU1834987890",  # Amundi STOXX Europe 600 Industrial G&S
    "LYX02P": "LU1834988278",  # Amundi STOXX Europe 600 Energy ESG Screened (LOGS.XETRA)
    "LYX02S": "LU1834988518",  # Amundi STOXX Europe 600 Technology
    "LYX02V": "LU1834988864",  # Amundi STOXX Europe 600 Utilities
    "LYX02M": "LU1834987973",  # Lyxor STOXX Europe 600 Insurance
    "LYX02U": "LU1834988781",  # Amundi STOXX Europe 600 Consumer Disc. (LTVL.XETRA)
    "A2H58A": "DE000A0Q4R44",  # iShares STOXX Europe 600 Real Estate (DE)
    "LYX02J": "FR0010344861",  # Lyxor STOXX Europe 600 Food & Beverage
    "A0RPSF": "IE00B5MJYB88",  # Invesco European Telecoms Sector UCITS ETF
    "LYX01Y": "FR0010345470",  # Lyxor STOXX Europe 600 Chemicals
}


def fetch_isin_from_wkn(wkn: str) -> str:
    if wkn in _WKN_ISIN_CACHE:
        return _WKN_ISIN_CACHE[wkn]
    # Fallback for any WKN not in the cache.
    url = f"https://www.ariva.de/{wkn}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    m = re.search(r"ISIN[^A-Z0-9]*([A-Z]{2}[A-Z0-9]{10})", resp.text, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not find ISIN for WKN {wkn}")
    return m.group(1).upper()


def resolve_ticker_from_isin(isin: str) -> tuple[str, str]:
    url = f"https://eodhd.com/api/search/{isin}?api_token={EODHD_API_TOKEN}&fmt=json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"Could not find EODHD ticker for ISIN {isin}")

    # Prefer XETRA listing when available, then major EU venues.
    preference = ["XETRA", "PA", "F", "STU", "DU", "BE", "MI", "AS", "LSE"]
    rank = {ex: i for i, ex in enumerate(preference)}
    data = sorted(data, key=lambda x: rank.get(x.get("Exchange", ""), 999))
    row = data[0]
    return f"{row['Code']}.{row['Exchange']}", row["Name"]


def fetch_adjusted_close(ticker: str, start_date: str) -> pd.Series:
    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?from={start_date}&period=d&order=a&api_token={EODHD_API_TOKEN}&fmt=json"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"No EOD data for {ticker}")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["adjusted_close"].astype(float)


def perf_asof(series: pd.Series, days: int) -> float:
    end_date = series.index.max()
    target = end_date - timedelta(days=days)
    hist = series.loc[:target]
    if not hist.empty:
        base = hist.iloc[-1]
    else:
        # Fallback for limited-history plans: use the nearest available point.
        nearest_pos = series.index.get_indexer([target], method="nearest")[0]
        if nearest_pos == -1:
            return np.nan
        base = series.iloc[nearest_pos]
    return series.iloc[-1] / base - 1.0


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%" if pd.notna(x) else ""


def format_num(x: float) -> str:
    return f"{x:.2f}" if pd.notna(x) else ""


def build_html_report(
    metrics_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    chart_path: str,
    start_date: str,
    end_date: str,
) -> str:
    metrics_show = metrics_df.copy()
    for col in ["1W", "1M", "3M", "1Y", "Ann Vol (1Y)"]:
        metrics_show[col] = metrics_show[col].map(format_pct)
    metrics_show["Sharpe (rf=0)"] = metrics_show["Sharpe (rf=0)"].map(format_num)

    metrics_html = metrics_show.to_html(index=False, classes="table table-metrics", border=0, escape=True)
    mapping_html = mapping_df.to_html(index=False, classes="table table-mapping", border=0, escape=True)

    chart_file = os.path.basename(chart_path)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STOXX 600 Sector Performance Report</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1e2a3a;
      --muted: #627086;
      --line: #d9e0ea;
      --accent: #0f4c81;
      --head: #eaf2fb;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(15, 76, 129, 0.06);
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      color: var(--accent);
    }}
    h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
    }}
    p {{
      margin: 4px 0;
      color: var(--muted);
    }}
    .chart {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .table th, .table td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .table th {{
      background: var(--head);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>STOXX Europe 600 Sector Report</h1>
      <p>Data source: EODHD API</p>
      <p>Window used: {start_date} to {end_date}</p>
    </div>

    <div class="card">
      <h2>Performance, Volatility, Sharpe</h2>
      {metrics_html}
    </div>

    <div class="card">
      <h2>Relative Performance (Base 100, 1Y)</h2>
      <img class="chart" src="{chart_file}" alt="Relative performance chart">
    </div>

    <div class="card">
      <h2>WKN to Ticker Mapping</h2>
      {mapping_html}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    if not EODHD_API_TOKEN:
        raise ValueError("EODHD_API_TOKEN is missing.")

    start_date = (datetime.now(timezone.utc) - timedelta(days=500)).date().isoformat()
    prices: dict[str, pd.Series] = {}
    mapping_rows = []

    for name, wkn in SERIES:
        isin = fetch_isin_from_wkn(wkn)
        ticker, instrument_name = resolve_ticker_from_isin(isin)
        s = fetch_adjusted_close(ticker, start_date)
        prices[name] = s
        mapping_rows.append(
            {
                "Series": name,
                "WKN": wkn,
                "ISIN": isin,
                "EODHD Ticker": ticker,
                "Instrument Name": instrument_name,
            }
        )

    px = pd.DataFrame(prices).sort_index()
    latest = px.dropna(how="all").index.max()
    one_year_ago = latest - timedelta(days=365)
    one_year_window = px.loc[px.index >= one_year_ago]

    metrics = []
    for name, _wkn in SERIES:
        s = px[name].dropna()
        s_1y = s.loc[s.index >= one_year_ago]
        daily_ret_1y = s_1y.pct_change().dropna()

        perf_1w = perf_asof(s, 7)
        perf_1m = perf_asof(s, 30)
        perf_3m = perf_asof(s, 90)
        perf_1y = perf_asof(s, 365)

        ann_vol = float(daily_ret_1y.std() * np.sqrt(252)) if len(daily_ret_1y) > 1 else np.nan
        sharpe = float(perf_1y / ann_vol) if ann_vol and not np.isnan(ann_vol) else np.nan

        metrics.append(
            {
                "Series": name,
                "1W": perf_1w,
                "1M": perf_1m,
                "3M": perf_3m,
                "1Y": perf_1y,
                "Ann Vol (1Y)": ann_vol,
                "Sharpe (rf=0)": sharpe,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    mapping_df = pd.DataFrame(mapping_rows)

    rel = one_year_window / one_year_window.iloc[0] * 100.0
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 8))
    for col in rel.columns:
        lw = 2.8 if col == "Stoxx 600" else 1.4
        alpha = 1.0 if col == "Stoxx 600" else 0.8
        ax.plot(rel.index, rel[col], label=col, linewidth=lw, alpha=alpha)

    ax.set_title("STOXX Europe 600 Sector ETF Relative Performance (Base=100, 1Y)", fontsize=13)
    ax.set_ylabel("Relative Performance (Base 100)")
    ax.set_xlabel("Date")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    chart_path = "output/stoxx600_sector_relative_performance.png"
    table_path = "output/stoxx600_sector_metrics.csv"
    map_path = "output/stoxx600_sector_mapping.csv"
    prices_path = "output/stoxx600_sector_prices.csv"
    html_path = "output/stoxx600_sector_report.html"
    fig.savefig(chart_path, dpi=160)
    metrics_df.to_csv(table_path, index=False)
    mapping_df.to_csv(map_path, index=False)
    px.to_csv(prices_path, index_label="date")
    html = build_html_report(
        metrics_df=metrics_df,
        mapping_df=mapping_df,
        chart_path=chart_path,
        start_date=str(one_year_window.index.min().date()),
        end_date=str(one_year_window.index.max().date()),
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    pd.options.display.float_format = "{:,.2%}".format
    pretty = metrics_df.copy()
    for col in ["Ann Vol (1Y)", "Sharpe (rf=0)"]:
        if col == "Sharpe (rf=0)":
            pretty[col] = metrics_df[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        else:
            pretty[col] = metrics_df[col]

    print("Date range used:", one_year_window.index.min().date(), "to", one_year_window.index.max().date())
    print("\nResolved mapping (WKN -> ISIN -> EODHD ticker):")
    print(mapping_df.to_string(index=False))
    print("\nPerformance table:")
    print(pretty.to_string(index=False))
    print(f"\nSaved: {table_path}")
    print(f"Saved: {chart_path}")
    print(f"Saved: {map_path}")
    print(f"Saved: {html_path}")


if __name__ == "__main__":
    main()
