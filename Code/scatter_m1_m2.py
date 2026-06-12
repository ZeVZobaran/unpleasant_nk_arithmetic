# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:38:57 2026

@author: C337191
"""

# ============================================================
# Public liabilities scatter plots:
#   1. M1 / GDP vs general government gross debt / GDP
#   2. Broad money / GDP vs general government gross debt / GDP
#
# Sources:
#   - FRED-hosted OECD monetary aggregates and GDP
#   - FRED-hosted IMF IFS monetary aggregates/GDP where useful
#   - FRED-hosted IMF WEO general government gross debt
#   - IMF DataMapper fallback for Israel debt
#
# Output:
#   - public_liabilities_panel.csv
#   - m1_vs_public_debt.png
#   - broad_money_vs_public_debt.png
# ============================================================

import json
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# User controls
# -----------------------------

START_YEAR = 1995
END_YEAR = 2022

CONNECT_TIME_PATH = True
LABEL_LATEST = True

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
IMF_DATAMAPPER = "https://www.imf.org/external/datamapper/api/v1/{indicator}/{country}"


# -----------------------------
# Euro legacy conversion rates
# national-currency units per 1 euro
# -----------------------------

EUR_CONVERSION = {
    "FRF": 6.55957,
    "DEM": 1.95583,
    "ITL": 1936.27,
}


# -----------------------------
# Country configuration
#
# Candidate lists allow graceful fallback:
#   - try newer OECD/FRED style series first
#   - if unavailable, use older IMF/FRED IFS series
#
# multiplier is applied to the raw series.
# For old euro-area national currencies:
#   legacy currency -> euro: divide by conversion rate.
# -----------------------------

COUNTRIES = {
    "United States": {
        "m1": [{"id": "MANMM101USM189S"}],
        "broad": [{"id": "MABMM301USM189S"}],
        "gdp": [{"id": "USAGDPNADSMEI"}],
        "debt_fred": "GGGDTAUSA188N",
    },
    "United Kingdom": {
        "m1": [{"id": "MANMM101GBM189S"}],
        "broad": [{"id": "MABMM301GBM189S"}],
        "gdp": [{"id": "GBRGDPNADSMEI"}],
        "debt_fred": "GGGDTAGBA188N",
    },
    "Canada": {
        "m1": [{"id": "MANMM101CAM189S"}],
        "broad": [{"id": "MABMM301CAM189S"}],
        "gdp": [{"id": "CANGDPNADSMEI"}],
        "debt_fred": "GGGDTACAA188N",
    },
    "Japan": {
        "m1": [{"id": "MANMM101JPM189S"}],
        "broad": [{"id": "MABMM301JPM189S"}],
        "gdp": [{"id": "JPNGDPNADSMEI"}],
        "debt_fred": "GGGDTAJPA188N",
    },
    "Australia": {
        "m1": [{"id": "MANMM101AUM189S"}],
        "broad": [{"id": "MABMM301AUM189S"}],
        "gdp": [{"id": "AUSGDPNADSMEI"}],
        "debt_fred": "GGGDTAAUA188N",
    },
    "Korea": {
        "m1": [{"id": "MANMM101KRM189S"}],
        "broad": [{"id": "MABMM301KRM189S"}],
        "gdp": [{"id": "KORGDPNADSMEI"}],
        "debt_fred": "GGGDTAKRA188N",
    },
    "Mexico": {
        "m1": [{"id": "MANMM101MXM189S"}],
        "broad": [{"id": "MABMM301MXM189S"}],
        "gdp": [{"id": "MEXGDPNADSMEI"}],
        "debt_fred": "GGGDTAMXA188N",
    },

    # Euro-area countries:
    # First try country OECD-style modern candidates. If unavailable,
    # fall back to old IMF IFS national-currency monetary series and
    # convert FRF/DEM/ITL to euros.
    "France": {
        "m1": [
            {"id": "MANMM101FRM189S", "multiplier": 1.0},
            {"id": "MYAGM1FRM189S", "multiplier": 1 / EUR_CONVERSION["FRF"]},
        ],
        "broad": [
            {"id": "MABMM301FRM189S", "multiplier": 1.0},
            {"id": "MYAGM2FRM189S", "multiplier": 1 / EUR_CONVERSION["FRF"]},
            {"id": "MYAGM3FRM189S", "multiplier": 1 / EUR_CONVERSION["FRF"]},
        ],
        "gdp": [{"id": "NGDPXDCFRA", "multiplier": 1_000_000.0}],
        "debt_fred": "GGGDTAFRA188N",
    },
    "Germany": {
        "m1": [
            {"id": "MANMM101DEM189S", "multiplier": 1.0},
            {"id": "MYAGM1DEM189S", "multiplier": 1 / EUR_CONVERSION["DEM"]},
        ],
        "broad": [
            {"id": "MABMM301DEM189S", "multiplier": 1.0},
            {"id": "MYAGM2DEM189S", "multiplier": 1 / EUR_CONVERSION["DEM"]},
            {"id": "MYAGM3DEM189S", "multiplier": 1 / EUR_CONVERSION["DEM"]},
        ],
        "gdp": [{"id": "NGDPXDCDEA", "multiplier": 1_000_000.0}],
        "debt_fred": "GGGDTADEA188N",
    },
    "Italy": {
        "m1": [
            {"id": "MANMM101ITM189S", "multiplier": 1.0},
            {"id": "MYAGM1ITM189N", "multiplier": 1 / EUR_CONVERSION["ITL"]},
        ],
        "broad": [
            {"id": "MABMM301ITM189S", "multiplier": 1.0},
            {"id": "MYAGM2ITM189N", "multiplier": 1 / EUR_CONVERSION["ITL"]},
            {"id": "MYAGM3ITM189N", "multiplier": 1 / EUR_CONVERSION["ITL"]},
        ],
        "gdp": [{"id": "NGDPXDCITA", "multiplier": 1_000_000.0}],
        "debt_fred": "GGGDTAITA188N",
    },

    "Brazil": {
        "m1": [{"id": "MANMM101BRM189S"}],
        "broad": [{"id": "MABMM301BRM189S"}],
        "gdp": [{"id": "BRAGDPNADSMEI"}],
        "debt_fred": "GGGDTABRA188N",
    },
    "Israel": {
        "m1": [{"id": "MANMM101ILM189S"}],
        "broad": [{"id": "MABMM301ILM189S"}],
        "gdp": [{"id": "ISRGDPNADSMEI"}],
        # FRED debt ID is not as easy/consistent for Israel, so use IMF DataMapper fallback.
        "debt_imf": {"indicator": "GGXWDG_NGDP", "country": "ISR"},
    },
    "South Africa": {
        "m1": [{"id": "MANMM101ZAM189S"}],
        "broad": [{"id": "MABMM301ZAM189S"}],
        "gdp": [{"id": "ZAFGDPNADSMEI"}],
        "debt_fred": "GGGDTAZAA188N",
    },
}


# -----------------------------
# Download helpers
# -----------------------------

def get_fred_series(series_id: str) -> pd.Series:
    """
    Download one FRED series from the public CSV endpoint.
    No FRED API key required.
    """
    url = FRED_CSV.format(sid=series_id)
    df = pd.read_csv(url, parse_dates=["observation_date"])

    if series_id not in df.columns:
        raise ValueError(f"Unexpected FRED response for {series_id}. Columns: {df.columns.tolist()}")

    s = (
        df.set_index("observation_date")[series_id]
          .replace(".", np.nan)
          .astype(float)
          .dropna()
    )
    s.name = series_id
    return s


def get_first_available_fred_series(candidates, label: str):
    """
    Try a list of candidate FRED series. Return the first that downloads.
    Each candidate can be:
      {"id": "...", "multiplier": ...}
    """
    for cand in candidates:
        sid = cand["id"]
        multiplier = cand.get("multiplier", 1.0)

        try:
            s = get_fred_series(sid) * multiplier
            return s, sid
        except Exception as e:
            print(f"[WARN] {label}: could not load {sid}: {e}")

    return pd.Series(dtype=float), None


def get_imf_datamapper_series(indicator: str, country: str) -> pd.Series:
    """
    Download one IMF DataMapper annual series.
    Example:
      indicator='GGXWDG_NGDP' = general government gross debt, % GDP
      country='ISR'
    """
    url = IMF_DATAMAPPER.format(indicator=indicator, country=country)

    with urlopen(url, timeout=30) as response:
        raw = json.loads(response.read().decode("utf-8"))

    values = raw["values"][indicator][country]
    s = pd.Series({int(year): float(value) for year, value in values.items() if value is not None})
    s.name = f"{indicator}_{country}"
    return s.sort_index()


# -----------------------------
# Frequency helpers
# -----------------------------

def annual_average_stock(s: pd.Series) -> pd.Series:
    """
    Convert monthly/quarterly stock data to annual average stock.
    """
    if s.empty:
        return s
    return s.groupby(s.index.year).mean()


def annual_value(s: pd.Series) -> pd.Series:
    """
    Convert annual or higher-frequency data to annual values.
    For annual data this is unchanged; for higher-frequency level data it takes the annual mean.
    """
    if s.empty:
        return s
    return s.groupby(s.index.year).mean()


def annual_debt_from_fred(series_id: str) -> pd.Series:
    s = get_fred_series(series_id)
    return annual_value(s)


def get_debt_series(country: str, cfg: dict) -> pd.Series:
    if "debt_fred" in cfg:
        try:
            return annual_debt_from_fred(cfg["debt_fred"])
        except Exception as e:
            print(f"[WARN] {country}: FRED debt failed: {e}")

    if "debt_imf" in cfg:
        imf = cfg["debt_imf"]
        try:
            return get_imf_datamapper_series(imf["indicator"], imf["country"])
        except Exception as e:
            print(f"[WARN] {country}: IMF DataMapper debt failed: {e}")

    return pd.Series(dtype=float)


# -----------------------------
# Panel construction
# -----------------------------

def build_country_panel(country: str, cfg: dict) -> pd.DataFrame:
    m1_raw, m1_id = get_first_available_fred_series(cfg.get("m1", []), f"{country} M1")
    broad_raw, broad_id = get_first_available_fred_series(cfg.get("broad", []), f"{country} broad money")
    gdp_raw, gdp_id = get_first_available_fred_series(cfg.get("gdp", []), f"{country} GDP")

    debt = get_debt_series(country, cfg)

    if gdp_raw.empty or debt.empty:
        print(f"[WARN] Skipping {country}: missing GDP or debt.")
        return pd.DataFrame()

    m1 = annual_average_stock(m1_raw)
    broad = annual_average_stock(broad_raw)
    gdp = annual_value(gdp_raw)

    years = sorted(set(gdp.index) | set(debt.index) | set(m1.index) | set(broad.index))
    out = pd.DataFrame(index=years)
    out.index.name = "year"

    out["m1"] = m1
    out["broad_money"] = broad
    out["gdp"] = gdp
    out["debt_gdp"] = debt

    out["m1_gdp"] = 100 * out["m1"] / out["gdp"]
    out["broad_money_gdp"] = 100 * out["broad_money"] / out["gdp"]

    out["country"] = country
    out["m1_series"] = m1_id
    out["broad_series"] = broad_id
    out["gdp_series"] = gdp_id
    out["debt_source"] = cfg.get("debt_fred", f"IMF:{cfg.get('debt_imf', {}).get('indicator')}")

    return out.reset_index()


def build_panel(countries: dict, start_year: int, end_year: int) -> pd.DataFrame:
    frames = []

    for country, cfg in countries.items():
        print(f"\n[INFO] Building {country}")
        df_country = build_country_panel(country, cfg)
        if not df_country.empty:
            frames.append(df_country)

    if not frames:
        raise RuntimeError("No country panels could be built.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)].copy()

    return panel


# -----------------------------
# Plotting
# -----------------------------

def plot_money_vs_debt(
    panel: pd.DataFrame,
    money_col: str,
    title: str,
    ylabel: str,
    savepath: str,
    connect_time_path: bool = True,
    label_latest: bool = True,
):
    data = panel.dropna(subset=["debt_gdp", money_col]).copy()

    fig, ax = plt.subplots(figsize=(11, 7.5))

    for country, g in data.groupby("country"):
        g = g.sort_values("year")

        if g.empty:
            continue

        if connect_time_path:
            ax.plot(
                g["debt_gdp"],
                g[money_col],
                marker="o",
                linewidth=1.0,
                markersize=3.5,
                alpha=0.75,
                label=country,
            )
        else:
            ax.scatter(
                g["debt_gdp"],
                g[money_col],
                s=24,
                alpha=0.75,
                label=country,
            )

        if label_latest:
            last = g.iloc[-1]
            ax.annotate(
                f"{country} {int(last['year'])}",
                xy=(last["debt_gdp"], last[money_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("General government gross debt (% of GDP)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax

def add_debt_money_share_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds debt-to-money ratios.

    Since debt_gdp, m1_gdp, and broad_money_gdp are all in % of GDP,
    the GDP denominator cancels:

        debt_to_m1 = debt / M1
        debt_to_broad_money = debt / broad money

    Interpretation:
        debt_to_m1 = 5 means public debt is 5 times the M1 stock.
    """
    out = panel.copy()

    out["debt_to_m1"] = out["debt_gdp"] / out["m1_gdp"]
    out["debt_to_broad_money"] = out["debt_gdp"] / out["broad_money_gdp"]

    return out

def plot_debt_money_ratio_histograms(
    panel: pd.DataFrame,
    bins: int = 30,
    savepath_m1: str = "hist_debt_to_m1.png",
    savepath_broad: str = "hist_debt_to_broad_money.png",
    winsorize_q: float | None = 0.99,
):
    """
    Plots separate histograms for:
        debt / M1
        debt / broad money

    Since debt_gdp, m1_gdp, and broad_money_gdp are all % of GDP,
    the GDP denominator cancels.

    winsorize_q:
        If 0.99, trims the extreme top 1% of each ratio for readability.
        Set to None to use the raw full sample.
    """
    data = add_debt_money_share_columns(panel)

    ratios = {
        "debt_to_m1": {
            "title": "Distribution of Public Debt / M1",
            "xlabel": "Public debt / M1",
            "savepath": savepath_m1,
        },
        "debt_to_broad_money": {
            "title": "Distribution of Public Debt / Broad Money",
            "xlabel": "Public debt / broad money",
            "savepath": savepath_broad,
        },
    }

    figs_axes = {}

    for col, meta in ratios.items():
        x = data[col].replace([np.inf, -np.inf], np.nan).dropna()

        if winsorize_q is not None:
            upper = x.quantile(winsorize_q)
            x_plot = x[x <= upper]
            subtitle = f"Top {(1 - winsorize_q) * 100:.0f}% trimmed"
        else:
            x_plot = x
            subtitle = "Full sample"

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.hist(
            x_plot,
            bins=bins,
            alpha=0.8,
            edgecolor="black",
        )

        ax.axvline(x_plot.mean(), linestyle="--", linewidth=1.2, label=f"Mean: {x_plot.mean():.2f}")
        ax.axvline(x_plot.median(), linestyle=":", linewidth=1.5, label=f"Median: {x_plot.median():.2f}")

        ax.set_title(f"{meta['title']}\n{subtitle}")
        ax.set_xlabel(meta["xlabel"])
        ax.set_ylabel("Country-year observations")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(meta["savepath"], dpi=300, bbox_inches="tight")

        figs_axes[col] = (fig, ax)

    return figs_axes
#%%
# -----------------------------
# Main run
# -----------------------------


if __name__ == "__main__":
    path = r'D:\Users\c337191\Documents\unpleasant_nk_arithmetic'
    panel = build_panel(COUNTRIES, START_YEAR, END_YEAR)

    panel.to_csv(f"{path}/Data/public_liabilities_panel.csv", index=False)

    fig1, ax1 = plot_money_vs_debt(
        panel=panel,
        money_col="m1_gdp",
        title=f"M1 vs Public Debt, {START_YEAR}-{END_YEAR}",
        ylabel="M1 (% of GDP)",
        savepath=f"{path}/Figs/m1_vs_public_debt.png",
        connect_time_path=CONNECT_TIME_PATH,
        label_latest=LABEL_LATEST,
    )

    fig2, ax2 = plot_money_vs_debt(
        panel=panel,
        money_col="broad_money_gdp",
        title=f"Broad Money vs Public Debt, {START_YEAR}-{END_YEAR}",
        ylabel="Broad money, M2/M3 depending on availability (% of GDP)",
        savepath=f"{path}/Figs/broad_money_vs_public_debt.png",
        connect_time_path=CONNECT_TIME_PATH,
        label_latest=LABEL_LATEST,
    )
    panel = add_debt_money_share_columns(panel)
    panel.to_csv(f"{path}/Data/public_liabilities_panel.csv", index=False)
    figs_hist = plot_debt_money_ratio_histograms(
        panel=panel,
        bins=30,
        savepath_m1=f"{path}/Figs/hist_debt_to_m1.png",
        savepath_broad=f"{path}/Figs/hist_debt_to_broad_money.png",
        winsorize_q=0.99,
    )

    print("\nCoverage summary:")
    summary = (
        panel.groupby("country")
             .agg(
                 first_year=("year", "min"),
                 last_year=("year", "max"),
                 n_m1=("m1_gdp", lambda x: x.notna().sum()),
                 n_broad=("broad_money_gdp", lambda x: x.notna().sum()),
                 n_debt=("debt_gdp", lambda x: x.notna().sum()),
             )
             .reset_index()
    )
    print(summary.to_string(index=False))

    plt.show()
