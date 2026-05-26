# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:34:05 2026

@author: José
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import brentq
from pathlib import Path
outdir = Path(r'D:\Economia\Agenda Pesquisa\unpleasant_nk_arithmetic\Figs')

@dataclass
class NKFiscalParams:
    # NK parameters
    beta: float = 0.99
    sigma: float = 1.0
    kappa: float = 0.10

    # Fiscal parameters
    bbar: float = 0.80     # steady-state debt/output
    d0: float = 0.02       # one-time deficit shock, as share of output

    # Monetary experiment
    h: float = 0.010       # initial interest-rate deviation above neutral
    L: int = 10             # number of periods for post-t0 adjustment rate
    T: int = 60            # simulation horizon
    t_h: int = 1
    
    # Numerics
    root_bracket: tuple = (-0.01, 0.1)
    max_bracket_expansions: int = 20


def interest_rate_path(post_rate_dev: float, p: NKFiscalParams) -> np.ndarray:
    """
    Returns i_hat_t, deviations from the neutral/natural nominal rate.

    Experiment:
        i_hat_0 = h > 0
        i_hat_t = post_rate_dev for t = 1,...,L
        i_hat_t = 0 afterward
    """
    i = np.zeros(p.T + 1)
    
    i[0] = p.h
    for iterable in range(p.t_h):
        i[iterable] = p.h
    
    last_adjustment_period = min(p.L, p.T)
    if last_adjustment_period >= 1:
        i[p.t_h:last_adjustment_period + 1] = post_rate_dev
    return i


def solve_nk_block(i: np.ndarray, p: NKFiscalParams):
    """
    Solves the perfect-foresight NK block backward, imposing terminal conditions:

        x_{T+1} = 0
        pi_{T+1} = 0

    Equations:
        x_t  = x_{t+1} - (1/sigma)(i_t - pi_{t+1})
        pi_t = beta*pi_{t+1} + kappa*x_t
    """
    x = np.zeros(p.T + 2)
    pi = np.zeros(p.T + 2)

    for t in range(p.T, -1, -1):
        x[t] = x[t + 1] - (i[t] - pi[t + 1]) / p.sigma
        pi[t] = p.beta * pi[t + 1] + p.kappa * x[t]

    return x[:p.T + 1], pi[:p.T + 1]


def solve_debt_path(i: np.ndarray, pi: np.ndarray, p: NKFiscalParams):
    """
    Solves the government budget constraint forward.

    Linearized GBC:
        b_t = beta^{-1} b_{t-1}
              + beta^{-1} bbar * (i_{t-1} - pi_t)
              - s_t

    Fiscal rule:
        s_0 = -d0
        s_t = 0 for t >= 1

    Initial conditions:
        b_{-1} = 0
        i_{-1} = 0
    """
    b = np.zeros(p.T + 1)
    s = np.zeros(p.T + 1)
    s[0] = -p.d0

    b_lag = 0.0
    i_lag = 0.0

    for t in range(p.T + 1):
        b[t] = (1 / p.beta) * b_lag + (1 / p.beta) * p.bbar * (i_lag - pi[t]) - s[t]
        b_lag = b[t]
        i_lag = i[t]

    return b, s


def simulate_given_post_rate(post_rate_dev: float, p: NKFiscalParams) -> pd.DataFrame:
    """
    Simulates the full NK-fiscal system for a candidate post-t0 rate.
    """
    i = interest_rate_path(post_rate_dev, p)
    x, pi = solve_nk_block(i, p)
    b, s = solve_debt_path(i, pi, p)

    df = pd.DataFrame({
        "t": np.arange(p.T + 1),
        "i_dev": i,
        "pi": pi,
        "x": x,
        "b": b,
        "s": s
    })

    # Useful annualized approximations if periods are quarters.
    df["i_dev_ann_pct"] = 400 * df["i_dev"]
    df["pi_ann_pct"] = 400 * df["pi"]

    return df


def terminal_debt(post_rate_dev: float, p: NKFiscalParams) -> float:
    """
    Terminal debt residual. The equilibrium post-rate solves terminal_debt = 0.
    """
    df = simulate_given_post_rate(post_rate_dev, p)
    return df["b"].iloc[-1]


def solve_required_post_rate(p: NKFiscalParams):
    """
    Finds the post-t0 rate deviation ell such that b_T = 0.

    If rates are zero after L and inflation/output are terminally zero,
    boundedness requires debt to be zero by the terminal date.
    """
    lo, hi = p.root_bracket
    f_lo = terminal_debt(lo, p)
    f_hi = terminal_debt(hi, p)

    expansions = 0
    while abs(f_lo * f_hi) > 0 and expansions < p.max_bracket_expansions:
        lo *= 2
        hi *= 2
        f_lo = terminal_debt(lo, p)
        f_hi = terminal_debt(hi, p)
        expansions += 1
    if f_lo * f_hi > 0:
        raise RuntimeError(
            "Could not bracket a solution for the required post-t0 rate. "
            "Try increasing the bracket, increasing L, or changing calibration."
        )

    ell = brentq(lambda z: terminal_debt(z, p), lo, hi)
    df = simulate_given_post_rate(ell, p)

    return ell, df


def plot_sargent_wallace_style(df_eq: pd.DataFrame, df_no_adjust: pd.DataFrame = None):
    """
    Plots inflation, interest rate, and debt.
    If df_no_adjust is provided, it overlays the no-post-adjustment case.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)

    # Inflation
    axes[0].plot(df_eq["t"], df_eq["pi"], linewidth=2, label="Equilibrium adjustment")
    if df_no_adjust is not None:
        axes[0].plot(df_no_adjust["t"], df_no_adjust["pi"], linewidth=2, linestyle="--",
                     label="No post-t0 adjustment")
    axes[0].set_ylabel("Inflation\nannualized pp")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Interest rate
    axes[1].plot(df_eq["t"], df_eq["i_dev"], linewidth=2, label="Equilibrium adjustment")
    if df_no_adjust is not None:
        axes[1].plot(df_no_adjust["t"], df_no_adjust["i_dev"], linewidth=2, linestyle="--",
                     label="No post-t0 adjustment")
    axes[1].axhline(0, linewidth=1)
    axes[1].set_ylabel("Rate deviation\nannualized pp")
    axes[1].grid(True, alpha=0.3)

    # Debt
    axes[2].plot(df_eq["t"], df_eq["b"], linewidth=2, label="Equilibrium adjustment")
    if df_no_adjust is not None:
        axes[2].plot(df_no_adjust["t"], df_no_adjust["b"], linewidth=2, linestyle="--",
                     label="No post-t0 adjustment")
    axes[2].axhline(0, linewidth=1)
    axes[2].set_ylabel("Debt deviation")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Example run
# ------------------------------------------------------------
p = NKFiscalParams(
    beta=0.99,
    sigma=0.1,
    kappa=5,
    bbar=0.8,
    d0=0.1,
    h=0,
    L=10,
    T=30
)
ell, df_eq = solve_required_post_rate(p)

def gridsearch_tomorrow():
    possible_sigma = np.linspace(0.8, 1.6, 5)
    possible_kappa = np.linspace(1, 11, 11)
    possible_d0 =  np.linspace(0.01, 0.1, 11)
    possible_t_h =  np.linspace(1, 3, 3)
    possible_h = np.linspace(0, 0.1, 11)
    for sigma in possible_sigma:
        for kappa in possible_kappa:
            for d in possible_d0:
                for t_h in possible_t_h:
                    inf_t0 = []
                    for h in possible_h:
                        p.sigma = sigma
                        p.kappa = kappa
                        p.d0 = d
                        p.t_h = int(t_h)
                        p.h = h
                        ell, df_eq = solve_required_post_rate(p)
                        inf_t0.append(df_eq['pi'].iloc[0])
                    inf_t0_series = pd.Series(inf_t0).dropna()
                    diff_tight_loose = inf_t0_series.diff().dropna()
                    if (diff_tight_loose<0).any():
                        print('found it')
                        return p, inf_t0_series
    print('no cigar')
    return None, None
p, inf_t0_series = gridsearch_tomorrow()

inf_t0_series.plot()
# %%

p = NKFiscalParams(
    beta=0.99,
    sigma=1,
    kappa=5,
    bbar=0.8,
    d0=0.02,
    h=0,
    L=10,
    T=30
)

possible_sigma = [1, 0.8]
possible_h = {'loose': 0.05, 'tight': 0.06}
dfs = {}
for sigma in possible_sigma:
    # sigma = 1 implements higher today
    series = {}
    for name, h in possible_h.items():
        if sigma == 1:
            p.kappa = 0.1
        else:
            print('here')
            p.kappa = 5
        p.sigma = sigma
        p.h = h
        print(p)
        ell, df_eq = solve_required_post_rate(p)
        series[f'infl_{name}'] = df_eq['pi'].iloc[:5]
        series[f'rate_{name}'] = df_eq['i_dev'].iloc[:5]
        series[f'gap_{name}'] = df_eq['x'].iloc[:5]
    dfs[sigma] = pd.DataFrame(series)

higher_today = dfs[1].reset_index().rename(columns={'index': 't'})

higher_tomorrow = dfs[possible_sigma[1]].reset_index().rename(columns={'index': 't'})
higher_tomorrow[['infl_loose', 'infl_tight']].plot()
higher_today[['gap_loose', 'gap_tight']].plot()


#%% Plotting

cases = {
    "higher_tomorrow": {
        "df": higher_tomorrow,
        "title": r"Higher inflation tomorrow",
        "i_tight": r"0.05",
        "i_loose": r"0.00",
    },
    "higher_today": {
        "df": higher_today,
        "title": r"Higher inflation today",
        "i_tight": r"0.05",
        "i_loose": r"0.00",
    },
}
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

def plot_series(df, y_tight, y_loose, ylabel, title, i_tight, i_loose, outfile):
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(df["t"], df[y_tight], marker="o", linewidth=1.9, label=rf"High rate, $\tilde i_0 = {i_tight}$")
    ax.plot(df["t"], df[y_loose], marker="s", linewidth=1.9, linestyle="--", label=rf"Loose money, $\tilde i_0 = {i_loose}$")
    ax.set_title(title)
    ax.set_xlabel("Date $(t)$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(df["t"])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

files = []

plot_specs = [
    ("higher_tomorrow", "infl_tight", "infl_loose", "Inflation rate", "Higher inflation tomorrow: inflation", "nk_higher_tomorrow_inflation"),
    ("higher_tomorrow", "rate_tight", "rate_loose", "Nominal Interest Rate Deviation", "Higher inflation tomorrow: interest rate", "nk_higher_tomorrow_rate"),
    ("higher_today", "infl_tight", "infl_loose", "Inflation rate", "Higher inflation today: inflation", "nk_higher_today_inflation"),
    ("higher_today", "rate_tight", "rate_loose", "Nominal Interest Rate Deviation", "Higher inflation today: interest rate", "nk_higher_today_rate"),
]

for case_key, y_tight, y_loose, ylabel, title, stem in plot_specs:
    c = cases[case_key]
    outfile = outdir / stem
    plot_series(
        c["df"], y_tight, y_loose, ylabel, title,
        c["i_tight"], c["i_loose"], outfile
    )

# Combined preview as one PNG/PDF; useful if the user wants one image instead of 4 included figures.
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))

combined_specs = [
    (axes[0, 0], higher_tomorrow, "infl_tight", "infl_loose", "Inflation rate", r"Higher inflation tomorrow", "0.06", "0.05"),
    (axes[0, 1], higher_tomorrow, "rate_tight", "rate_loose", "Nominal Interest Rate Deviation", r"Higher inflation tomorrow", "0.06", "0.05"),
    (axes[1, 0], higher_today, "infl_tight", "infl_loose", "Inflation rate", r"Higher inflation today", "0.06", "0.05"),
    (axes[1, 1], higher_today, "rate_tight", "rate_loose", "Nominal Interest Rate Deviation", r"Higher inflation today", "0.06", "0.05"),
]

for ax, df, y_tight, y_loose, ylabel, title, i_tight, i_loose in combined_specs:
    ax.plot(df["t"], df[y_tight], marker="o", linewidth=1.8, label=rf"Tight, $\tilde i_0={i_tight}$")
    ax.plot(df["t"], df[y_loose], marker="s", linewidth=1.8, linestyle="--", label=rf"Loose, $\tilde i_0={i_loose}$")
    ax.set_title(title)
    ax.set_xlabel("Date $(t)$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(df["t"])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")

fig.tight_layout()
combined_png = outdir / "nk_four_panel_charts.png"
fig.savefig(combined_png, bbox_inches="tight")
plt.close(fig)


