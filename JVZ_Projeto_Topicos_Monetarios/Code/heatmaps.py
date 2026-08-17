# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:19:40 2026
Heatmaps

@author: C337191
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

path = r'D:\Users\c337191\Documents\unpleasant_nk_arithmetic'
sims = pd.read_parquet(f'{path}/Data/sims_FTPL.parquet')

# Regarding steady state public liabilities (money and debt)
# Only the ratio matters for financing objs AND for debt and money deviations!
# However, for macro outcomes absolute values and ratios matter

# %% Helpers
def plot_heatmap(
    data,
    x=None,
    y=None,
    title=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    cmap="viridis",
    annotate=False,
    fmt=".2f",
):
    """
    Plot a simple heatmap.

    Parameters
    ----------
    data : array-like or pd.DataFrame
        2D data to plot.
    x, y : list-like, optional
        Labels for columns and rows. If data is a DataFrame and these are not
        provided, column/index labels are used.
    title, xlabel, ylabel, cbar_label : str, optional
        Plot labels.
    cmap : str
        Matplotlib colormap.
    annotate : bool
        Whether to write values inside each cell.
    fmt : str
        Number format for annotations.

    Returns
    -------
    fig, ax, im, cbar
        Matplotlib figure, axis, heatmap image, and colorbar objects.
    """

    # Handle pandas DataFrames gracefully
    if hasattr(data, "values"):
        if x is None:
            x = list(data.columns)
        if y is None:
            y = list(data.index)
        data = data.values

    data = np.asarray(data)

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    if x is not None:
        max_ticks = 10
        xticks = np.linspace(
            0,
            len(x)-1,
            min(max_ticks, len(x)),
            dtype=int
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [f"{x[i]:.2f}" for i in xticks],
            ha='right'
        )

    if y is not None:
        max_ticks = 10
        yticks = np.linspace(
            0,
            len(y)-1,
            min(max_ticks, len(y)),
            dtype=int
        )
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [f"{y[i]:.2f}" for i in yticks],
            ha='right'
        )

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center"
                )

    fig.tight_layout()

    return fig, ax, im, cbar

# %% Graphs

# Percentage of debt financing done by dilution (the rest is segniorage)
t = sims.query('debt_money_ratio == 1.2')[[
    'debt_fin_debt_dilution','debt_fin_pi_tax', 'debt_fin_money_creation', 'phi_pi'
    ]].set_index('phi_pi').sort_index()
#t = sims.query('phi_pi == 0.8910 and bbar==0.1')[['mean_pi_gap', 'mubar']].set_index('mubar').sort_index()
t.plot()
plt.show()

fin_df = sims.query(
    'debt_money_ratio < 10.1'
    ).groupby(
        ['debt_money_ratio', 'phi_pi']
        )[['debt_fin_debt_dilution',
           'debt_fin_pi_tax',
           'debt_fin_money_creation',
           'mean_mu_gap',
           'mean_b_gap']].mean().reset_index()

fin_df['mean_ell_gap'] = fin_df['mean_mu_gap'] + fin_df['mean_b_gap']

col_titles = {
    'mean_b_gap': 'govt. debt gap',
    'mean_mu_gap': 'money holdings gap',
    'mean_ell_gap': 'total public liabilities gap',
    'debt_fin_debt_dilution': 'debt share financed by dilution',
    'debt_fin_pi_tax': 'debt share financed by the inflation tax',
    'debt_fin_money_creation': 'debt share financed by money creation'
    }
for col, title in col_titles.items():
    heatmap_data = fin_df.pivot(
        index='debt_money_ratio',
        columns='phi_pi',
        values=col
    )
    
    fig, ax, im, cbar = plot_heatmap(
        heatmap_data,
        xlabel=r'$\phi_\pi$',
        ylabel='Debt/Money Ratio',
        cbar_label=title,
        cmap='magma'
    )
    plt.show()
    fig.savefig(f'{path}/Figs/Money_NK_FTPL/{col}.png', transparent=True)

# %% heleper plotter

def plot_gap_moments(
    df: pd.DataFrame,
    x_name: str = "phi_pi",
    scale: float = 100,
    title: str = "Mean Gaps as Monetary Policy Reactivity Changes",
    ylabel: str | None = None,
    add_taylor_line: bool = False,
    savepath: str | None = None,
):
    """
    Pretty line plot for a dataframe indexed by phi_pi and with columns like:
        mean_y_gap, mean_pi_gap, mean_i_gap

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with phi_pi as index or as a column.
    x_name : str
        Name of the x variable.
    scale : float
        100 converts decimals to percentage points.
        10_000 converts decimals to basis points.
        1 keeps raw values.
    title : str
        Plot title.
    ylabel : str | None
        Y-axis label. If None, inferred from scale.
    add_taylor_line : bool
        Adds vertical line at phi_pi = 1.
    savepath : str | None
        If provided, saves the figure.

    Returns
    -------
    fig, ax
    """

    data = df.copy()

    # Get x variable
    if x_name in data.columns:
        x = data[x_name].astype(float)
        data = data.drop(columns=[x_name])
    else:
        x = data.index.astype(float)

    data = data.sort_index()
    x = np.asarray(x)

    # Keep only numeric columns
    plot_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    label_map = {
        "mean_y_gap": "Output gap",
        "mean_pi_gap": "Inflation gap",
        "mean_i_gap": "Interest-rate gap",
    }

    if ylabel is None:
        if scale == 100:
            ylabel = "Mean gap, percentage points"
        elif scale == 10_000:
            ylabel = "Mean gap, basis points"
        else:
            ylabel = "Mean gap"

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in plot_cols:
        ax.plot(
            x,
            data[col].to_numpy() * scale,
            linewidth=2,
            label=label_map.get(col, col.replace("_", " ").title()),
        )

    if add_taylor_line:
        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=r"Taylor principle: $\phi_\pi = 1$",
        )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(r"Inflation response coefficient, $\phi_\pi$")
    ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    if scale == 100:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    elif scale == 10_000:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))

    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax
# %%
# The SHAPE is determined by the relative share
# as bbar/mubar increases:
# pi, i:, the convex part gets larger and the peak gets higher
# at the limit of mubar=0, pi and i are purely convex and explosive at phi_pi=1
# ygap: at bbar=0, it is drecreasing in phi_pi
# as bbar increase, a region of increasing ygap emerges
# from phi_pi=0 to 0.8 +-. After that, it is strongly decreasing, assimptoting 
# to zero at phi_pi=1
# The SUM changes the scale of the impact, but not the shape in phi_pi

# if money is M1, b/mu is ~3.5, but can be as high sa 10 (brazil)
# if money is M2, more well behaved. b/mu ~ 0.9, going from 0.2 to 1.5
# the sum is usually high, ranging from 1 to 2

# Thus, we want to plot a low-low summing to 1, low-high (0.5-1), high-low (1-0.5), high-high (2)
# and finally, verylow-high (0.5-1.5) for m1


dict_specs = {
    'low_low': [0.5, 0.5],
    'low_high': [0.5, 1.0],
    'high_low': [1.0, 0.5],
    'high_high': [1.0, 1.0],
    'verylow_high': [0.5, 1.5]
    }

for name, specs in dict_specs.items():
    plot_df = sims.query(f'mubar=={specs[0]} and bbar=={specs[1]}')[
    ['mean_y_gap', 'mean_pi_gap', 'mean_i_gap', 'phi_pi']
    ].set_index('phi_pi').sort_index()
    fig, ax = plot_gap_moments(
        plot_df,
        scale=100,
        title=rf"Fiscal Shock Impact on Macro Variables, $\bar \mu = {specs[0]}, \bar b = {specs[1]}$",
        savepath=f'{path}/Figs/Money_NK_FTPL/macro_{name}.png',
    )

t = t.rename({
    'debt_fin_debt_dilution': 'Dilution/Denominator Effect',
    'debt_fin_pi_tax': 'Inflation Tax',
    'debt_fin_money_creation': 'Money Creation'
    }, axis=1)
fig, ax = plot_gap_moments(t, scale=1, title='Debt Financing After a Deficit Episode\n money-to-debt ratio of 1.2',
                           ylabel='Share of Financing')
plt.show()











