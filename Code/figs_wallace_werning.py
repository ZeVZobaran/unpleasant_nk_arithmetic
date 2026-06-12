# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:56:58 2026

@author: c337191
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import zipfile

outdir = Path(r"D:\Users\c337191\Documents\unpleasant_mon_arit\figs")
outdir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Data
# ----------------------------

higher_tomorrow = pd.DataFrame({
    "t": range(1, 11),
    "infl_tight": [1.0043, 1.0089, 1.0150, 1.0227, 1.0326, 1.0449, 1.0601, 1.0781, 1.0989, 1.1221],
    "infl_loose": [1.0192, 1.0221, 1.0258, 1.0306, 1.0367, 1.0444, 1.0539, 1.0656, 1.0796, 1.0960],
    "money_tight": [0.2468, 0.2433, 0.2388, 0.2330, 0.2256, 0.2163, 0.2030, 0.1915, 0.1759, 0.1585],
    "money_loose": [0.2356, 0.2335, 0.2307, 0.2249, 0.2225, 0.2167, 0.2096, 0.2008, 0.1903, 0.1780],
})

higher_today = pd.DataFrame({
    "t": range(1, 11),
    "infl_tight": [1.0842, 1.0841, 1.0841, 1.0841, 1.0841, 1.0840, 1.0840, 1.0840, 1.0839, 1.0839],
    "infl_loose": [1.0825, 1.0808, 1.0789, 1.0768, 1.0743, 1.0716, 1.0684, 1.0647, 1.0605, 1.0556],
    "money_tight": [0.1448, 0.1448, 0.1449, 0.1449, 0.1449, 0.1450, 0.1450, 0.1450, 0.1451, 0.1451],
    "money_loose": [0.1469, 0.1490, 0.1514, 0.1540, 0.1571, 0.1606, 0.1641, 0.1691, 0.1744, 0.1805],
})

cases = {
    "higher_tomorrow": {
        "df": higher_tomorrow,
        "title": r"Higher inflation tomorrow",
        "theta_tight": r"0.01",
        "theta_loose": r"0.03",
    },
    "higher_today": {
        "df": higher_today,
        "title": r"Higher inflation today",
        "theta_tight": r"0.106",
        "theta_loose": r"0.120",
    },
}

# ----------------------------
# Plotting
# ----------------------------

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

def plot_series(df, y_tight, y_loose, ylabel, title, theta_tight, theta_loose, outfile):
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(df["t"], df[y_tight], marker="o", linewidth=1.9, label=rf"Tight money, $\theta={theta_tight}$")
    ax.plot(df["t"], df[y_loose], marker="s", linewidth=1.9, linestyle="--", label=rf"Loose money, $\theta={theta_loose}$")
    ax.set_title(title)
    ax.set_xlabel("Date $(t)$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(df["t"])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

files = []

plot_specs = [
    ("higher_tomorrow", "infl_tight", "infl_loose", "Inflation rate", "Higher inflation tomorrow: inflation", "sw_higher_tomorrow_inflation"),
    ("higher_tomorrow", "money_tight", "money_loose", "Per capita real money balances", "Higher inflation tomorrow: money", "sw_higher_tomorrow_money"),
    ("higher_today", "infl_tight", "infl_loose", "Inflation rate", "Higher inflation today: inflation", "sw_higher_today_inflation"),
    ("higher_today", "money_tight", "money_loose", "Per capita real money balances", "Higher inflation today: money", "sw_higher_today_money"),
]

for case_key, y_tight, y_loose, ylabel, title, stem in plot_specs:
    c = cases[case_key]
    outfile = outdir / stem
    plot_series(
        c["df"], y_tight, y_loose, ylabel, title,
        c["theta_tight"], c["theta_loose"], outfile
    )
    files.extend([outfile.with_suffix(".pdf"), outfile.with_suffix(".png")])

# Combined preview as one PNG/PDF; useful if the user wants one image instead of 4 included figures.
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))

combined_specs = [
    (axes[0, 0], higher_tomorrow, "infl_tight", "infl_loose", "Inflation rate", r"Higher inflation tomorrow", "0.01", "0.03"),
    (axes[0, 1], higher_tomorrow, "money_tight", "money_loose", "Real money balances", r"Higher inflation tomorrow", "0.01", "0.03"),
    (axes[1, 0], higher_today, "infl_tight", "infl_loose", "Inflation rate", r"Higher inflation today", "0.106", "0.120"),
    (axes[1, 1], higher_today, "money_tight", "money_loose", "Real money balances", r"Higher inflation today", "0.106", "0.120"),
]

for ax, df, y_tight, y_loose, ylabel, title, theta_tight, theta_loose in combined_specs:
    ax.plot(df["t"], df[y_tight], marker="o", linewidth=1.8, label=rf"Tight, $\theta={theta_tight}$")
    ax.plot(df["t"], df[y_loose], marker="s", linewidth=1.8, linestyle="--", label=rf"Loose, $\theta={theta_loose}$")
    ax.set_title(title)
    ax.set_xlabel("Date $(t)$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(df["t"])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")

fig.tight_layout()
combined_pdf = outdir / "sw_four_panel_charts.pdf"
combined_png = outdir / "sw_four_panel_charts.png"
fig.savefig(combined_pdf, bbox_inches="tight")
fig.savefig(combined_png, bbox_inches="tight")
plt.close(fig)
files.extend([combined_pdf, combined_png])



"""
Annotated Laffer-curve figures for Werning's interest-rate version of
Sargent-Wallace unpleasant arithmetic.

The object plotted is:
    R(i) = i/(1+i) * L(i)

where L(i) is money demand. The default choice below, L(i)=exp(-eta*i),
is purely illustrative and chosen to generate a clean hump-shaped curve.
"""
#%%


# ----------------------------
# 1. Basic objects
# ----------------------------

def L(i, eta=1.6):
    """Illustrative money demand: decreasing in nominal interest rate i."""
    return np.exp(-eta * i)


def R(i, eta=1.6):
    """Inflation-tax / seigniorage revenue: R(i) = i/(1+i) L(i)."""
    return (i / (1.0 + i)) * L(i, eta=eta)


def i_peak(eta=1.6, grid=None):
    """Numerical peak of the illustrative Laffer curve."""
    if grid is None:
        grid = np.linspace(1e-4, 1.0, 5000)
    return grid[np.argmax(R(grid, eta=eta))]


# ----------------------------
# 2. Plot helpers
# ----------------------------

def clean_axes(ax):
    """Remove top/right spines and make axes look like lecture-note axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$i_t$", fontsize=15, loc="right")
    ax.set_ylabel(r"$S(i)=\dfrac{i}{1+i}L(i_t)$", fontsize=15, rotation=0, labelpad=35)
    ax.yaxis.set_label_coords(0.04, 1.01)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_bounds(0, ax.get_ylim()[1])
    ax.spines["bottom"].set_bounds(0, ax.get_xlim()[1])

def draw_curve(ax, eta=1.6):
    """Draw the Laffer curve."""
    i_grid = np.linspace(0.0, 1.0, 800)
    ax.plot(i_grid, R(i_grid, eta=eta), linewidth=2.4)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.012, R(i_peak(eta), eta=eta) * 1.23)
    return i_grid


def mark_point(ax, x, label, eta=1.6, revenue_label=None):
    """Mark x, draw vertical and horizontal guides, and label the point."""
    y = R(x, eta=eta)
    ax.plot(x, y, marker="o", markersize=7)
    ax.vlines(x, 0, y, linestyles="--", linewidth=1.0)
    ax.hlines(y, 0, x, linestyles="--", linewidth=1.0)
    ax.plot(x, 0, marker="o", markersize=4)
    ax.text(x, -0.006, label, ha="center", va="top", fontsize=13)
    if revenue_label is not None:
        ax.text(-0.012, y, revenue_label, ha="right", va="center", fontsize=12)
    return y


def add_arrow(ax, xy_from, xy_to, text, text_xy, connectionstyle="arc3,rad=0.0", fontsize=11):
    """Add an arrow and a nearby text label."""
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.6,
            connectionstyle=connectionstyle
        )
    )
    ax.text(text_xy[0], text_xy[1], text, ha="center", va="center", fontsize=fontsize)


# ----------------------------
# 3. Figure 1
# ----------------------------
ETA = 3
def figure_result_1(output_path="werning_result_1_good_side.png", eta=ETA):
    """
    Result #1:
    Starting on the good side, a current cut i0 -> i1 lowers revenue.
    The intertemporal budget then requires a later increase i2 > i0,
    so Fisher ordering gives pi1 < pi0 < pi2.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    draw_curve(ax, eta=eta)
    clean_axes(ax)

    # Chosen all on the good side of the curve.
    i1, i0, i2 = 0.12, 0.25, 0.38
    y1 = mark_point(ax, i1, r"$i_1$", eta=eta, revenue_label=r"$S(i_1)$")
    y0 = mark_point(ax, i0, r"$i_0$", eta=eta, revenue_label=r"$S(i_0)$")
    y2 = mark_point(ax, i2, r"$i_2$", eta=eta, revenue_label=r"$S(i_2)$")


    add_arrow(
        ax,
        xy_from=(i0-0.01, y0*1.02),
        xy_to=(i1-0.01, y1*1.01),
        text=r"cut to $i_1$ today",
        text_xy=(0.15, 0.1),
        connectionstyle="arc3,rad=0.20",
        fontsize=12
    )

    add_arrow(
        ax,
        xy_from=(i1 + 0.01, y1 * 0.55),
        xy_to=(i2, y2 * 0.72),
        text=r"raise to $i_2$" + "\n" + "to recover shortfall + interest",
        text_xy=(0.3, 0.05),
        connectionstyle="arc3,rad=-0.28",
        fontsize=11
    )

    fig.subplots_adjust(bottom=0.32, left=0.10, right=0.98, top=0.90)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ----------------------------
# 4. Figure 2
# ----------------------------

def figure_result_2(output_path="werning_result_2_bad_side.png", eta=ETA):
    """
    Result #2:
    Move from i0 on the good side to i1 on the bad side, with i1 > i0
    but R(i1) < R(i0). A later adjustment to i2, with i0 < i2 < i1
    and R(i2) > R(i0), restores revenue. Fisher ordering then gives
    pi0 < pi2 < pi1.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    draw_curve(ax, eta=eta)
    clean_axes(ax)

    # Peak determines the border between good and bad sides.
    ip = i_peak(eta=eta)

    # i0 good side; i1 bad side with lower revenue; i2 between them
    # and chosen just to the right of the peak, with higher revenue than i0.
    i0, i2, i1 = 0.15, 0.3, 0.5
    y1 = mark_point(ax, i1, r"$i_1$", eta=eta, revenue_label=r"$S(i_1)$")
    y0 = mark_point(ax, i0, r"$i_0$", eta=eta, revenue_label=r"$S(i_0)$")
    y2 = mark_point(ax, i2, r"$i_2$", eta=eta, revenue_label=r"$S(i_2)$")
 
    # Mark peak lightly.
    ax.vlines(ip, 0, R(ip, eta=eta), linestyles=":", linewidth=1.0)
    ax.text(ip, -0.006, r"$\bar{i}$", ha="center", va="top", fontsize=12)

    add_arrow(
        ax,
        xy_from=(i0, y0),
        xy_to=(i1, y1),
        text=r"move to $i_1$",
        text_xy=(0.37, 0.068),
        connectionstyle="arc3,rad=0.12",
        fontsize=12
    )

    add_arrow(
        ax,
        xy_from=(i1 - 0.02, y1 * 1.05),
        xy_to=(i2, y2),
        text=r"later adjust to " + "\n" + "to recover revenue",
        text_xy=(0.45, 0.1),
        connectionstyle="arc3,rad=0.22",
        fontsize=11
    )


    fig.subplots_adjust(bottom=0.32, left=0.10, right=0.98, top=0.90)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ----------------------------
# 5. Generate files
# ----------------------------

if __name__ == "__main__":
    out_dir = Path(".")
    figure_result_1(outdir / "werning_result_1_good_side.png")
    figure_result_2(outdir / "werning_result_2_bad_side.png")
