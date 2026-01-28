from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
import warnings

from utilities.process_results import save_figure_for_paper, print_h5_structure, setup_matplotlib_for_paper
from matplotlib import rcParams

colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"

possible_plants = ["Vernasca", "Robilante", "Monselice", "Fanna"]
plant_analyzed = "Vernasca"
path_processed_data = Path("./dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
av_el_price = electricity_price_data["el_price_itNord"].mean()
electricity_price_norm = electricity_price_data["el_price_itNord"]/av_el_price

clinker_data = pd.read_excel(path_processed_data, sheet_name="clinker_production")
clinker_demand_norm = clinker_data[f"clinker_{plant_analyzed}"]/ clinker_data[f"clinker_{plant_analyzed}"].mean()

# ------------------------------------------------------------
# PAPER SETUP
# ------------------------------------------------------------
setup_matplotlib_for_paper("single")

# ------------------------------------------------------------
# ECDF helper
# ------------------------------------------------------------
def ecdf(series):
    x = np.sort(series.values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# ------------------------------------------------------------
# FIGURE & GRID (NO sharex!)
# ------------------------------------------------------------
fig, axs = plt.subplots(
    nrows=2,
    ncols=2
)

# Panel lettering helper
def panel_label(ax, label, title):
    ax.text(
        0.02, 0.95,
        f"{label}  {title}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=rcParams["axes.titlesize"],
        fontweight="bold"
    )

# ============================================================
# (a) ELECTRICITY PRICE — TIME SERIES
# ============================================================
axs[0, 0].plot(
    electricity_price_norm.index,
    electricity_price_norm,
    color=batlow_colors[0],
    alpha=0.35,
    linewidth=0.8
)
axs[0, 0].plot(
    electricity_price_norm.index,
    electricity_price_norm.rolling(24).mean(),
    color=batlow_colors[0],
    linewidth=1.5
)

axs[0, 0].set_ylabel("Normalized price [-]")
panel_label(axs[0, 0], "(a)", "Electricity price – time series")

# ============================================================
# (b) ELECTRICITY PRICE — ECDF
# ============================================================
x, y = ecdf(electricity_price_norm)
axs[0, 1].plot(x, y, color=batlow_colors[0], linewidth=1.5)

axs[0, 1].set_ylabel("Cumulative probability [-]")
axs[0, 1].set_ylim(0, 1)
panel_label(axs[0, 1], "(b)", "Electricity price – ECDF")

# ============================================================
# (c) CLINKER DEMAND — TIME SERIES
# ============================================================
axs[1, 0].plot(
    clinker_demand_norm.index,
    clinker_demand_norm,
    color=batlow_colors[2],
    alpha=0.35,
    linewidth=0.8
)
axs[1, 0].plot(
    clinker_demand_norm.index,
    clinker_demand_norm.rolling(24).mean(),
    color=batlow_colors[2],
    linewidth=1.5
)

axs[1, 0].set_xlabel("Time [h]")
axs[1, 0].set_ylabel("Clinker demand [-]")
panel_label(axs[1, 0], "(c)", "Clinker demand – time series")

# ============================================================
# (d) CLINKER DEMAND — ECDF
# ============================================================
x, y = ecdf(clinker_demand_norm)
axs[1, 1].plot(x, y, color=batlow_colors[2], linewidth=1.5)

axs[1, 1].set_xlabel("Clinker demand [-]")
axs[1, 1].set_ylabel("Cumulative probability [-]")
axs[1, 1].set_ylim(0, 1)
panel_label(
    axs[1, 1],
    "(d)",
    f"Clinker demand – ECDF ({plant_analyzed})"
)

# ------------------------------------------------------------
# FINALIZE & SAVE
# ------------------------------------------------------------
fig.tight_layout(pad=0.6)
save_figure_for_paper(fig, "cement_timeseries_and_ecdf_inputs", figures_path)

plt.show()