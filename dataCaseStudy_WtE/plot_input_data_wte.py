from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
from matplotlib import rcParams
import warnings
from scipy.interpolate import interp1d
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper
import seaborn as sns
from matplotlib import rcParams
# Set global styling for the plots

colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"


json_wasteCHP = Path("./technologies_json/WasteCHP.json")
info_wasteCHP = json.loads(json_wasteCHP.read_text())
lhv = info_wasteCHP["Performance"]["LHV"]
th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
emission_factor = info_wasteCHP["Performance"]["emission_factor"]

path_processed_data_cement = Path("../dataCaseStudy_Cement/dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data_cement, sheet_name="electricity_prices")
max_el_price = electricity_price_data["el_price_itNord"].max()
electricity_price_norm = electricity_price_data["el_price_itNord"]/max_el_price

path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
plant_analyzed = "PAIP" # one between: "silla2", "gerbido", "PAIP", "piacenza"
co2_concentration = data["co2_concentration_"+plant_analyzed]
norm_heat_demand = data["normalized_heat_demand_milan"]
wasteProcessed_demand_norm = data["waste_in_PAIP"]
emissions = data[f"emission_{plant_analyzed}"]
average_conc = co2_concentration.mean()
emissions_norm = emissions/max(emissions)

other_data_path = Path("../adopt_net0")
other_data_path = (
        other_data_path
        / "database/templates/technology_data/Industrial/WasteCaL_data/wasteCaL_sheet.xlsx"
)

emission_factor_data = pd.read_excel(
    other_data_path, sheet_name="emission_factor_waste", index_col=0
)
possible_concentrations = emission_factor_data.columns.tolist()
interp = interp1d(possible_concentrations, emission_factor_data.loc["emission_factor_tco2_twaste"],
                          kind="linear",
                          fill_value="extrapolate")
emission_factor = interp(co2_concentration)


wasteProcessed_demand_norm = (emissions/emissions.max()) / emission_factor


# Plot time series

from matplotlib import rcParams

rolling_window = 24  # hours

# ------------------------------------------------------------
# PAPER SETUP
# ------------------------------------------------------------
setup_matplotlib_for_paper("double")

rolling_window = 24  # hours

# ------------------------------------------------------------
# ECDF helper
# ------------------------------------------------------------
def ecdf(series):
    x = np.sort(series.values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# ------------------------------------------------------------
# DATA STRUCTURE
# ------------------------------------------------------------
rows = [
    ("El. price [-]", electricity_price_norm, batlow_colors[0]),
    ("Heat demand [-]",       norm_heat_demand,       batlow_colors[1]),
    ("CO$_2$ emisssions [-]",   emissions_norm,  batlow_colors[2]),
    ("CO$_2$ conc. [%]", co2_concentration*100,      batlow_colors[3]),
]

# ------------------------------------------------------------
# FIGURE & GRID
# ------------------------------------------------------------
fig, axs = plt.subplots(
    nrows=4,
    ncols=2,
    sharex=False,
    sharey=False
)

# Column headers (explain the matrix once)
axs[0, 0].set_title("Time series")
axs[0, 1].set_title("Cumulative distribution")

# ============================================================
# PLOTTING
# ============================================================
for i, (ylabel, series, color) in enumerate(rows):

    # --- Time series (left)
    axs[i, 0].plot(
        series.index,
        series,
        color=color,
        alpha=0.35,
        linewidth=0.8
    )
    axs[i, 0].plot(
        series.index,
        series.rolling(rolling_window).mean(),
        color=color,
        linewidth=1.5
    )
    axs[i, 0].set_ylabel(ylabel)

    # --- ECDF (right)
    x, y = ecdf(series)
    axs[i, 1].plot(
        x, y,
        color=color,
        linewidth=1.5
    )
    axs[i, 1].set_ylim(0, 1)

# ------------------------------------------------------------
# AXIS LABELS (only where meaningful)
# ------------------------------------------------------------
axs[-1, 0].set_xlabel("Time [h]")
axs[-1, 1].set_xlabel("Value [-]")
axs[1, 1].set_ylabel("Cumulative probability [-]")

# ------------------------------------------------------------
# FINALIZE & SAVE
# ------------------------------------------------------------
fig.tight_layout(pad=0.6)
save_figure_for_paper(fig, "wte_timeseries_and_ecdf_inputs_matrix", figures_path)

plt.show()


# ##------------------------------------ fancy visualization ---------------------------------
# variables = {
#     "Electricity Price (norm)": electricity_price_norm,
#     "Normalized Heat Demand": norm_heat_demand,
#     "Waste Processed Demand": wasteProcessed_demand,
#     "CO2 Concentration": co2_concentration
# }
#
#
# # --- Figure setup ---
# sns.set_theme(style="darkgrid")
# fig, axes = plt.subplots(len(variables), 3, figsize=(18, 4*len(variables)))
# rolling_window = 24
# quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
#
# for row_idx, (name, series) in enumerate(variables.items()):
#     # Ensure datetime index
#     start_time = pd.Timestamp("2025-01-01 00:00")
#     series.index = pd.date_range(start=start_time, periods=len(series), freq="H")
#
#     # --- 1. Raw data + rolling quantiles ---
#     rolling_q = pd.DataFrame({q: series.rolling(rolling_window, min_periods=1).quantile(q) for q in quantiles})
#     ax = axes[row_idx,0]
#     ax.plot(series.index, series, color='lightgrey', alpha=0.5, label='Raw data')
#     # Rolling median
#     ax.plot(rolling_q[0.5].index, rolling_q[0.5], color=batlow_colors[2], label='Median')
#     # IQR
#     ax.fill_between(rolling_q[0.25].index, rolling_q[0.25], rolling_q[0.75],
#                     color=batlow_colors[2], alpha=0.3, label='IQR')
#     # 5–95%
#     ax.fill_between(rolling_q[0.05].index, rolling_q[0.05], rolling_q[0.95],
#                     color=batlow_colors[3], alpha=0.2, label='5-95%')
#     ax.set_title(f"{name} — Raw + Rolling Quantiles")
#     ax.set_xlabel("Time")
#     ax.set_ylabel(name)
#     if row_idx==0:
#         ax.legend(loc='upper right')
#
#     # --- 2. Hour-of-day heatmap ---
#     df_heat = pd.DataFrame({
#         "day": series.index.date,
#         "hour": series.index.hour,
#         "value": series.values
#     })
#     df_pivot = df_heat.pivot_table(index="hour", columns="day", values="value", aggfunc="mean")
#     ax = axes[row_idx,1]
#     sns.heatmap(df_pivot, cmap="viridis", cbar_kws={"label": name}, ax=ax)
#     ax.set_title(f"{name} — Hour-of-Day Heatmap")
#     ax.set_xlabel("Day")
#     ax.set_ylabel("Hour of Day")
#
#     # --- 3. Diurnal distribution (hourly violins) ---
#     df_hourly = series.groupby(series.index.hour).apply(list)
#     ax = axes[row_idx,2]
#     sns.violinplot(data=df_hourly.tolist(), palette=batlow_colors[:len(df_hourly)], ax=ax, inner='quartile')
#     ax.set_xticks(range(24))
#     ax.set_xticklabels(range(24))
#     ax.set_title(f"{name} — Diurnal Distribution")
#     ax.set_xlabel("Hour of Day")
#     ax.set_ylabel(name)
#
# plt.tight_layout()
plt.show()