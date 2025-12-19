from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
from matplotlib import rcParams
import warnings
from scipy.interpolate import interp1d
from utilities.process_results import save_figure_for_paper, print_h5_structure
import seaborn as sns

colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']


json_wasteCHP = Path("./technologies_json/WasteCHP.json")
info_wasteCHP = json.loads(json_wasteCHP.read_text())
lhv = info_wasteCHP["Performance"]["LHV"]
th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
emission_factor = info_wasteCHP["Performance"]["emission_factor"]

path_processed_data_cement = Path("../dataCaseStudy_Cement/dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data_cement, sheet_name="electricity_prices")
av_el_price = electricity_price_data["el_price_itNord"].mean()
electricity_price_norm = electricity_price_data["el_price_itNord"]/av_el_price

path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
plant_analyzed = "PAIP" # one between: "silla2", "gerbido", "PAIP", "piacenza"
co2_concentration = data["co2_concentration_"+plant_analyzed]
norm_heat_demand = data["normalized_heat_demand_milan"]
emissions = data[f"emission_{plant_analyzed}"]
average_conc = co2_concentration.mean()


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


wasteProcessed_demand = emissions / emission_factor


# Plot time series

rolling_window = 24  # hours

fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

# --- Normalized electricity price ---
axs[0].plot(electricity_price_norm.index,
            electricity_price_norm,
            color=batlow_colors[0],
            alpha=0.4,
            linewidth=0.8)

axs[0].plot(electricity_price_norm.index,
            electricity_price_norm.rolling(rolling_window).mean(),
            color=batlow_colors[0],
            linewidth=2)

axs[0].set_title("Normalized electricity price")


# --- Normalized heat demand ---
axs[1].plot(norm_heat_demand.index,
            norm_heat_demand,
            color=batlow_colors[1],
            alpha=0.4,
            linewidth=0.8)

axs[1].plot(norm_heat_demand.index,
            norm_heat_demand.rolling(rolling_window).mean(),
            color=batlow_colors[1],
            linewidth=2)

axs[1].set_title("Normalized heat demand")


# --- Waste processed demand ---
axs[2].plot(wasteProcessed_demand.index,
            wasteProcessed_demand,
            color=batlow_colors[2],
            alpha=0.4,
            linewidth=0.8)

axs[2].plot(wasteProcessed_demand.index,
            wasteProcessed_demand.rolling(rolling_window).mean(),
            color=batlow_colors[2],
            linewidth=2)

axs[2].set_title("Waste processed demand")


# --- CO₂ concentration ---
axs[3].plot(co2_concentration.index,
            co2_concentration,
            color=batlow_colors[3],
            alpha=0.4,
            linewidth=0.8)

axs[3].plot(co2_concentration.index,
            co2_concentration.rolling(rolling_window).mean(),
            color=batlow_colors[3],
            linewidth=2)

axs[3].set_title("CO₂ concentration")

axs[-1].set_xlabel("Time")

plt.tight_layout()


# Plot cumulative distribution
def ecdf(series):
    x = np.sort(series.values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=False)

x, y = ecdf(electricity_price_norm)
axs[0].plot(x, y, color=batlow_colors[0])
axs[0].set_title("ECDF – Normalized electricity price")

x, y = ecdf(norm_heat_demand)
axs[1].plot(x, y, color=batlow_colors[1])
axs[1].set_title("ECDF – Normalized heat demand")

x, y = ecdf(wasteProcessed_demand)
axs[2].plot(x, y, color=batlow_colors[2])
axs[2].set_title("ECDF – Waste processed demand")

x, y = ecdf(co2_concentration)
axs[3].plot(x, y, color=batlow_colors[3])
axs[3].set_title("ECDF – CO₂ concentration")

axs[-1].set_xlabel("Value")

plt.tight_layout()




##------------------------------------ fancy visualization ---------------------------------
variables = {
    "Electricity Price (norm)": electricity_price_norm,
    "Normalized Heat Demand": norm_heat_demand,
    "Waste Processed Demand": wasteProcessed_demand,
    "CO2 Concentration": co2_concentration
}


# --- Figure setup ---
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(len(variables), 3, figsize=(18, 4*len(variables)))
rolling_window = 24
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

for row_idx, (name, series) in enumerate(variables.items()):
    # Ensure datetime index
    start_time = pd.Timestamp("2025-01-01 00:00")
    series.index = pd.date_range(start=start_time, periods=len(series), freq="H")

    # --- 1. Raw data + rolling quantiles ---
    rolling_q = pd.DataFrame({q: series.rolling(rolling_window, min_periods=1).quantile(q) for q in quantiles})
    ax = axes[row_idx,0]
    ax.plot(series.index, series, color='lightgrey', alpha=0.5, label='Raw data')
    # Rolling median
    ax.plot(rolling_q[0.5].index, rolling_q[0.5], color=batlow_colors[2], label='Median')
    # IQR
    ax.fill_between(rolling_q[0.25].index, rolling_q[0.25], rolling_q[0.75],
                    color=batlow_colors[2], alpha=0.3, label='IQR')
    # 5–95%
    ax.fill_between(rolling_q[0.05].index, rolling_q[0.05], rolling_q[0.95],
                    color=batlow_colors[3], alpha=0.2, label='5-95%')
    ax.set_title(f"{name} — Raw + Rolling Quantiles")
    ax.set_xlabel("Time")
    ax.set_ylabel(name)
    if row_idx==0:
        ax.legend(loc='upper right')

    # --- 2. Hour-of-day heatmap ---
    df_heat = pd.DataFrame({
        "day": series.index.date,
        "hour": series.index.hour,
        "value": series.values
    })
    df_pivot = df_heat.pivot_table(index="hour", columns="day", values="value", aggfunc="mean")
    ax = axes[row_idx,1]
    sns.heatmap(df_pivot, cmap="viridis", cbar_kws={"label": name}, ax=ax)
    ax.set_title(f"{name} — Hour-of-Day Heatmap")
    ax.set_xlabel("Day")
    ax.set_ylabel("Hour of Day")

    # --- 3. Diurnal distribution (hourly violins) ---
    df_hourly = series.groupby(series.index.hour).apply(list)
    ax = axes[row_idx,2]
    sns.violinplot(data=df_hourly.tolist(), palette=batlow_colors[:len(df_hourly)], ax=ax, inner='quartile')
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_title(f"{name} — Diurnal Distribution")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(name)

plt.tight_layout()
plt.show()