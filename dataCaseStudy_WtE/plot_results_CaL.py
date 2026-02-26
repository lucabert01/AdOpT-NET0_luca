import h5py
from pathlib import Path
from adopt_net0.result_management.read_results import (
    print_h5_tree,
    extract_datasets_from_h5group,
)
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import rcParams
import warnings
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import rcParams


colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"


## -----------------  Electricity price --------------------------

explored_el_price = [100,125, 150, 175, 200] # average el prices explored in the analysis
rolling_av_hours = 1
import_price_RDF = 20

path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
av_el_price = data["el_price_itNord"].mean()
electricity_price_norm = data["el_price_itNord"]/av_el_price

json_WasteCaL_CCS = Path("./technologies_json/WasteCaL_CCS.json")
info_WasteCaL_CCS = json.loads(json_WasteCaL_CCS.read_text())
lhv = info_WasteCaL_CCS["Performance"]["LHV"]
lhv_rdf = info_WasteCaL_CCS["Performance"]["LHV_RDF"]
th_efficiency = info_WasteCaL_CCS["Performance"]["th_efficiency"]
el_efficiency = info_WasteCaL_CCS["Performance"]["el_efficiency"]
emission_factor = info_WasteCaL_CCS["Performance"]["emission_factor"]
emission_factor_rdf = info_WasteCaL_CCS["Performance"]["emission_factor_RDF"]
ccr = info_WasteCaL_CCS["Performance"]["capture_rate"]


num_cases = len(explored_el_price)
raw_results_path = Path("./raw_results/CaL")
# Get all directories that contain 'el_price' in the name
el_price_dirs = [d for d in raw_results_path.iterdir()
                 if d.is_dir() and "el_price" in d.name]

# Sort directories by name
dir_results_sorted = sorted(el_price_dirs)

# Get the most recent ones
file_names = [d.name for d in dir_results_sorted[-num_cases:]]
explored_el_price_str = [str(r) for r in explored_el_price]

results_summary = {}
for i in range(0,len(file_names)):
    file_path = raw_results_path / f"{file_names[i]}/optimization_results.h5"

    # Check if each explored_el_price[i] is in file_names[i]
    el_price_str = explored_el_price_str[i]
    results_summary[el_price_str] = {}
    if f"{el_price_str}" in file_names[i]:
        print(f"{el_price_str} found in {file_names[i]}")
    else:
        print(f"{el_price_str} NOT found in {file_names[i]}")


    def print_structure(name, obj):
        indent = "  " * name.count('/')
        obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
        print(f"{indent}{name} ({obj_type})")


    with h5py.File(file_path, 'r') as hdf_file:
        hdf_file.visititems(print_structure)


    with h5py.File(file_path, 'r') as hdf_file:
        df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
        df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
        df_design_network = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]))
    print(df_operation)

    co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]
    w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCaL_CCS')]
    boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG_existing')]
    w2e_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCaL_CCS')]
    boiler_operation = df_operation.loc[:,
                    ('technology_operation', 'period1', 'industrial_cluster', 'Boiler_Industrial_NG_existing')]
    heat_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','heat', 'demand')]

    waste_processed_out = w2e_operation['wasteProcessed_output']
    waste_in_rdf = w2e_operation['wasteInRDF_input']
    heat_out = w2e_operation['heat_output']
    el_out = w2e_operation['electricity_output']
    waste_out = w2e_operation['wasteProcessed_output']
    waste_in = w2e_operation['wasteIn_input']
    co2_captured_w2e = w2e_operation['CO2captured_output']
    emissions_w2e = waste_in * emission_factor +  waste_in_rdf * emission_factor_rdf
    fraction_co2_captured = sum(co2_captured_w2e) / sum(emissions_w2e)
    size_cal = w2e_design["size_cal"]
    fraction_size_cal = size_cal/ (max(emissions_w2e)*ccr)
    ccs_capacity_factor = sum(co2_captured_w2e)/(size_cal*8760)
    boiler_load_factor = sum(boiler_operation['heat_output'])/ sum(heat_demand)
    final_emissions = w2e_operation['emissions_pos']
    # economics
    el_price = electricity_price_norm * explored_el_price[i]
    capex_cal = w2e_design["capex_tot"]
    opex_fixed = w2e_design["opex_fixed"]
    opex_variable = w2e_design["opex_variable"] + sum(waste_in_rdf*import_price_RDF)
    revenue_el_cal = sum(w2e_operation['el_cal']*el_price)
    pipeline_cost = df_design_network['capex'].values.flatten()[0]
    storage_cost = co2_storage_design['opex_variable']
    transport_stor_cost = storage_cost + pipeline_cost


    results_summary[el_price_str]['size_cal'] = size_cal
    results_summary[el_price_str]['fraction_size_cal'] = fraction_size_cal
    results_summary[el_price_str]['ccs_capacity_factor'] = ccs_capacity_factor
    results_summary[el_price_str]['hourly_co2_captured'] = co2_captured_w2e
    results_summary[el_price_str]['hourly_el_tot_wte'] = el_out
    results_summary[el_price_str]['hourly_el_cal_only'] = w2e_operation['el_cal']
    results_summary[el_price_str]['hourly_el_wte_only'] = w2e_operation['el_wte_only']
    results_summary[el_price_str]['hourly_wte_heat_for_el'] = w2e_operation['el_wte_only']/el_efficiency
    results_summary[el_price_str]['hourly_emissions'] = emissions_w2e

    results_summary[el_price_str]['capex_tot'] = capex_cal
    results_summary[el_price_str]['opex_fixed'] = opex_fixed
    results_summary[el_price_str]['opex_variable'] = opex_variable
    results_summary[el_price_str]['transport_stor_cost'] = transport_stor_cost
    results_summary[el_price_str]['revenue_el_cal'] = revenue_el_cal
    results_summary[el_price_str]['tot_co2_avoided'] = sum(waste_in*emission_factor)-(sum(final_emissions))
    results_summary[el_price_str]['tot_co2_captured'] = sum(co2_captured_w2e)

    results_summary['hourly_boiler_heat_out'] = boiler_operation['heat_output']
    results_summary['boiler_load_factor'] = boiler_load_factor
    results_summary['hourly_heat_demand'] = heat_demand
    results_summary['hourly_heat_prod_wte'] = heat_out
    results_summary['hourly_wasteProcessed'] = waste_processed_out





# Extract fraction_size_cal values (convert pandas series to scalar)
el_prices = []
fraction_size_cal_values = []

for el_price in explored_el_price_str:
    size_series = results_summary[el_price]['fraction_size_cal']
    if hasattr(size_series, "iloc"):  # ensure it's a scalar
        size_val = float(size_series.iloc[0])
    else:
        size_val = float(size_series)
    el_prices.append(float(el_price))
    fraction_size_cal_values.append(size_val)

# Use exactly your defined colors
colors = batlow_colors[:len(el_prices)]
#
# plt.figure(figsize=(7,5))
#
# # Scatter plot with batlow colors
# for i, (x, y) in enumerate(zip(el_prices, fraction_size_cal_values)):
#     plt.scatter(x, y, color=colors[i], s=100, edgecolor=colors[i], zorder=3)
#
# # Connect points with a neutral line
# plt.plot(el_prices, fraction_size_cal_values, linestyle="--", color="gray", alpha=0.6, zorder=2)
#
# plt.xlabel("Electricity Price [€/MWh]")
# plt.ylabel("Fraction CO2 treated")
# plt.title("Fraction CaL Size vs Electricity Price")
#
# plt.show()
#
#
el_prices = []
capacity_factor_cal_values = []

for el_price in explored_el_price_str:
    size_series = results_summary[el_price]['ccs_capacity_factor']
    if hasattr(size_series, "iloc"):  # ensure it's a scalar
        size_val = float(size_series.iloc[0])
    else:
        size_val = float(size_series)
    el_prices.append(float(el_price))
    capacity_factor_cal_values.append(size_val)
#
# # Use exactly your defined colors
# colors = batlow_colors[:len(el_prices)]
#
# plt.figure(figsize=(7,5))
#
# # Scatter plot with batlow colors
# for i, (x, y) in enumerate(zip(el_prices, capacity_factor_cal_values)):
#     plt.scatter(x, y, color=colors[i], s=100, edgecolor=colors[i], zorder=3)
#
# # Connect points with a neutral line
# plt.plot(el_prices, capacity_factor_cal_values, linestyle="--", color="gray", alpha=0.6, zorder=2)
#
# plt.xlabel("Electricity price [€/MWh]")
# plt.ylabel("CaL load factor [-]")
# plt.title("Load factor CaL vs Electricity Price")
#
# plt.show()

# Plot fraction CaL and load factor at the same time


# ------------------------------------------------------------
# PAPER SETUP
# ------------------------------------------------------------
setup_matplotlib_for_paper()

# ------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------
fig, ax1 = plt.subplots()

# ============================================================
# LEFT AXIS — CaL size
# ============================================================
for i, (x, y) in enumerate(zip(el_prices, fraction_size_cal_values)):
    ax1.scatter(
        x, y,
        color=colors[1],
        marker="o",
        s=50,
        edgecolor=colors[1],
        zorder=3
    )

ax1.plot(
    el_prices,
    fraction_size_cal_values,
    linestyle="--",
    color="gray",
    alpha=0.6,
    zorder=2
)

ax1.set_xlabel("Electricity price [€/MWh]")
ax1.set_ylabel("CaL size [-]", color=colors[1])
ax1.set_ylim(0.85, 0.95)
ax1.tick_params(axis='y', labelcolor=colors[1])

# ============================================================
# RIGHT AXIS — CaL load factor
# ============================================================
ax2 = ax1.twinx()

for i, (x, y) in enumerate(zip(el_prices, capacity_factor_cal_values)):
    ax2.scatter(
        x, y,
        color=colors[2],
        marker="^",
        s=50,
        edgecolor=colors[2],
        zorder=3
    )

ax2.plot(
    el_prices,
    capacity_factor_cal_values,
    linestyle="--",
    color="black",
    alpha=0.6,
    zorder=2
)

ax2.set_ylabel("CaL load factor [-]", color=colors[2])
ax2.set_ylim(0.85, 0.95)
ax2.tick_params(axis='y', labelcolor=colors[2])

# ============================================================
# ANNOTATIONS (Replacing Legend)
# ============================================================

# 1. Size - Pointing to the 2nd point (index 1)
# xytext adjusted to be slightly further left (-15) and higher (+0.025)
# to ensure it sits clear of the point and the dashed line.
ax1.annotate(
    "Size",
    xy=(el_prices[1], fraction_size_cal_values[1]),
    xytext=(el_prices[1] - 15, fraction_size_cal_values[1] + 0.025),
    color=colors[1],
    fontweight='bold',
    arrowprops=dict(
        arrowstyle="->",
        color=colors[1],
        lw=1.2,
        connectionstyle="arc3,rad=.2",
        shrinkA=3,  # Buffer at the text end
        shrinkB=5   # Buffer at the point end to prevent overlap
    )
)

# 2. Load factor - Pointing to the 4th point (index 3)
# xytext adjusted to be to the right (+10) and slightly below (-0.02)
ax2.annotate(
    "Load factor",
    xy=(el_prices[3], capacity_factor_cal_values[3]),
    xytext=(el_prices[3] + 10, capacity_factor_cal_values[3] - 0.02),
    color=colors[2],
    fontweight='bold',
    arrowprops=dict(
        arrowstyle="->",
        color=colors[2],
        lw=1.2,
        connectionstyle="arc3,rad=-.2",
        shrinkA=3,
        shrinkB=5
    )
)

# ------------------------------------------------------------
# FINALIZE & SAVE
# ------------------------------------------------------------
fig.tight_layout(pad=0.6)
save_figure_for_paper(fig, "cal_load_factor_vs_size", figures_path)









## Plot the economics

capex_tot = []
opex_fixed = []
opex_variable = []
revenue_el_cal = []
correct_for_avoided = []
tot_co2_captured = []
abatement_cost = []
capture_cost = []
transport_stor_cost = []


economics = {
    "capex_tot": capex_tot,
    "opex_fixed": opex_fixed,
    "opex_variable": opex_variable,
    "transport_stor_cost": transport_stor_cost,
    "revenue_el_cal": revenue_el_cal,
}

emissions ={
    "correct_for_avoided": correct_for_avoided,
    "tot_co2_captured": tot_co2_captured,
}

for el_price in explored_el_price_str:
    # Emissions
    for emissions_param, storage_list in emissions.items():
        if emissions_param == "tot_co2_captured":
            val = results_summary[el_price]["tot_co2_captured"]
        elif emissions_param == "correct_for_avoided":
            if results_summary[el_price]["tot_co2_avoided"] > 0:
                val = (
                    results_summary[el_price]["tot_co2_captured"]
                    / results_summary[el_price]["tot_co2_avoided"]
                )
            else:
                val = 0
        storage_list.append(val)

    # Economics (€/tCO2)
    values = {}
    for economic_param, storage_list in economics.items():
        numerator = results_summary[el_price][economic_param]
        denominator = results_summary[el_price]["tot_co2_captured"]

        # Ensure we are dealing with a single float, not a Series/Array
        if hasattr(numerator, "__len__"):
            numerator = numerator[0]
        if hasattr(denominator, "__len__"):
            denominator = denominator[0]

        val = (numerator / denominator) if denominator > 0 else 0
        storage_list.append(val)
        values[economic_param] = val

    # capture cost
    if results_summary[el_price]["tot_co2_avoided"] > 0:
        capture = (
            values["capex_tot"]
            + values["opex_fixed"]
            + values["opex_variable"]
            + values["transport_stor_cost"]
            - values["revenue_el_cal"]
        )
    else:
        capture = 0

    capture_cost.append(capture)



# # Scatter points for capture cost
# for i, (x, y) in enumerate(zip(el_prices, capture_cost)):
#     plt.scatter(x, y, color=colors[i], marker="s", s=100, edgecolor=colors[i], zorder=3, label="Capture Cost" if i == 0 else "")
#
# # Connect points with a dashed line
# plt.plot(el_prices, capture_cost, linestyle="--", color="black", alpha=0.6, zorder=2)
#
# # Labels
# plt.xlabel("Electricity Price [€/MWh]")
# plt.ylabel("Cost [€/tCO₂]")
#
# # Legend
# plt.legend()


# BArchart
labels = [
    "CAPEX",
    "OPEX fixed",
    "OPEX variable",
    "Transport & storage",
    "El. revenues",
]
item_colors = {
    "CAPEX": batlow_colors[0],
    "OPEX fixed": batlow_colors[1],
    "OPEX variable": batlow_colors[2],
    "Transport & storage": batlow_colors[3],
    "El. revenues": batlow_colors[4],
}


capex   = np.array(economics["capex_tot"], dtype=float).ravel()
opex_f  = np.array(economics["opex_fixed"], dtype=float).ravel()
opex_v  = np.array(economics["opex_variable"], dtype=float).ravel()
t_s     = np.array(economics["transport_stor_cost"], dtype=float).ravel()
revenue = -np.array(economics["revenue_el_cal"], dtype=float).ravel()

correct_factor = np.array(emissions["correct_for_avoided"], dtype=float).ravel()

capex_corr   = np.array(economics["capex_tot"], dtype=float).ravel() * correct_factor
opex_f_corr  = np.array(economics["opex_fixed"], dtype=float).ravel() * correct_factor
opex_v_corr  = np.array(economics["opex_variable"], dtype=float).ravel() * correct_factor
t_s_corr     = np.array(economics["transport_stor_cost"], dtype=float).ravel() * correct_factor
revenue_corr = -np.array(economics["revenue_el_cal"], dtype=float).ravel() * correct_factor


width = 0.35
x = np.arange(len(el_prices))
setup_matplotlib_for_paper("single")

def stacked_bar(ax, x_pos, capex, opex_f, opex_v, t_s, revenue,
                colors, hatch_pattern=None, return_tops=False):

    bottom = np.zeros(len(x_pos), dtype=float)

    # Positive cost stack
    ax.bar(x_pos, capex, width, bottom=bottom,
           facecolor=colors["CAPEX"], edgecolor='black', linewidth=0.8, hatch=hatch_pattern)
    bottom += capex

    ax.bar(x_pos, opex_f, width, bottom=bottom,
           facecolor=colors["OPEX fixed"], edgecolor='black', linewidth=0.8, hatch=hatch_pattern)
    bottom += opex_f

    ax.bar(x_pos, opex_v, width, bottom=bottom,
           facecolor=colors["OPEX variable"], edgecolor='black', linewidth=0.8, hatch=hatch_pattern)
    bottom += opex_v

    ax.bar(x_pos, t_s, width, bottom=bottom,
           facecolor=colors["Transport & storage"], edgecolor='black', linewidth=0.8, hatch=hatch_pattern)
    bottom += t_s

    # Revenue plotted relative to zero
    ax.bar(x_pos, revenue, width, bottom=0,
           facecolor=colors["El. revenues"],
           edgecolor='black',
           linewidth=0.8,
           hatch=hatch_pattern)

    if return_tops:
        return bottom + revenue  # TRUE total including revenue

# --- Plot figure ---
fig_width, fig_height = setup_matplotlib_for_paper(column="single")
fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # slightly bigger figure if needed

# Corrected bars: LEFT, solid
tops_corrected = stacked_bar(ax, x - width/2,
                             capex_corr, opex_f_corr, opex_v_corr, t_s_corr, revenue_corr,
                             item_colors,
                             hatch_pattern=None,
                             return_tops=True)

# Non-corrected bars: RIGHT, hatch
stacked_bar(ax, x + width/2,
            capex, opex_f, opex_v, t_s, revenue,
            item_colors,
            hatch_pattern='////')

# Marker for CO2 capture cost: dot for all bars
for xi, vals_corr, vals_orig in zip(x, zip(capex_corr, opex_f_corr, opex_v_corr, t_s_corr, revenue_corr),
                                    zip(capex, opex_f, opex_v, t_s, revenue)):
    total_corr = sum(vals_corr)
    total_orig = sum(vals_orig)
    ax.scatter(xi - width/2, total_corr, color='black', marker='o', s=50, zorder=5)
    ax.scatter(xi + width/2, total_orig, color='black', marker='o', s=50, zorder=5)

# Axes
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(el_prices)
ax.set_xlabel("Electricity price [€/MWh]")
ax.set_ylabel("Cost [€/tCO$_2$]")

# Expand y-limit for legend space
all_totals = np.concatenate([
    capex + opex_f + opex_v + t_s + revenue,
    capex_corr + opex_f_corr + opex_v_corr + t_s_corr + revenue_corr
])

ymax = all_totals.max() * 1.15
ymin = min(all_totals.min() * 1.15, 0)

ax.set_ylim(-230, 400)

ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# -----------------------------
# Legends (bottom)
# -----------------------------

# Legend 1: Cost component legend (colors) - bottom-left
handles_components = [mpatches.Patch(facecolor=c, edgecolor='black', label=l) for l, c in item_colors.items()]
labels_components = [l for l in item_colors.keys()]

legend1 = ax.legend(handles_components,
                    labels_components,
                    ncol=2,
                    frameon=False,
                    loc='lower left',
                    bbox_to_anchor=(0.02, 0.02),
                    borderaxespad=0.0)
ax.add_artist(legend1)

# Legend 2: Scenario + Net Cost marker - bottom-right
scenario_handles = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch=None, label='CO$_2$ avoidance cost'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='////', label='CO$_2$ capture cost'),
    # Add the Circle marker here
    Line2D([0], [0], marker='o', color='none', label='Net cost',
           markerfacecolor='black', markeredgecolor='white',
           markeredgewidth=0.5, markersize=8)
]

legend2 = ax.legend(handles=scenario_handles,
                    frameon=False,
                    loc='lower right',
                    bbox_to_anchor=(0.98, 0.02),
                    borderaxespad=0.0)

plt.tight_layout(pad=0.6)
save_figure_for_paper(fig, "cal_cost_breakdown", figures_path)

plt.show()



