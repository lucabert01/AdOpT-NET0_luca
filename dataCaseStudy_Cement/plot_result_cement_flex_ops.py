import h5py
from pathlib import Path
from adopt_net0.result_management.read_results import (
    print_h5_tree,
    extract_datasets_from_h5group,
)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
import warnings
import seaborn as sns
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper
from matplotlib import rcParams
import pprint
import math


colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"


## -----------------  Carbon and electricity price --------------------------
explored_std_el = [1, 2]
explored_el_price = [50, 150] # average el prices explored in the analysis
explored_tec = ["mea","mea_inflex", "oxy", "oxy_inflex"]
cost_extra_fuel = 15

path_processed_data = Path("./dataSources/data_processed.xlsx")
data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
av_el_price = data["el_price_itNord"].mean()
electricity_price_norm = data["el_price_itNord"]/av_el_price
json_cement = Path("./technologies_json/CementEmitter.json")
info_cement = json.loads(json_cement.read_text())
emission_factor_clinker_baseline = info_cement["Performance"]["emission_factor"]# tCo2/tClinker, without oxyfuel calciner
json_heat_pump = Path("./technologies_json/HeatPump.json")
info_heat_pump = json.loads(json_heat_pump.read_text())
cop_hp = info_heat_pump["Performance"]["performance"]["out"]["heat"][1]
json_oxy_ccs = Path("./technologies_json/CementHybridCCS.json")
info_oxy_ccs = json.loads(json_oxy_ccs.read_text())
emission_factor_clinker_oxy = info_oxy_ccs["Performance"]["performance"]["tCO2_tclinker"]

num_el_prices = len(explored_el_price)
num_std_el = len(explored_std_el)
num_tec = len(explored_tec)
raw_results_path = Path("./raw_results/flexible_ops")
explored_std_el_str = [str(r) for r in explored_std_el]
explored_el_price_str = [str(r) for r in explored_el_price]
explored_tec_str = explored_tec
results_summary = {}

for i_tec in range(0, num_tec):
    tec_str = explored_tec[i_tec]
    results_summary[tec_str] = {}

    # Logic to separate "standard" from "inflex" versions
    tec_dirs = []
    for d in raw_results_path.iterdir():
        if not d.is_dir():
            continue

        # Case 1: Searching for "inflex" (e.g., "mea_inflex")
        if "_inflex" in tec_str:
            if tec_str in d.name:
                tec_dirs.append(d)

        # Case 2: Searching for standard (e.g., "mea" or "oxy")
        # We check if it contains the tech name BUT NOT the _inflex suffix
        else:
            if tec_str in d.name and f"{tec_str}_inflex" not in d.name:
                tec_dirs.append(d)

    # Sort and slice as you did before
    dir_results_sorted = sorted(tec_dirs)
    tec_names = [d.name for d in dir_results_sorted[-num_el_prices * num_std_el:]]
    for j in range(0,num_std_el):
        std_el = explored_std_el[j]
        std_el_str = f"std_{explored_std_el_str[j]}"
        results_summary[tec_str][std_el_str] = {}

        # Get all file names that contain 'std_el_str' in the name
        std_el_dirs = [d for d in tec_names if std_el_str in d]

        # Sort them by name
        dir_results_sorted = sorted(std_el_dirs)

        # Get the most recent ones at the value of carbon tax std_el_str
        std_el_names = [d for d in dir_results_sorted[-num_el_prices:]]
        for i in range(0,num_el_prices):
            el_price = explored_el_price[i] * electricity_price_norm

            file_path = raw_results_path / f"{std_el_names[i]}/optimization_results.h5"

            # Check if each explored_el_price[i] is in file_names[i]
            el_price_str = f"el_price_{explored_el_price[i]}"
            results_summary[tec_str][std_el_str][el_price_str] = {}
            if f"el_price_{el_price_str}" in std_el_names[i]:
                print(f"{el_price_str} found in {std_el_names[i]}")
            else:
                print(f"{el_price_str} NOT found in {std_el_names[i]}")

            with h5py.File(file_path, 'r') as hdf_file:
                df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
                df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
                df_design_network = pd.DataFrame(extract_datasets_from_h5group(
                    hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]))
            #print(df_operation)
            if "oxy" not in tec_str:
                cement_mea_design = df_design.loc[:, ('industrial_cluster', 'CementEmitter')]
                cement_mea_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
                heat_pump_design = df_design.loc[:, ('industrial_cluster', 'HeatPump')]
                heat_pump_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'HeatPump')]
            if "mea" not in tec_str:
                cement_oxy_design = df_design.loc[:, ('industrial_cluster', 'CementHybridCCS')]
                cement_oxy_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementHybridCCS')]

            co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]
            clinker_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','clinker', 'demand')]
            emissions_cement = clinker_demand * emission_factor_clinker_baseline
            pipeline_cost = df_design_network['capex'].values.flatten()[0]
            storage_cost = co2_storage_design['opex_variable']
            transport_stor_cost = storage_cost + pipeline_cost

            # economics
            el_price = electricity_price_norm*explored_el_price[i]
            if tec_str in ["both", "both_inflex"]:
                if cement_mea_design["size_ccs"].iloc[0] > 0:

                    type_installed = "MEA"
                    capex = cement_mea_design["capex_tot"] + heat_pump_design["capex_tot"]
                    opex_fixed = cement_mea_design["opex_fixed"]+ heat_pump_design["opex_fixed"]
                    opex_variable = cement_mea_design["opex_variable"]
                    energy_cost = sum(cement_mea_operation["electricity_var_input_ccs"]*el_price) + sum(cement_mea_operation["heat_var_input_ccs"]/cop_hp*el_price)
                    co2_captured = cement_mea_operation['CO2captured_var_output_ccs']
                    tot_co2_avoided = sum(cement_mea_operation["clinker_output"]*emission_factor_clinker_baseline) - sum(cement_mea_operation["emissions_pos"])
                    clinker_output = cement_mea_operation["clinker_output"]

                elif cement_oxy_design["size"].iloc[0] > 0:
                    if cement_oxy_design["size_mea"].iloc[0] > 0:
                        type_installed = "Oxyfuel + PCC"
                    else:
                        type_installed = "Partial oxyfuel"

                    capex = cement_oxy_design["capex_tot"]
                    opex_fixed = cement_oxy_design["opex_fixed"]
                    opex_variable = cement_oxy_design["opex_variable"]
                    co2_captured = cement_oxy_operation['CO2captured_output']
                    energy_cost = sum(cement_oxy_operation["electricity_input"]*el_price) + sum(cement_oxy_operation["extra_fuel_input"]*cost_extra_fuel)
                    tot_co2_avoided = sum(cement_oxy_operation["clinker_output"]*emission_factor_clinker_baseline) - sum(cement_oxy_operation["emissions_pos"])
                    clinker_output = cement_oxy_operation["clinker_output"]
                else:
                    type_installed = "none"
                    capex = 0
                    opex_fixed = 0
                    opex_variable = 0
                    co2_captured = 0
                    energy_cost = 0
                    tot_co2_avoided = 0
                    clinker_output = 0

            if tec_str in ["oxy", "oxy_inflex"]:
                if cement_oxy_design["size"].iloc[0] > 0:
                    if cement_oxy_design["size_mea"].iloc[0] > 0:
                        type_installed = "Oxyfuel + PCC"
                    else:
                        type_installed = "Partial oxyfuel"

                    capex = cement_oxy_design["capex_tot"]
                    opex_fixed = cement_oxy_design["opex_fixed"]
                    opex_variable = cement_oxy_design["opex_variable"]
                    co2_captured = cement_oxy_operation['CO2captured_output']
                    energy_cost = sum(cement_oxy_operation["electricity_input"] * el_price) + sum(
                        cement_oxy_operation["extra_fuel_input"] * cost_extra_fuel)
                    tot_co2_avoided = sum(cement_oxy_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
                        cement_oxy_operation["emissions_pos"])
                    clinker_output = cement_oxy_operation["clinker_output"]

                else:
                    type_installed = "none"
                    capex = 0
                    opex_fixed = 0
                    opex_variable = 0
                    co2_captured = 0
                    energy_cost = 0
                    tot_co2_avoided = 0
                    clinker_output = 0

            if tec_str in ["mea", "mea_inflex"]:
                if cement_mea_design["size_ccs"].iloc[0] > 0:

                    type_installed = "MEA"
                    capex = cement_mea_design["capex_tot"] + heat_pump_design["capex_tot"]
                    opex_fixed = cement_mea_design["opex_fixed"]+ heat_pump_design["opex_fixed"]
                    opex_variable = cement_mea_design["opex_variable"]
                    energy_cost = sum(cement_mea_operation["electricity_var_input_ccs"] * el_price) + sum(
                        cement_mea_operation["heat_var_input_ccs"] / cop_hp * el_price)
                    co2_captured = cement_mea_operation['CO2captured_var_output_ccs']
                    tot_co2_avoided = sum(cement_mea_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
                        cement_mea_operation["emissions_pos"])
                    clinker_output = cement_mea_operation["clinker_output"]

                else:
                    type_installed = "none"
                    capex = 0
                    opex_fixed = 0
                    opex_variable = 0
                    co2_captured = 0
                    energy_cost = 0
                    tot_co2_avoided = 0
                    clinker_output = 0

            tot_co2_captured = (sum(co2_captured) if not isinstance(co2_captured, int) else pd.Series([0]))
            results_summary[tec_str][std_el_str][el_price_str]['hourly_co2_captured'] = co2_captured
            results_summary[tec_str][std_el_str][el_price_str]['hourly_clinker_output'] = clinker_output
            results_summary[tec_str][std_el_str][el_price_str]['hourly_clinker_demand'] = clinker_demand
            results_summary[tec_str][std_el_str][el_price_str]['capex_tot'] = capex
            results_summary[tec_str][std_el_str][el_price_str]['opex_fixed'] = opex_fixed
            results_summary[tec_str][std_el_str][el_price_str]['opex_variable'] = opex_variable
            results_summary[tec_str][std_el_str][el_price_str]['energy_cost'] = energy_cost
            results_summary[tec_str][std_el_str][el_price_str]['transport_stor_cost'] = transport_stor_cost
            results_summary[tec_str][std_el_str][el_price_str]['tot_co2_captured'] = tot_co2_captured
            results_summary[tec_str][std_el_str][el_price_str]['tot_co2_avoided'] = tot_co2_avoided
            results_summary[tec_str][std_el_str][el_price_str]['correct_for_avoided'] = tot_co2_captured/tot_co2_avoided
            results_summary[tec_str][std_el_str][el_price_str]['cost_of_avoided'] = ((capex + opex_fixed + opex_variable + energy_cost + transport_stor_cost)
                                                                                /tot_co2_avoided if not isinstance(co2_captured, int) else pd.Series([0]))
            results_summary[tec_str][std_el_str][el_price_str]['type_installed'] = type_installed

for j in range(0, num_std_el):
    std_el = explored_std_el[j]
    std_el_str = f"std_{explored_std_el_str[j]}"
    for i in range(0, num_el_prices):
        el_price = explored_el_price[i] * electricity_price_norm
        el_price_str = f"el_price_{explored_el_price[i]}"
        cost_avoided_mea = results_summary["mea"][std_el_str][el_price_str]['cost_of_avoided']
        cost_avoided_mea_inflex = results_summary["mea_inflex"][std_el_str][el_price_str]['cost_of_avoided']
        cost_avoided_oxy = results_summary["oxy"][std_el_str][el_price_str]['cost_of_avoided']
        cost_avoided_oxy_inflex = results_summary["oxy_inflex"][std_el_str][el_price_str]['cost_of_avoided']
        results_summary["mea"][std_el_str][el_price_str]['delta_cost_abatement'] = cost_avoided_mea- cost_avoided_mea_inflex
        results_summary["oxy"][std_el_str][el_price_str]['delta_cost_abatement'] = cost_avoided_oxy - cost_avoided_oxy_inflex
# ------------------------------------------------------------

# Define types and colors once
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E',
                 '#F7D03C', '#D491B8', '#012E4D']
types = ["none", "MEA", "Partial oxyfuel", "Oxyfuel + PCC"]
type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}

cost_matrix = {}
type_matrix = {}
cost_difference_matrix = {}

# Loop through each technology scenario
for tec_str in explored_tec_str:
    data = []
    for ct in explored_std_el_str:
        for ep in explored_el_price_str:
            # Accessing the 3-layer structure
            entry = results_summary[tec_str][f"std_{ct}"][f"el_price_{ep}"]

            # Determine cost (handling both Series and scalar)
            cost_val = entry["cost_of_avoided"]
            cost_scalar = cost_val.iloc[0] if hasattr(cost_val, "iloc") else cost_val

            # Determine type (handling both Series and scalar)
            type_val = entry["type_installed"]
            type_scalar = type_val.iloc[0] if hasattr(type_val, "iloc") else type_val

            data.append({
                "std": float(ct),
                "el_price": float(ep),
                "cost_of_avoided": cost_scalar,
                "type_installed": type_scalar
            })

    df = pd.DataFrame(data)

    # Pivot both metrics into matrices
    cost_matrix[tec_str] = df.pivot(index="el_price", columns="std", values="cost_of_avoided")
    type_matrix[tec_str] = df.pivot(index="el_price", columns="std", values="type_installed")

    # Sort indices: Electricity price (y) usually high to low, Std (x) low to high
    cost_matrix[tec_str] = cost_matrix[tec_str].sort_index(ascending=False)
    type_matrix[tec_str] = type_matrix[tec_str].sort_index(ascending=False)
    cost_matrix[tec_str] = cost_matrix[tec_str].sort_index(axis=1, ascending=True)
    type_matrix[tec_str] = type_matrix[tec_str].sort_index(axis=1, ascending=True)

# ------- PLOTTING ----------

for tec_str in explored_tec_str:
    setup_matplotlib_for_paper("single")
    fig, ax = plt.subplots()
    tm = type_matrix[tec_str]
    cm = cost_matrix[tec_str]

    # Iterate through the matrices to draw the cells
    for i, ep in enumerate(tm.index):
        for j, std in enumerate(tm.columns):
            tech = tm.loc[ep, std]
            cost = cm.loc[ep, std]

            # Draw the colored background based on Technology Type
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1,
                    facecolor=type_to_color.get(tech, "#CCCCCC"),  # fallback to grey
                    edgecolor="white",
                    linewidth=0.8
                )
            )

            # Add the Cost as text
            ax.text(
                j + 0.5, i + 0.5,
                f"{cost:.1f}",
                ha="center", va="center",
                color="white",
                fontsize=rcParams["font.size"] - 2,
                fontweight="bold"
            )

    # --- AXES FORMATTING ---
    ax.set_xlim(0, len(tm.columns))
    ax.set_ylim(0, len(tm.index))

    ax.set_xticks(np.arange(len(tm.columns)) + 0.5)
    ax.set_yticks(np.arange(len(tm.index)) + 0.5)

    ax.set_xticklabels(tm.columns)
    ax.set_yticklabels(tm.index)

    # Invert Y to show high prices at the top if necessary (already handled by sort_index)
    ax.invert_yaxis()

    ax.set_xlabel("Std increase [-]")
    ax.set_ylabel("Electricity price [€/MWh]")
    ax.set_title(f"Scenario: {tec_str}", pad=30)

    # --- LEGEND ---
    legend_patches = [
        mpatches.Patch(color=type_to_color[t], label=t) for t in types if t in tm.values
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_patches),
        frameon=True,
        fontsize='small'
    )

    fig.tight_layout(pad=0.6)
    save_figure_for_paper(fig, f"flex_tech_selection_{tec_str}", figures_path)


#-------------Plot time series ---------------------------

time_axis = np.arange(8760)
explored_tec_str = ["mea", "oxy"]

for tec_str in explored_tec_str:
    n_rows, n_cols = len(explored_el_price_str), len(explored_std_el_str)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3),
                             sharex=True, sharey='row')
    axes = np.atleast_2d(axes)

    for i, ep in enumerate(reversed(explored_el_price_str)):
        for j, ct in enumerate(explored_std_el_str):
            ax = axes[i, j]
            entry = results_summary[tec_str][f"std_{ct}"][f"el_price_{ep}"]

            ax.fill_between(time_axis, entry['hourly_clinker_demand'],
                            color='gray', alpha=0.2, label='Demand', zorder=1)

            ax.plot(time_axis, entry['hourly_clinker_output'],
                    color='#222A6A', linewidth=0.7, label='Production', zorder=2)

            # Delta cost annotation — subtle, bottom-left
            delta_cost = float(entry['delta_cost_abatement'])
            ax.annotate(
                r"$\Delta CCA = $" + f"${delta_cost:.0f}$" + r"€/tCO$_2$",
                xy=(0.03, 0.06),
                xycoords='axes fraction',
                ha='left', va='bottom',
                fontsize=7.5,
                color='black',
                alpha=0.75,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.5)
            )

            ax.set_ylim(0, None)
            ax.spines[['top', 'right']].set_visible(False)

            if i == 0: ax.set_title(f"Std: {ct}")
            if j == 0: ax.set_ylabel(f"El. price: {ep} €\nClinker [t/h]")
            if i == n_rows - 1: ax.set_xlabel("Hour [h]")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False)

    save_figure_for_paper(fig, f"clinker_only_{tec_str}", figures_path)

plt.show()