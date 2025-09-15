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
from matplotlib import rcParams
import warnings

from utilities.process_results import save_figure_for_paper, print_h5_structure


colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']


## -----------------  Carbon and electricity price --------------------------
explored_carbon_tax = [50, 75, 100,125, 150]
explored_el_price = [25, 50, 75, 100,125] # average el prices explored in the analysis
cost_extra_fuel = 15

path_processed_data = Path("./dataSources/data_processed.xlsx")
data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
av_el_price = data["el_price_itNord"].mean()
electricity_price_norm = data["el_price_itNord"]/av_el_price
json_cement = Path("./technologies_json/CementEmitter.json")
info_cement = json.loads(json_cement.read_text())
emission_factor = info_cement["Performance"]["emission_factor"]
json_heat_pump = Path("./technologies_json/HeatPump.json")
info_heat_pump = json.loads(json_heat_pump.read_text())
cop_hp = info_heat_pump["Performance"]["performance"]["out"]["heat"][1]


num_el_prices = len(explored_el_price)
num_carbon_tax = len(explored_carbon_tax)
raw_results_path = Path("./raw_results")
explored_carbon_tax_str = [str(r) for r in explored_carbon_tax]

explored_carbon_tax_str = [str(r) for r in explored_carbon_tax]
explored_el_price_str = [str(r) for r in explored_el_price]
results_summary = {}

for j in range(0,num_carbon_tax):

    carbon_tax_str = f"carbon_tax_{explored_carbon_tax_str[j]}"
    results_summary[carbon_tax_str] = {}

    # Get all directories that contain 'carbon_tax_str' in the name
    carbon_tax_dirs = [d for d in raw_results_path.iterdir()
                       if d.is_dir() and carbon_tax_str in d.name]

    # Sort directories by name
    dir_results_sorted = sorted(carbon_tax_dirs)

    # Get the most recent ones
    file_names = [d.name for d in dir_results_sorted[-num_el_prices:]]
    for i in range(0,len(file_names)):
        file_path = raw_results_path / f"{file_names[i]}/optimization_results.h5"

        # Check if each explored_el_price[i] is in file_names[i]
        el_price_str = f"el_price_{explored_el_price[i]}"
        results_summary[carbon_tax_str][el_price_str] = {}
        if f"el_price_{el_price_str}" in file_names[i]:
            print(f"{el_price_str} found in {file_names[i]}")
        else:
            print(f"{el_price_str} NOT found in {file_names[i]}")

        with h5py.File(file_path, 'r') as hdf_file:
            df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
            df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
        #print(df_operation)

        cement_mea_design = df_design.loc[:, ('industrial_cluster', 'CementEmitter')]
        cement_mea_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
        heat_pump_design = df_design.loc[:, ('industrial_cluster', 'HeatPump')]
        heat_pump_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'HeatPump')]
        cement_oxy_design = df_design.loc[:, ('industrial_cluster', 'CementHybridCCS')]
        cement_oxy_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementHybridCCS')]


        clinker_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','clinker', 'demand')]
        emissions_cement = clinker_demand * emission_factor


        # economics
        el_price = electricity_price_norm*explored_el_price[i]
        if cement_mea_design["size_ccs"].iloc[0] > 0:
            type_installed = "MEA"
            capex = cement_mea_design["capex_tot"] + heat_pump_design["capex_tot"]
            opex_fixed = cement_mea_design["opex_fixed"]
            opex_variable = cement_mea_design["opex_variable"]
            energy_cost = sum(cement_mea_operation["electricity_var_input_ccs"]*el_price) + sum(cement_mea_operation["heat_var_input_ccs"]/cop_hp*el_price)
            co2_captured = cement_mea_operation['CO2captured_var_output_ccs']

        elif cement_oxy_design["size"].iloc[0] > 0:
            if cement_oxy_design["size_mea"].iloc[0] > 0:
                type_installed = "Oxyfuel + MEA"
            else:
                type_installed = "Partial oxyfuel"

            capex = cement_oxy_design["capex_tot"]
            opex_fixed = cement_oxy_design["opex_fixed"]
            opex_variable = cement_oxy_design["opex_variable"]
            co2_captured = cement_oxy_operation['CO2captured_output']
            energy_cost = sum(cement_oxy_operation["electricity_input"]*el_price) + sum(cement_oxy_operation["extra_fuel_input"]*cost_extra_fuel)

        else:
            type_installed = "none"



        results_summary[carbon_tax_str][el_price_str]['hourly_co2_captured'] = co2_captured
        results_summary[carbon_tax_str][el_price_str]['capex_tot'] = capex
        results_summary[carbon_tax_str][el_price_str]['opex_fixed'] = opex_fixed
        results_summary[carbon_tax_str][el_price_str]['opex_variable'] = opex_variable
        results_summary[carbon_tax_str][el_price_str]['energy_cost'] = energy_cost
        results_summary[carbon_tax_str][el_price_str]['tot_co2_captured'] = sum(co2_captured)
        results_summary[carbon_tax_str][el_price_str]['cost_of_capture'] = (capex + opex_fixed + opex_variable + energy_cost)/sum(co2_captured)
        results_summary[carbon_tax_str][el_price_str]['type_installed'] = type_installed



# Flatten nested dict into a list of rows
data = []
for ct in explored_carbon_tax_str:
    for ep in explored_el_price_str:
        entry = results_summary[f"carbon_tax_{ct}"][f"el_price_{ep}"]
        data.append({
            "carbon_tax": ct,
            "el_price": ep,
            "cost_of_capture": entry["cost_of_capture"].iloc[0],
            "type_installed": entry["type_installed"]
        })

df = pd.DataFrame(data)

# Pivot to matrices
cost_matrix = df.pivot(index="el_price", columns="carbon_tax", values="cost_of_capture")
type_matrix = df.pivot(index="el_price", columns="carbon_tax", values="type_installed")




# --- Define batlow colors ---
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E',
                 '#F7D03C', '#D491B8', '#012E4D']

# --- Plot 1: Heatmap of costs (continuous) ---
plt.figure(figsize=(7,5))
im = plt.imshow(cost_matrix.values.astype(float),
                cmap=plt.cm.colors.ListedColormap(batlow_colors),
                aspect="auto")

# add text annotations
for i in range(cost_matrix.shape[0]):
    for j in range(cost_matrix.shape[1]):
        plt.text(j, i, f"{cost_matrix.iloc[i, j]:.1f}",
                 ha="center", va="center", color="white", fontsize=8)

plt.xticks(range(len(cost_matrix.columns)), cost_matrix.columns)
plt.yticks(range(len(cost_matrix.index)), cost_matrix.index)
plt.xlabel("Carbon tax [€/tCO_2]")
plt.ylabel("Electricity price [€/MWh]")
plt.colorbar(im, label="LCOC [€/tCO_2]")
plt.show()

# --- Plot 2: Grid of installed type (categorical) ---
types = ["none", "MEA", "Partial oxyfuel", "Oxyfuel + MEA"]
type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}

plt.figure(figsize=(7,5))
ax = plt.gca()
for i, ep in enumerate(type_matrix.index):
    for j, ct in enumerate(type_matrix.columns):
        t = type_matrix.loc[ep, ct]
        ax.add_patch(
            plt.Rectangle((j, i), 1, 1, color=type_to_color[t])
        )
        # # add text label
        # ax.text(j+0.5, i+0.5, t,
        #         ha="center", va="center", color="white", fontsize=8)

ax.set_xlim(0, len(type_matrix.columns))
ax.set_ylim(0, len(type_matrix.index))
ax.set_xticks([x+0.5 for x in range(len(type_matrix.columns))])
ax.set_yticks([y+0.5 for y in range(len(type_matrix.index))])
ax.set_xticklabels(type_matrix.columns)
ax.set_yticklabels(type_matrix.index)
ax.invert_yaxis()
plt.xlabel("Carbon Tax")
plt.ylabel("Electricity Price")
plt.title("Installed Type")

# Legend
patches = [mpatches.Patch(color=type_to_color[t], label=t) for t in types]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()
