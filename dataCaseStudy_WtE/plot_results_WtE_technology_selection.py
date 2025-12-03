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



colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']


## -----------------  Carbon and electricity price --------------------------
explored_carbon_tax = [50, 75, 100,125, 150]
explored_el_price = [25, 50, 75, 100,125] # average el prices explored in the analysis
gas_price = 40
import_price_RDF = 20

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
explored_el_price_str = [str(r) for r in explored_el_price]
results_summary = {}

for j in range(0,num_carbon_tax):
    carbon_tax = explored_carbon_tax[j]
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
        el_price = explored_el_price[i] * electricity_price_norm
        
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

        boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG_existing')]
        heat_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster', 'heat', 'demand')]
        boiler_output = df_operation.loc[:,
                        ('technology_operation', 'period1', 'industrial_cluster', 'Boiler_Industrial_NG_existing')]

        w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCHP')]
        w2e_CaL_design = df_design.loc[:, ('industrial_cluster', 'WasteCaL_CCS')]

        if w2e_design["size"].iloc[0]>0 and w2e_design["size_ccs"].iloc[0]>0:
            json_wasteCHP = Path("./technologies_json/WasteCHP.json")
            info_wasteCHP = json.loads(json_wasteCHP.read_text())
            lhv = info_wasteCHP["Performance"]["LHV"]
            th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
            el_efficiency = info_wasteCHP["Performance"]["el_efficiency"]
            emission_factor = info_wasteCHP["Performance"]["emission_factor"]
            w2e_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCHP')]
            json_boiler = Path("./technologies_json/Boiler_Industrial_NG.json")
            info_boiler = json.loads(json_boiler.read_text())
            th_efficiency_boiler = info_boiler["Performance"]["performance"]["out"]["heat"][1]
            emission_factor_boiler = info_boiler["Performance"]["emission_factor"]
            waste_in = w2e_operation['wasteIn_input']
            el_out = w2e_operation['electricity_output']

            # Compute and retrieve necessary parameters
            # El. production if CCS didn't exist
            baseline_el_prod = ((waste_in * lhv - heat_demand / th_efficiency) * el_efficiency).where(
                (waste_in * lhv - heat_demand / th_efficiency) > 0, 0)
            baseline_boiler_prod = (heat_demand - waste_in * lhv * th_efficiency).where(
                (heat_demand - waste_in * lhv * th_efficiency) > 0, 0)

            loss_el_revenues = sum((baseline_el_prod-el_out)*el_price)
            extra_cost_boiler = sum(
                boiler_output['heat_output'] - baseline_boiler_prod) / th_efficiency_boiler * (
                                                                             emission_factor_boiler * carbon_tax + gas_price)
            
            # Relevant KPIs for economics
            type_installed = "MEA"
            capex = w2e_design["capex_ccs"]
            opex_fixed = w2e_design["opex_fixed_ccs"]
            opex_variable = w2e_design["opex_variable_ccs"]
            energy_cost = loss_el_revenues + extra_cost_boiler
            co2_captured = w2e_operation['CO2captured_var_output_ccs']

        
        elif w2e_CaL_design["size"].iloc[0] > 0:
            w2e_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCaL_CCS')]
            revenue_el_cal = sum(w2e_operation['el_cal'] * el_price)
            co2_captured = w2e_operation['CO2captured_var_output_ccs']

            
            type_installed = "CaL"
            capex = w2e_CaL_design["capex_tot"]
            opex_fixed = w2e_CaL_design["opex_fixed"]
            opex_variable = w2e_CaL_design["opex_variable"]
            co2_captured = w2e_operation['CO2captured_output']
            energy_cost = revenue_el_cal + sum(w2e_operation["wasteInRDF_input"]*import_price_RDF)

        else:
            type_installed = "none"
            opex_fixed = 0
            opex_variable = 0
            co2_captured = 0
            energy_cost = 0

        results_summary[carbon_tax_str][el_price_str]['hourly_co2_captured'] = co2_captured
        results_summary[carbon_tax_str][el_price_str]['capex_tot'] = capex
        results_summary[carbon_tax_str][el_price_str]['opex_fixed'] = opex_fixed
        results_summary[carbon_tax_str][el_price_str]['opex_variable'] = opex_variable
        results_summary[carbon_tax_str][el_price_str]['energy_cost'] = energy_cost
        results_summary[carbon_tax_str][el_price_str]['tot_co2_captured'] = (sum(co2_captured) if not isinstance(co2_captured, int) else pd.Series([0]))
        results_summary[carbon_tax_str][el_price_str]['cost_of_capture'] = ((capex + opex_fixed + opex_variable + energy_cost)/sum(co2_captured) if not isinstance(co2_captured, int) else pd.Series([0]))
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
df["carbon_tax"] = pd.to_numeric(df["carbon_tax"])
df["el_price"]   = pd.to_numeric(df["el_price"])

# Pivot to matrices
cost_matrix = df.pivot(index="el_price", columns="carbon_tax", values="cost_of_capture")
type_matrix = df.pivot(index="el_price", columns="carbon_tax", values="type_installed")
cost_matrix = cost_matrix.sort_index(ascending=False).sort_index(axis=1)
type_matrix = type_matrix.sort_index(ascending=False).sort_index(axis=1)



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
plt.xlabel("Carbon tax [€/tCO$_2$]")
plt.ylabel("Electricity price [€/MWh]")
plt.colorbar(im, label="LCOC [€/tCO$_2$]")
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


ax.set_xlim(0, len(type_matrix.columns))
ax.set_ylim(0, len(type_matrix.index))
ax.set_xticks([x+0.5 for x in range(len(type_matrix.columns))])
ax.set_yticks([y+0.5 for y in range(len(type_matrix.index))])
ax.set_xticklabels(type_matrix.columns)
ax.set_yticklabels(type_matrix.index)
ax.invert_yaxis()
plt.xlabel("Carbon tax [€/tCO$_2$]")
plt.ylabel("Electricity price [€/MWh]")

# Legend outside at the top, horizontal
patches = [mpatches.Patch(color=type_to_color[t], label=t) for t in types]
ax.legend(handles=patches, loc="lower center",
          bbox_to_anchor=(0.5, 1.05), ncol=len(types))

plt.tight_layout()
plt.show()



# --- Plot 2: Grid of installed type (categorical) ---
types = ["none", "MEA", "Partial oxyfuel", "Oxyfuel + MEA"]
type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}

plt.figure(figsize=(7,5))
ax = plt.gca()
for i, ep in enumerate(type_matrix.index):
    for j, ct in enumerate(type_matrix.columns):
        t = type_matrix.loc[ep, ct]
        c = cost_matrix.loc[ep, ct]  # cost at same position
        ax.add_patch(
            plt.Rectangle((j, i), 1, 1, color=type_to_color[t])
        )
        # add cost value as label
        ax.text(j+0.5, i+0.5, f"{c:.1f}",
                ha="center", va="center", color="white", fontsize=12, fontweight="bold")

ax.set_xlim(0, len(type_matrix.columns))
ax.set_ylim(0, len(type_matrix.index))
ax.set_xticks([x+0.5 for x in range(len(type_matrix.columns))])
ax.set_yticks([y+0.5 for y in range(len(type_matrix.index))])
ax.set_xticklabels(type_matrix.columns)
ax.set_yticklabels(type_matrix.index)
ax.invert_yaxis()
plt.xlabel("Carbon tax [€/tCO$_2$]")
plt.ylabel("Electricity price [€/MWh]")

# Legend outside at the top, horizontal
patches = [mpatches.Patch(color=type_to_color[t], label=t) for t in types]
ax.legend(handles=patches, loc="lower center",
          bbox_to_anchor=(0.5, 1.05), ncol=len(types))

plt.tight_layout()
plt.show()
