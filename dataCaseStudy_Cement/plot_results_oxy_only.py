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

from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap


# Set global styling for the plots
colors = []
pink_cmap = LinearSegmentedColormap.from_list(
    "white_pink", ["#FAF0F6","#D491B8"]
)
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"


## -----------------  Carbon and electricity price --------------------------
explored_carbon_tax = [150, 151, 250]
explored_el_price = [107] # average el prices explored in the analysis
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
num_carbon_tax = len(explored_carbon_tax)
raw_results_path = Path("./raw_results/oxy_only")
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
            df_design_network = pd.DataFrame(extract_datasets_from_h5group(
                hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]))
        #print(df_operation)



        co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]

        clinker_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','clinker', 'demand')]
        emissions_cement_baseline = clinker_demand * emission_factor_clinker_baseline

        # economics
        pipeline_cost = df_design_network['capex'].values.flatten()[0]
        storage_cost = co2_storage_design['opex_variable']

        el_price = electricity_price_norm*explored_el_price[i]


        if ('industrial_cluster', 'CementEmitter') in df_design.columns:
            cement_mea_design = df_design.loc[:, ('industrial_cluster', 'CementEmitter')]
            cement_mea_operation = df_operation.loc[:,
                                   ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
            heat_pump_design = df_design.loc[:, ('industrial_cluster', 'HeatPump')]
            heat_pump_operation = df_operation.loc[:,
                                  ('technology_operation', 'period1', 'industrial_cluster', 'HeatPump')]
            type_installed = "MEA"
            capex = cement_mea_design["capex_tot"] + heat_pump_design["capex_tot"]
            opex_fixed = cement_mea_design["opex_fixed"] + heat_pump_design["opex_fixed"]
            opex_variable = cement_mea_design["opex_variable"]
            energy_cost = sum(cement_mea_operation["electricity_var_input_ccs"]*el_price) + sum(cement_mea_operation["heat_var_input_ccs"]/cop_hp*el_price)
            co2_captured = cement_mea_operation['CO2captured_var_output_ccs']
            tot_co2_avoided = sum(cement_mea_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
                cement_mea_operation["emissions_pos"])
            transport_stor_cost = storage_cost + pipeline_cost
            ccs_size = cement_mea_design["size_ccs"]
            net_emissions = sum(cement_mea_operation["emissions_pos"])
            load_factor = sum(co2_captured)/(ccs_size*8760)
            fraction_avoided = tot_co2_avoided/sum(emissions_cement_baseline)

        elif ('industrial_cluster', 'CementHybridCCS') in df_design.columns:

            cement_oxy_design = df_design.loc[:, ('industrial_cluster', 'CementHybridCCS')]
            cement_oxy_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementHybridCCS')]
            if cement_oxy_design["size_mea"].iloc[0] > 0:
                type_installed = "Oxyfuel + PCC"
            else:
                type_installed = "Oxyfuel"

            capex = cement_oxy_design["capex_tot"]
            opex_fixed = cement_oxy_design["opex_fixed"]
            opex_variable = cement_oxy_design["opex_variable"]
            co2_captured = cement_oxy_operation['CO2captured_output']
            energy_cost = sum(cement_oxy_operation["electricity_input"]*el_price) + sum(cement_oxy_operation["extra_fuel_input"]*cost_extra_fuel)
            tot_co2_avoided = sum(cement_oxy_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
            cement_oxy_operation["emissions_pos"])
            transport_stor_cost = storage_cost + pipeline_cost
            ccs_size = max(co2_captured)
            net_emissions = sum(cement_oxy_operation["emissions_pos"])
            load_factor = sum(co2_captured)/(ccs_size*8760)
            fraction_avoided = tot_co2_avoided/sum(emissions_cement_baseline)



        results_summary[carbon_tax_str][el_price_str]['hourly_co2_captured'] = co2_captured
        results_summary[carbon_tax_str][el_price_str]['capex_tot'] = capex
        results_summary[carbon_tax_str][el_price_str]['opex_fixed'] = opex_fixed
        results_summary[carbon_tax_str][el_price_str]['opex_variable'] = opex_variable
        results_summary[carbon_tax_str][el_price_str]['energy_cost'] = energy_cost
        results_summary[carbon_tax_str][el_price_str]['transport_stor_cost'] = transport_stor_cost
        results_summary[carbon_tax_str][el_price_str]['tot_co2_captured'] = (sum(co2_captured) if not isinstance(co2_captured, int) else pd.Series([0]))
        results_summary[carbon_tax_str][el_price_str]['cost_of_avoided'] = ((capex + opex_fixed + opex_variable + energy_cost
                                                                             +transport_stor_cost)/tot_co2_avoided if not isinstance(co2_captured, int) else pd.Series([0]))
        results_summary[carbon_tax_str][el_price_str]['type_installed'] = type_installed
        results_summary[carbon_tax_str][el_price_str]['net_emissions'] = net_emissions
        results_summary[carbon_tax_str][el_price_str]['size_ccs'] = ccs_size
        results_summary[carbon_tax_str][el_price_str]['load_factor_ccs'] = load_factor
        results_summary[carbon_tax_str][el_price_str]['fraction_avoided'] = fraction_avoided
        results_summary[carbon_tax_str][el_price_str]['tot_co2_avoided'] = tot_co2_avoided



# Flatten nested dict into a list of rows
data = []
for ct in explored_carbon_tax_str:
    for ep in explored_el_price_str:
        entry = results_summary[f"carbon_tax_{ct}"][f"el_price_{ep}"]
        data.append({
            "carbon_tax": ct,
            "el_price": ep,
            "cost_of_avoided": entry["cost_of_avoided"].iloc[0],
            "type_installed": entry["type_installed"]
        })

df = pd.DataFrame(data)
df["carbon_tax"] = pd.to_numeric(df["carbon_tax"])
df["el_price"]   = pd.to_numeric(df["el_price"])

# Pivot to matrices
cost_matrix = df.pivot(index="el_price", columns="carbon_tax", values="cost_of_avoided")
type_matrix = df.pivot(index="el_price", columns="carbon_tax", values="type_installed")
cost_matrix = cost_matrix.sort_index(ascending=False).sort_index(axis=1)
type_matrix = type_matrix.sort_index(ascending=False).sort_index(axis=1)



# --- Define batlow colors ---
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E',
                 '#F7D03C', '#D491B8', '#012E4D']




# ------------------------------------------------------------
# PAPER SETUP
# ------------------------------------------------------------
setup_matplotlib_for_paper("single")

types = ["MEA","Oxyfuel", "Oxyfuel + PCC"]
type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}

# ------------------------------------------------------------
# CEMENT CCS COST BREAKDOWN BAR CHART
# ------------------------------------------------------------

def to_scalar(val):
    """Return a plain float from a pd.Series, pd.DataFrame, or scalar."""
    if hasattr(val, "iloc"):
        return float(val.iloc[0])
    return float(val)

setup_matplotlib_for_paper("single")

item_colors_cement = {
    "CAPEX":               batlow_colors[0],
    "OPEX fixed":          batlow_colors[1],
    "Energy cost":         batlow_colors[4],
    "Transport & Storage": batlow_colors[3],
}

# --- Collect data ---
bar_labels       = []
bar_capex        = []
bar_opex_f       = []
bar_opex_v       = []
bar_energy       = []
bar_transp       = []
bar_frac_avoided = []

for ct in explored_carbon_tax_str:
    for ep in explored_el_price_str:
        entry = results_summary[f"carbon_tax_{ct}"][f"el_price_{ep}"]
        tot_co2_avoided = to_scalar(entry["tot_co2_avoided"])

        if tot_co2_avoided == 0:
            continue

        bar_capex.append(to_scalar(entry["capex_tot"])            / tot_co2_avoided)
        bar_opex_f.append(to_scalar(entry["opex_fixed"])          / tot_co2_avoided)
        bar_energy.append(entry["energy_cost"]                    / tot_co2_avoided)
        bar_transp.append(to_scalar(entry["transport_stor_cost"]) / tot_co2_avoided)
        bar_frac_avoided.append(to_scalar(entry["fraction_avoided"]) * 100)
        bar_labels.append(f"{entry['type_installed']}")

capex_arr        = np.array(bar_capex,        dtype=float)
opex_f_arr       = np.array(bar_opex_f,       dtype=float)
energy_arr       = np.array(bar_energy,       dtype=float)
transp_arr       = np.array(bar_transp,       dtype=float)
frac_avoided_arr = np.array(bar_frac_avoided, dtype=float)

x_cement     = np.arange(len(bar_labels))
width_cement = 0.5
edge_width   = 0.7  # Defined once for consistency

fig, ax = plt.subplots()
ax2 = ax.twinx()

# --- Stacked bars with Black Edges ---
bottom = np.zeros(len(x_cement))

ax.bar(x_cement, capex_arr, width_cement, bottom=bottom,
       color=item_colors_cement["CAPEX"], edgecolor='black', linewidth=edge_width)
bottom += capex_arr

ax.bar(x_cement, opex_f_arr, width_cement, bottom=bottom,
       color=item_colors_cement["OPEX fixed"], edgecolor='black', linewidth=edge_width)
bottom += opex_f_arr


ax.bar(x_cement, energy_arr, width_cement, bottom=bottom,
       color=item_colors_cement["Energy cost"], edgecolor='black', linewidth=edge_width)
bottom += energy_arr

ax.bar(x_cement, transp_arr, width_cement, bottom=bottom,
       color=item_colors_cement["Transport & Storage"], edgecolor='black', linewidth=edge_width)

# --- Horizontal grid lines ---
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='grey', linewidth=0.4, linestyle='--', alpha=0.5)
ax.set_xticks(x_cement)
ax.set_xticklabels(bar_labels, ha='center')

# --- Fraction avoided on secondary axis ---
ax2.scatter(x_cement, frac_avoided_arr,
            color='black', marker='D', s=9, zorder=5)
# Add text labels for each marker
for i, val in enumerate(frac_avoided_arr):
    ax2.text(
        x_cement[i],
        val - 15,
        f"{val:.1f}%",
        ha='center',
        va='bottom',
        fontsize=7,
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='white',
            edgecolor='none',
            alpha=0.6,
        )
    )
# --- Style Adjustments ---
ax.set_ylabel("CCA [€/tCO$_2$]")
ax2.set_ylabel("CO$_2$ avoidance [%]")
ax.set_ylim(0, 200)
ax2.set_ylim(0, 140) # Reset to 100 as it's a percentage

# --- Legend Reconstruction (Matching the Black Edges) ---
# Use facecolor and edgecolor to match the bars exactly
cost_handles = [mpatches.Patch(facecolor=c, edgecolor='black', linewidth=edge_width, label=l)
                for l, c in item_colors_cement.items()]

frac_handle  = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                           markersize=4, label='CO$_2$ avoid. [%]')

# Place Legend
ax.legend(
    handles=cost_handles + [frac_handle],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.01), # Adjusted slightly for extra breathing room
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.4,
    frameon=False,
)
save_figure_for_paper(fig, "cement_oxy_only_cost_breakdown", figures_path)
plt.show()