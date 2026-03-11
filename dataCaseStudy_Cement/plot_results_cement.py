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


colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"


## -----------------  Carbon and electricity price --------------------------
explored_carbon_tax = [100, 150, 200, 250, 300]
explored_el_price = [50, 100, 150, 200, 250] # average el prices explored in the analysis
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
raw_results_path = Path("./raw_results/tech_selection")
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
            df_design_network = pd.DataFrame(extract_datasets_from_h5group(
                hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]))
        #print(df_operation)

        cement_mea_design = df_design.loc[:, ('industrial_cluster', 'CementEmitter')]
        cement_mea_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
        heat_pump_design = df_design.loc[:, ('industrial_cluster', 'HeatPump')]
        heat_pump_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'HeatPump')]
        cement_oxy_design = df_design.loc[:, ('industrial_cluster', 'CementHybridCCS')]
        cement_oxy_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementHybridCCS')]
        co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]

        clinker_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','clinker', 'demand')]
        emissions_cement_baseline = clinker_demand * emission_factor_clinker_baseline

        # economics
        pipeline_cost = df_design_network['capex'].values.flatten()[0]
        storage_cost = co2_storage_design['opex_variable']

        el_price = electricity_price_norm*explored_el_price[i]
        if cement_mea_design["size_ccs"].iloc[0] > 0:
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
            tot_co2_avoided = sum(cement_oxy_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
            cement_oxy_operation["emissions_pos"])
            transport_stor_cost = storage_cost + pipeline_cost
            ccs_size = max(co2_captured)
            net_emissions = sum(cement_oxy_operation["emissions_pos"])
            load_factor = sum(co2_captured)/(ccs_size*8760)
            fraction_avoided = tot_co2_avoided/sum(emissions_cement_baseline)


        else:
            type_installed = "none"
            opex_fixed = 0
            opex_variable = 0
            co2_captured = 0
            energy_cost = 0
            tot_co2_avoided = 0
            capex = 0
            transport_stor_cost = 0
            ccs_size = 0
            net_emissions = sum(emissions_cement_baseline)
            load_factor = 0
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

# # --- Plot 1: Heatmap of costs (continuous) ---
# plt.figure(figsize=(7,5))
# im = plt.imshow(cost_matrix.values.astype(float),
#                 cmap=plt.cm.colors.ListedColormap(batlow_colors),
#                 aspect="auto")
#
# # add text annotations
# for i in range(cost_matrix.shape[0]):
#     for j in range(cost_matrix.shape[1]):
#         plt.text(j, i, f"{cost_matrix.iloc[i, j]:.1f}",
#                  ha="center", va="center", color="white", fontsize=8)
#
# plt.xticks(range(len(cost_matrix.columns)), cost_matrix.columns)
# plt.yticks(range(len(cost_matrix.index)), cost_matrix.index)
# plt.xlabel("Carbon tax [€/tCO$_2$]")
# plt.ylabel("Electricity price [€/MWh]")
# plt.colorbar(im, label="LCOC [€/tCO$_2$]")
# plt.show()

# --- Plot 2: Grid of installed type (categorical) ---
# types = ["none", "MEA", "Partial oxyfuel", "Oxyfuel + PCC"]
# type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}
#
# plt.figure(figsize=(7,5))
# ax = plt.gca()
# for i, ep in enumerate(type_matrix.index):
#     for j, ct in enumerate(type_matrix.columns):
#         t = type_matrix.loc[ep, ct]
#         ax.add_patch(
#             plt.Rectangle((j, i), 1, 1, color=type_to_color[t])
#         )
#
#
# ax.set_xlim(0, len(type_matrix.columns))
# ax.set_ylim(0, len(type_matrix.index))
# ax.set_xticks([x+0.5 for x in range(len(type_matrix.columns))])
# ax.set_yticks([y+0.5 for y in range(len(type_matrix.index))])
# ax.set_xticklabels(type_matrix.columns)
# ax.set_yticklabels(type_matrix.index)
# ax.invert_yaxis()
# plt.xlabel("Carbon tax [€/tCO$_2$]")
# plt.ylabel("Electricity price [€/MWh]")
#
# # Legend outside at the top, horizontal
# patches = [mpatches.Patch(color=type_to_color[t], label=t) for t in types]
# ax.legend(handles=patches, loc="lower center",
#           bbox_to_anchor=(0.5, 1.05), ncol=len(types))
#
# plt.tight_layout()
# plt.show()



# ------------------------------------------------------------
# PAPER SETUP
# ------------------------------------------------------------
setup_matplotlib_for_paper("single")

types = ["none", "MEA", "Partial oxyfuel", "Oxyfuel + PCC"]
type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}

# ------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------
fig, ax = plt.subplots()

# ------------------------------------------------------------
# GRID PLOT
# ------------------------------------------------------------
for i, ep in enumerate(type_matrix.index):
    for j, ct in enumerate(type_matrix.columns):
        tech = type_matrix.loc[ep, ct]
        cost = cost_matrix.loc[ep, ct]

        # colored cell
        ax.add_patch(
            plt.Rectangle(
                (j, i), 1, 1,
                facecolor=type_to_color[tech],
                edgecolor="black",
                linewidth=0.8
            )
        )

        # cost annotation
        ax.text(
            j + 0.5, i + 0.5,
            f"{cost:.1f}",
            ha="center", va="center",
            color="white",
            fontsize=rcParams["font.size"] - 2,
            fontweight="bold"
        )

# ------------------------------------------------------------
# AXES FORMATTING
# ------------------------------------------------------------
ax.set_xlim(0, len(type_matrix.columns))
ax.set_ylim(0, len(type_matrix.index))

ax.set_xticks(np.arange(len(type_matrix.columns)) + 0.5)
ax.set_yticks(np.arange(len(type_matrix.index)) + 0.5)

ax.set_xticklabels(type_matrix.columns)
ax.set_yticklabels(type_matrix.index)

ax.invert_yaxis()

ax.set_xlabel("Carbon tax [€/tCO$_2$]")
ax.set_ylabel("Electricity price [€/MWh]")


legend_patches = [
    mpatches.Patch(color=type_to_color[t], label=t) for t in types
]

ax.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(types),
    frameon=False
)

fig.tight_layout(pad=0.6)
save_figure_for_paper(fig, "cement_tech_selection", figures_path)



# --- PLOT SECONDARY VARIABLES ---

# --- 1. DATA PREPARATION ---
results_data = {
    "net_em_ccs": {},
    "load_factor_ccs": {},
    "size_ccs": {},
    "fraction_avoided": {}
}

rows_ccs = []

for ct in explored_carbon_tax_str:
    ct_key = f"carbon_tax_{ct}"
    for ep in explored_el_price_str:
        ep_key = f"el_price_{ep}"

        d_ccs = results_summary[ct_key][ep_key]

        # Helper to safely handle pandas Series vs scalar floats
        def get_val(item):
            return float(item.iloc[0]) if hasattr(item, 'iloc') else float(item)

        em_val = get_val(d_ccs['net_emissions']) / 1000  # converted to kt
        lf_val = get_val(d_ccs['load_factor_ccs'])
        sz_val = get_val(d_ccs['size_ccs'])
        fa_val = get_val(d_ccs['fraction_avoided'])

        rows_ccs.append({
            "ct": int(ct), "ep": int(ep),
            "net_em": em_val, "lf": lf_val, "sz": sz_val, "fa": fa_val
        })


# Helper function to pivot lists into heatmaps
def pivot_data(rows, val_col):
    df_temp = pd.DataFrame(rows)
    return df_temp.pivot(index="ep", columns="ct", values=val_col).sort_index(ascending=False).astype(float)


# Store all matrices
results_data["net_em_ccs"]      = pivot_data(rows_ccs, "net_em")
results_data["load_factor_ccs"] = pivot_data(rows_ccs, "lf")
results_data["size_ccs"]        = pivot_data(rows_ccs, "sz")
results_data["fraction_avoided"]= pivot_data(rows_ccs, "fa")


# --- 2. STANDARDIZED PLOTTING FUNCTION ---
def plot_heatmap(df, label, filename, cmap, is_pct=False):
    setup_matplotlib_for_paper("single")
    fig, ax = plt.subplots(layout="constrained")

    data = df.to_numpy()
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            txt = f"{val:.1%}" if is_pct else f"{val:.1f}"

            # Contrast logic
            curr_vmin, curr_vmax = data.min(), data.max()
            rel_val = (val - curr_vmin) / (curr_vmax - curr_vmin) if (curr_vmax - curr_vmin) != 0 else 0
            color = "white" if rel_val > 0.6 else "black"

            ax.text(j, i, txt, ha="center", va="center", color=color,
                    fontsize=rcParams["axes.labelsize"] - 2, fontweight="bold")

    # Axes Setup
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel(r"Carbon tax [€/tCO$_2$]")
    ax.set_ylabel("Electricity price [€/MWh]")

    plt.colorbar(im, ax=ax, label=label)
    save_figure_for_paper(fig, filename, figures_path)


# --- 3. EXECUTION ---
metrics = [
    (results_data["net_em_ccs"],      r"Net emissions [ktCO$_2$/y]", "net_emissions",    "RdBu_r",  False),
    (results_data["load_factor_ccs"], "CCS load factor [-]",         "ccs_lf",           "YlGn",    True),
    (results_data["size_ccs"],        "CCS size [t/h]",              "ccs_size",         "Purples",  False),
    (results_data["fraction_avoided"],"Fraction avoided [-]",        "fraction_avoided", "Blues",    True),
]

for df, label, suffix, cmap, is_pct in metrics:
    plot_heatmap(df, label, f"cement_{suffix}", cmap, is_pct)

print("All CCS plots (Net Emissions, Load Factor, CCS Size, Fraction Avoided) have been saved.")
plt.show()
