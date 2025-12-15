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
import warnings

from utilities.process_results import save_figure_for_paper, print_h5_structure



colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
results_path = "../figures/WtE_MEA"

## -----------------  DH ratio --------------------------

explored_dh_ratio = [0, 0.25, 0.5, 0.75,1] # ratio of peak DH demand to supply compared to peak heat prod. from WtE
gas_price = 40
carbon_tax = 100


path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
el_price = data["el_price_itNord"]

json_wasteCHP = Path("./technologies_json/WasteCHP.json")
info_wasteCHP = json.loads(json_wasteCHP.read_text())
lhv = info_wasteCHP["Performance"]["LHV"]
th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
el_efficiency = info_wasteCHP["Performance"]["el_efficiency"]
emission_factor = info_wasteCHP["Performance"]["emission_factor"]
json_mea = Path("./technologies_json/MEA_medium.json")
info_mea = json.loads(json_mea.read_text())
ccr = info_mea["Performance"]["capture_rate"]


json_boiler = Path("./technologies_json/Boiler_Industrial_NG.json")
info_boiler = json.loads(json_boiler.read_text())
th_efficiency_boiler = info_boiler["Performance"]["performance"]["out"]["heat"][1]
emission_factor_boiler = info_boiler["Performance"]["emission_factor"]
num_cases = len(explored_dh_ratio)
raw_results_path = Path("./raw_results/MEA")
# Get all directories that contain 'dh_ratio' in the name
dh_ratio_dirs = [d for d in raw_results_path.iterdir()
                 if d.is_dir() and "dh_ratio" in d.name]

# Sort directories by name
dir_results_sorted = sorted(dh_ratio_dirs)

# Get the most recent ones
file_names = [d.name for d in dir_results_sorted[-num_cases:]]
explored_dh_ratio_str = [str(r) for r in explored_dh_ratio]

results_summary = {}
for i in range(0,len(file_names)):
    file_path = raw_results_path / f"{file_names[i]}/optimization_results.h5"
    # # print the third h5file
    # with h5py.File(file_path, "r") as hdf_file:
    #     print_h5_structure(hdf_file)
    # Check if each explored_dh_ratio[i] is in file_names[i]
    dh_ratio_str = explored_dh_ratio_str[i]
    results_summary[dh_ratio_str] = {}
    if f"{dh_ratio_str}" in file_names[i]:
        print(f"{dh_ratio_str} found in {file_names[i]}")
    else:
        print(f"{dh_ratio_str} NOT found in {file_names[i]}")

    with h5py.File(file_path, 'r') as hdf_file:


        df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
        df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
    print(df_operation)

    w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCHP')]
    boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG_existing')]
    w2e_output = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCHP')]
    boiler_output = df_operation.loc[:,
                    ('technology_operation', 'period1', 'industrial_cluster', 'Boiler_Industrial_NG_existing')]
    heat_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','heat', 'demand')]

    waste_processed_out = w2e_output['wasteProcessed_output']
    heat_out = w2e_output['heat_output']
    el_out = w2e_output['electricity_output']
    waste_out = w2e_output['wasteProcessed_output']
    waste_in = w2e_output['wasteIn_input']
    co2_captured_w2e = w2e_output['CO2captured_var_output_ccs']
    emissions_w2e = waste_in * emission_factor
    fraction_co2_captured = sum(co2_captured_w2e) / sum(emissions_w2e)
    size_ccs = w2e_design["size_ccs"]
    fraction_size_ccs = size_ccs/ (max(emissions_w2e)*ccr)
    ccs_capacity_factor = sum(co2_captured_w2e)/(size_ccs*8760)

    if explored_dh_ratio[i] == 0:
        boiler_load_factor = 0
    else:
        boiler_load_factor = sum(boiler_output['heat_output'])/ sum(heat_demand)

    results_summary[dh_ratio_str]['size_ccs'] = size_ccs
    results_summary[dh_ratio_str]['fraction_size_ccs'] = fraction_size_ccs
    results_summary[dh_ratio_str]['ccs_capacity_factor'] = ccs_capacity_factor
    results_summary[dh_ratio_str]['boiler_load_factor'] = boiler_load_factor
    results_summary[dh_ratio_str]['hourly_co2_captured'] = co2_captured_w2e
    results_summary[dh_ratio_str]['hourly_boiler_heat_out'] = boiler_output['heat_output']
    results_summary[dh_ratio_str]['hourly_heat_demand'] = heat_demand
    results_summary[dh_ratio_str]['hourly_el_prod_wte'] = el_out
    results_summary[dh_ratio_str]['hourly_heat_prod_wte'] = heat_out
    results_summary[dh_ratio_str]['hourly_el_for_ccs'] = w2e_output['electricity_var_input_ccs']
    results_summary[dh_ratio_str]['hourly_heat_for_ccs'] = w2e_output['heat_var_input_ccs']
    # warning: next line is ok as long as boiler_out is always smaller than demand
    residual_heat_demand  = [hd - bo for hd, bo in zip(heat_demand, boiler_output['heat_output'])]
    results_summary[dh_ratio_str]['hourly_wte_heat_to_demand'] = []
    results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'] = []
    # El. production if CCS didn't exist
    baseline_el_prod = ((waste_in * lhv - heat_demand / th_efficiency) * el_efficiency).where((waste_in * lhv - heat_demand / th_efficiency) > 0, 0)
    baseline_boiler_prod = (heat_demand - waste_in * lhv * th_efficiency).where((heat_demand - waste_in * lhv * th_efficiency) > 0, 0)
    if all(d >= 0 for d in residual_heat_demand):
        results_summary[dh_ratio_str]['hourly_wte_heat_to_demand'] = (heat_demand - boiler_output['heat_output'])/th_efficiency
        results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'] = w2e_output['heat_var_input_ccs'] / th_efficiency
    else:
        for t in range(0, len(residual_heat_demand)):
            if residual_heat_demand[t] >= 0:
                results_summary[dh_ratio_str]['hourly_wte_heat_to_demand'].append((heat_demand[t] - boiler_output[
                    'heat_output'][t]) / th_efficiency)
                results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'].append(w2e_output[
                                                                                    'heat_var_input_ccs'][t] / th_efficiency)
            else:
                results_summary[dh_ratio_str]['hourly_wte_heat_to_demand'].append(0)
                results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'].append((w2e_output[
                                                                                         'heat_var_input_ccs'][
                                                                                         t] - (boiler_output['heat_output'][t] - heat_demand[t])) / th_efficiency)


        warnings.warn(
            f"Boiler output exceeds heat demand at {dh_ratio_str} dh_ratio.",
            UserWarning
        )
    results_summary[dh_ratio_str]['hourly_wte_heat_for_el'] = el_out/el_efficiency
    results_summary[dh_ratio_str]['hourly_wte_heat_for_el_ccs'] =  w2e_output['electricity_var_input_ccs']/el_efficiency
    results_summary[dh_ratio_str]['capex_tot'] = w2e_design["capex_ccs"]
    results_summary[dh_ratio_str]['opex_fixed'] = w2e_design["opex_fixed_ccs"]
    results_summary[dh_ratio_str]['opex_variable'] = w2e_design["opex_variable"]
    results_summary[dh_ratio_str]['loss_el_revenues'] = sum((baseline_el_prod-el_out)*el_price)
    results_summary[dh_ratio_str]['extra_cost_boiler'] = sum(boiler_output['heat_output']-baseline_boiler_prod)/th_efficiency_boiler*(emission_factor_boiler*carbon_tax + gas_price)
    results_summary[dh_ratio_str]['tot_co2_captured'] = sum(co2_captured_w2e)
    results_summary['hourly_emissions'] = emissions_w2e
    results_summary['hourly_wasteProcessed'] = waste_processed_out


# # Plot
# fraction_size_ccs = [results_summary[dh]['fraction_size_ccs'] for dh in explored_dh_ratio_str]
# ccs_capacity_factor= [results_summary[dh]['ccs_capacity_factor'] for dh in explored_dh_ratio_str]
# boiler_load_factor= [results_summary[dh]['boiler_load_factor'] for dh in explored_dh_ratio_str]
#
# fig, ax = plt.subplots(figsize=figsize)
#
# ax.plot(explored_dh_ratio, fraction_size_ccs, color=batlow_colors[1],
#         linewidth=2, marker='o', label='Frac. CCS size')
# ax.plot(explored_dh_ratio, ccs_capacity_factor , color=batlow_colors[2],
#         linewidth=2, marker='o', label='CCS load factor')
# ax.plot(explored_dh_ratio, boiler_load_factor, color=batlow_colors[3],
#         linewidth=2, marker='o', label='Boiler load factor')
#
# ax.set_xlabel('DH peak demand compared to max WtE output [%]')
# ax.set_ylabel('[%]')  # Adjust if units are known
# ax.legend(loc='upper right')
#
# plt.tight_layout()
# plt.show()
#
#
# markers = ['*', 'p', 'h', 'P', 'X']
# fig, ax = plt.subplots(figsize=figsize)
# for i in range(0,len(explored_dh_ratio)):
#     dh_ratio_str = explored_dh_ratio_str[i]
#     co2_captured = results_summary[dh_ratio_str]['hourly_co2_captured']
#     ax.plot(range(0,len(co2_captured)), co2_captured, color=batlow_colors[i],
#             linewidth=2, marker=markers[i], label=dh_ratio_str)
#
# ax.set_xlabel('Time [h]')
# ax.set_ylabel('[%]')
# ax.legend(loc='upper right')
#
# plt.tight_layout()
# plt.show()
#
#
# # Plot single DH simulation
# n_plots = len(explored_dh_ratio_str)
#
# # Choose subplot grid size (e.g. 2 columns)
# ncols = 2
# nrows = (n_plots + ncols - 1) // ncols
#
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), sharex=False)
#
# # Flatten axes for easier indexing
# axes = axes.flatten()
# for i, dh_ratio_str in enumerate(explored_dh_ratio_str):
#     ax1 = axes[i]
#
#     co2_captured = results_summary[dh_ratio_str]['hourly_co2_captured']
#     heat_demand = results_summary[dh_ratio_str]['hourly_heat_demand']
#
#     # Plot heat demand on left axis
#     ln1 = ax1.plot(range(len(co2_captured)), co2_captured,
#                    color=batlow_colors[2], linewidth=1,
#                    label="CO2 captured")
#     ax1.set_xlabel("Time [h]")
#     ax1.set_ylabel("CO2 captured [t/h]")
#     ax1.grid(True)
#
#     # Second y-axis
#     ax2 = ax1.twinx()
#     ln2 = ax2.plot(range(len(heat_demand)), heat_demand,
#                    color=batlow_colors[1], linewidth=1,
#                    label="Heat demand")
#     ax2.set_ylabel("Heat demand [MW]")
#
#     # Combine legends
#     lines = ln1 + ln2
#     labels = [l.get_label() for l in lines]
#     ax1.legend(lines, labels, loc="lower right")
#
#     # Title with dh_ratio_str
#     ax1.set_title(dh_ratio_str)
#
# # Remove unused subplots (if any)
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout()
# plt.show()



#---------- Plot how the heat is used every hour---------------------

# Colors for the 4 stacked components
stack_colors = [batlow_colors[0], batlow_colors[1],
                batlow_colors[2], batlow_colors[3]]

n_plots = len(explored_dh_ratio_str)

# Subplot grid size (e.g. 2 columns)
ncols = 2
nrows = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(6*ncols, 4*nrows),
                         sharex=False, sharey=True)

axes = axes.flatten()

for i, dh_ratio_str in enumerate(explored_dh_ratio_str):
    ax = axes[i]
    print(f"{dh_ratio_str} -> fraction_size_ccs = {results_summary[dh_ratio_str]['fraction_size_ccs']}")
    print(f"{dh_ratio_str} -> capacity_factor_ccs = {results_summary[dh_ratio_str]['ccs_capacity_factor']}")
    # Extract series
    rolling_av_hours = 24
    total_heat_production = results_summary['hourly_wasteProcessed'] * lhv
    heat_for_ccs = []
    wte_heat_to_demand = []
    wte_heat_for_el = []
    wte_heat_for_el_ccs = []
    boiler_output_frac = []
    co2_captured_frac = []

    normalization = "max_wte_heat_prod" #"hourly_wte_heat_prod" "max_wte_heat_prod"


    for j in range(len(total_heat_production)):
        if normalization == ["hourly_wte_heat_prod"]:
            denominator = total_heat_production[j]
        else:
            denominator = max(total_heat_production)


        if total_heat_production[j] > 0:
            heat_for_ccs.append(results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'][j] / denominator)
            wte_heat_to_demand.append(results_summary[dh_ratio_str]['hourly_wte_heat_to_demand'][j] / denominator)
            wte_heat_for_el.append(results_summary[dh_ratio_str]['hourly_wte_heat_for_el'][j] / denominator)
            wte_heat_for_el_ccs.append(results_summary[dh_ratio_str]['hourly_wte_heat_for_el_ccs'][j] / denominator)
            boiler_output_frac.append(results_summary[dh_ratio_str]['hourly_boiler_heat_out'][j] / denominator)
            co2_captured_frac.append(results_summary[dh_ratio_str]['hourly_co2_captured'][j] / results_summary['hourly_emissions'][j])
        else:
            heat_for_ccs.append(0)
            wte_heat_to_demand.append(0)
            wte_heat_for_el.append(0)
            wte_heat_for_el_ccs.append(0)
            boiler_output_frac.append(0)
            co2_captured_frac.append(0)

    time = range(len(heat_for_ccs))

    # Stacked area plot
    ax.stackplot(
        time,
        pd.Series(wte_heat_to_demand).rolling(window=rolling_av_hours).mean(),
        pd.Series(wte_heat_for_el).rolling(window=rolling_av_hours).mean(),
        pd.Series(heat_for_ccs).rolling(window=rolling_av_hours).mean(),
        pd.Series(wte_heat_for_el_ccs).rolling(window=rolling_av_hours).mean(),
        labels=[
            "Heat to demand",
            "Heat for grid el.",
            "Heat for CCS",
            "Heat for CCS el."
        ],
        colors=stack_colors,
        alpha=0.8
    )

    # Add red line for boiler output
    ax.plot(
        time,
        pd.Series(boiler_output_frac).rolling(window=rolling_av_hours).mean(),
        color="red",
        linewidth=1.5,
        label="Boiler output"
    )
    # ax2 = ax.twinx()
    # ax2.plot(
    #     time,
    #     pd.Series(co2_captured_frac).rolling(window=rolling_av_hours).mean(),
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.2,
    #     label="CO₂ captured"
    # )
    # ax2.set_ylabel("Fraction CO₂ captured [-]")  # adjust units to your data
    # ax.set_title(dh_ratio_str)
    # ax.set_xlabel("Time [h]")
    # ax.set_ylabel("Fraction of total heat [-]")
    # ax.set_ylim(0, 1)

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Fraction of heat [-]")
    ax.set_ylim(0, 1.5)
    # Annotation: DH ratio at top-center of each subplot
    ax.text(
        0.5, 0.97,
        f"DH ratio {dh_ratio_str}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
    )

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Collect all handles and labels from the last axis
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for global legend
plt.show()
save_figure_for_paper(fig, "MEA_operations_allDH", results_path)


## Plot the economics

capex_tot = []
opex_fixed = []
opex_variable = []
loss_el_revenues = []
extra_cost_boiler = []
tot_co2_captured = []
capture_cost = []


economics = {
    "capex_tot": capex_tot,
    "opex_fixed": opex_fixed,
    "opex_variable": opex_variable,
    "loss_el_revenues": loss_el_revenues,
    "extra_cost_boiler": extra_cost_boiler,
    "tot_co2_captured": tot_co2_captured,
}

for dh_ratio in explored_dh_ratio_str:
    values = {}
    # Collect each parameter first
    for economic_param, storage_list in economics.items():
        val = results_summary[dh_ratio][economic_param]
        storage_list.append(val)
        values[economic_param] = val

    # Compute abatement cost
    if results_summary[dh_ratio_str]['tot_co2_captured'] > 0:
        capture = (
            values["capex_tot"]
            + values["opex_fixed"]
            + values["opex_variable"]
            + values["loss_el_revenues"]
            + values["extra_cost_boiler"]
        ) / values["tot_co2_captured"]

    else:
        capture_cost = 0

    capture_cost.append(capture)


fig, ax = plt.subplots(figsize=(6, 4))  # You can adjust size if needed

# Scatter points for capture cost
for i, (x, y) in enumerate(zip(explored_dh_ratio, capture_cost)):
    ax.scatter(
        x, y,
        color=batlow_colors[i],
        marker="s",
        s=100,
        edgecolor=batlow_colors[i],
        zorder=3,
        label="Capture Cost" if i == 0 else ""
    )

# Connect points with a dashed line
ax.plot(
    explored_dh_ratio,
    capture_cost,
    linestyle="--",
    color="black",
    alpha=0.6,
    zorder=2
)

# Labels and legend
ax.set_xlabel("Peak district heating demand [-]")
ax.set_ylabel("Capture cost [€/tCO₂]")
ax.legend()

# Show and save
plt.show()
save_figure_for_paper(fig, "MEA_economics_DH50", results_path)


## Cost breakdown
tot_co2_captured = (np.array(tot_co2_captured).reshape(-1, 1))
capex_norm = (np.array(capex_tot) / tot_co2_captured).flatten()
opex_fixed_norm = (np.array(opex_fixed) / tot_co2_captured).flatten()
opex_variable_norm = (np.array(opex_variable) / tot_co2_captured).flatten()
loss_el_revenues_norm = (np.array(loss_el_revenues).reshape(-1, 1) / tot_co2_captured).flatten()
extra_cost_boiler_norm = (np.array(extra_cost_boiler).reshape(-1, 1) / tot_co2_captured).flatten()

plt.figure(figsize=(9, 6))
bar_width = 0.1

plt.bar(explored_dh_ratio, capex_norm, width=bar_width, label="CAPEX", color=batlow_colors[0])
plt.bar(explored_dh_ratio, opex_fixed_norm, width=bar_width, bottom=capex_norm, label="OPEX", color=batlow_colors[1])
plt.bar(explored_dh_ratio, loss_el_revenues_norm, width=bar_width,
        bottom=capex_norm + opex_fixed_norm,
        label="Lost electricity revenues", color=batlow_colors[3])
plt.bar(explored_dh_ratio, extra_cost_boiler_norm, width=bar_width,
        bottom=capex_norm + opex_fixed_norm + loss_el_revenues_norm,
        label="Extra cost boiler", color=batlow_colors[4])

plt.xticks(explored_dh_ratio, explored_dh_ratio_str, rotation=45)
plt.xlabel("Peak district heating demand [-]")
plt.ylabel("Cost breakdown [€/tCO₂]")
plt.ylim([0, 80])
plt.legend()
plt.tight_layout()
plt.show()