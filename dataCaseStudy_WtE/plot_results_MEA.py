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
from matplotlib import rcParams
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper



colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']
figures_path = "../figures"


## -----------------  DH ratio --------------------------

explored_dh_ratio = [0, 0.25, 0.5, 0.75,1] # ratio of peak DH demand to supply compared to peak heat prod. from WtE
gas_price = 40
carbon_tax = 150
sim_is_timeless = 1

if sim_is_timeless:
    name_sim = "MEA_timeless"
else:
    name_sim = "MEA"

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
raw_results_path = Path("./raw_results/"+name_sim)
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
        df_design_network = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]))
    print(df_operation)

    co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]
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

    pipeline_cost = df_design_network['capex'].values.flatten()[0]
    storage_cost = co2_storage_design['opex_variable']
    transport_stor_cost = storage_cost + pipeline_cost

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
    extra_usage_boiler = sum(boiler_output['heat_output']-baseline_boiler_prod)/th_efficiency_boiler
    results_summary[dh_ratio_str]['hourly_wte_heat_for_el'] = (el_out-w2e_output['electricity_var_input_ccs'])/el_efficiency
    results_summary[dh_ratio_str]['hourly_wte_heat_for_el_ccs'] =  w2e_output['electricity_var_input_ccs']/el_efficiency
    results_summary[dh_ratio_str]['capex_tot'] = w2e_design["capex_ccs"]
    results_summary[dh_ratio_str]['opex_fixed'] = w2e_design["opex_fixed_ccs"]
    results_summary[dh_ratio_str]['opex_variable'] = w2e_design["opex_variable"]
    results_summary[dh_ratio_str]['loss_el_revenues'] = sum((baseline_el_prod-el_out)*el_price)
    results_summary[dh_ratio_str]['extra_cost_boiler'] = extra_usage_boiler*(emission_factor_boiler*carbon_tax + gas_price)
    results_summary[dh_ratio_str]['transport_stor_cost'] = transport_stor_cost
    results_summary[dh_ratio_str]['tot_co2_captured'] = sum(co2_captured_w2e)
    results_summary[dh_ratio_str]['tot_co2_avoided'] = sum(emissions_w2e)-(sum(emissions_w2e-co2_captured_w2e)+extra_usage_boiler*emission_factor_boiler)
    results_summary['hourly_emissions'] = emissions_w2e
    results_summary['hourly_wasteProcessed'] = waste_processed_out
    results_summary[dh_ratio_str]['heat_demand'] = heat_demand
    results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'] = w2e_output['heat_var_input_ccs'] / th_efficiency

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


# PAPER SETUP (ONE LINE SWITCH)
setup_matplotlib_for_paper(column="double")   # "single" or "double"

# COLORS
stack_colors = [
    batlow_colors[0],
    batlow_colors[1],
    batlow_colors[2],
    batlow_colors[3]
]

# SUBPLOT GRID
n_plots = len(explored_dh_ratio_str)
ncols = 2
nrows = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    sharex=False,
    sharey=True
)

axes = axes.flatten()

# LOOP OVER CASES
for i, dh_ratio_str in enumerate(explored_dh_ratio_str):

    ax = axes[i]

    rolling_av_hours = 24
    total_heat_production = results_summary['hourly_wasteProcessed'] * lhv

    heat_for_ccs = []
    wte_heat_to_demand = []
    wte_heat_for_el = []
    wte_heat_for_el_ccs = []
    boiler_output_frac = []

    for j in range(len(total_heat_production)):

        denominator = max(total_heat_production)

        if total_heat_production[j] > 0:
            heat_for_ccs.append(
                results_summary[dh_ratio_str]['hourly_wte_heat_for_heat_ccs'][j] / denominator
            )
            wte_heat_to_demand.append(
                results_summary[dh_ratio_str]['heat_demand'][j] / denominator
            )
            wte_heat_for_el.append(
                results_summary[dh_ratio_str]['hourly_wte_heat_for_el'][j] / denominator
            )
            wte_heat_for_el_ccs.append(
                results_summary[dh_ratio_str]['hourly_wte_heat_for_el_ccs'][j] / denominator
            )
            boiler_output_frac.append(
                results_summary[dh_ratio_str]['hourly_boiler_heat_out'][j] / denominator
            )
        else:
            heat_for_ccs.append(0)
            wte_heat_to_demand.append(0)
            wte_heat_for_el.append(0)
            wte_heat_for_el_ccs.append(0)
            boiler_output_frac.append(0)

    time = range(len(heat_for_ccs))

    # --------------------------------------------------------------
    # STACKED AREA
    ax.stackplot(
        time,
        pd.Series(wte_heat_to_demand).rolling(rolling_av_hours).mean(),
        pd.Series(wte_heat_for_el).rolling(rolling_av_hours).mean(),
        pd.Series(heat_for_ccs).rolling(rolling_av_hours).mean(),
        pd.Series(wte_heat_for_el_ccs).rolling(rolling_av_hours).mean(),
        labels=[
            "District heating demand",
            "Electricity to grid",
            "Heat for CCS",
            "Electricity for CCS"
        ],
        colors=stack_colors,
        alpha=0.8
    )

    # BOILER OUTPUT
    ax.plot(
        time,
        pd.Series(boiler_output_frac).rolling(rolling_av_hours).mean(),
        color="red",
        linewidth=1.2,
        label="Boiler output"
    )

    # AXIS LABELS
    if dh_ratio_str in {"0", "0.5", "1"}:
        ax.set_ylabel("Fraction of heat [-]")
    if dh_ratio_str in {"0.75", "1"}:
        ax.set_xlabel("Time [h]")

    ax.set_xlim(0, 8760)

    # PANEL TITLE
    ax.text(
        0.5,
        0.97,
        f"DH ratio {dh_ratio_str}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=rcParams["axes.labelsize"],
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.8,
            edgecolor="none"
        )
    )

# LEGEND IN EMPTY PANEL
empty_ax_idx = i + 1
handles, labels = axes[i].get_legend_handles_labels()

legend_ax = axes[empty_ax_idx]
legend_ax.axis("off")

legend = legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=True,
    ncol=1,
    fontsize=rcParams["legend.fontsize"]
)

legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.0)

# Remove unused axes
for j in range(empty_ax_idx + 1, len(axes)):
    fig.delaxes(axes[j])


fig.tight_layout(pad=0.6)
save_figure_for_paper(fig, f"{name_sim}_operations_allDH", figures_path)





## Plot the economics

capex_tot = []
opex_fixed = []
opex_variable = []
loss_el_revenues = []
extra_cost_boiler = []
transport_stor_cost = []
correct_for_avoided = []
tot_co2_captured = []
capture_cost = []


economics = {
    "capex_tot": capex_tot,
    "opex_fixed": opex_fixed,
    "opex_variable": opex_variable,
    "loss_el_revenues": loss_el_revenues,
    "extra_cost_boiler": extra_cost_boiler,
    "transport_stor_cost": transport_stor_cost
}

emissions ={
    "correct_for_avoided": correct_for_avoided,
    "tot_co2_captured": tot_co2_captured,
}

for dh_ratio in explored_dh_ratio_str:
    for emissions_param, storage_list in emissions.items():
        if emissions_param == "tot_co2_captured":
            val = results_summary[dh_ratio]["tot_co2_captured"]
        elif emissions_param == "correct_for_avoided":
            val = (
                results_summary[dh_ratio]["tot_co2_captured"]
                / results_summary[dh_ratio]["tot_co2_avoided"]
            )
        storage_list.append(val)
    values = {}
    # Collect each parameter first
    for economic_param, storage_list in economics.items():
        val = results_summary[dh_ratio][economic_param]/ results_summary[dh_ratio]['tot_co2_captured']
        storage_list.append(val)
        values[economic_param] = val

    # Compute abatement cost
    if results_summary[dh_ratio]['tot_co2_captured'] > 0:
        capture = (
            values["capex_tot"]
            + values["opex_fixed"]
            + values["opex_variable"]
            + values["loss_el_revenues"]
            + values["extra_cost_boiler"]
            + values["transport_stor_cost"]
        )

    else:
        capture_cost = 0

    capture_cost.append(capture)


# fig, ax = plt.subplots(figsize=(6, 4))  # You can adjust size if needed
#
# # Scatter points for capture cost
# for i, (x, y) in enumerate(zip(explored_dh_ratio, capture_cost)):
#     ax.scatter(
#         x, y,
#         color=batlow_colors[i],
#         marker="s",
#         s=100,
#         edgecolor=batlow_colors[i],
#         zorder=3,
#         label="Capture Cost" if i == 0 else ""
#     )
#
# # Connect points with a dashed line
# ax.plot(
#     explored_dh_ratio,
#     capture_cost,
#     linestyle="--",
#     color="black",
#     alpha=0.6,
#     zorder=2
# )
#
# # Labels and legend
# ax.set_xlabel("Peak district heating demand [-]")
# ax.set_ylabel("Capture cost [€/tCO₂]")
# ax.legend()
#
# save_figure_for_paper(fig, "MEA_economics_DH50", figures_path)


## Cost breakdown

correct_factor = np.array(emissions["correct_for_avoided"], dtype=float).ravel()

capex   = np.array(economics["capex_tot"], dtype=float).ravel() * correct_factor
opex_f  = np.array(economics["opex_fixed"], dtype=float).ravel() * correct_factor
t_s     = np.array(economics["transport_stor_cost"], dtype=float).ravel() * correct_factor
extra_cost_boiler = np.array(economics["extra_cost_boiler"], dtype=float).ravel() * correct_factor
loss_el_revenues = np.array(economics["loss_el_revenues"], dtype=float).ravel() * correct_factor

fig_width, fig_height = setup_matplotlib_for_paper(column="single")
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

bar_width = 0.1
ax.bar(explored_dh_ratio, capex, width=bar_width, label="CAPEX", color=batlow_colors[0])
ax.bar(explored_dh_ratio, opex_f, width=bar_width, bottom=capex,
       label="OPEX fixed", color=batlow_colors[1])
ax.bar(explored_dh_ratio, t_s, width=bar_width,
       bottom=capex + opex_f, label="Transport & Storage", color=batlow_colors[3])
ax.bar(explored_dh_ratio, loss_el_revenues, width=bar_width,
       bottom=capex + opex_f + t_s,
       label="Lost electricity revenues", color=batlow_colors[4])
ax.bar(explored_dh_ratio, extra_cost_boiler, width=bar_width,
       bottom=capex + opex_f + t_s + loss_el_revenues,
       label="Extra cost boiler", color=batlow_colors[5])

ax.set_xticks(explored_dh_ratio)
ax.set_xticklabels(explored_dh_ratio_str, rotation=45)
ax.set_xlabel("Peak district heating demand [-]")
ax.set_ylabel("CAC [€/tCO$_2$]")
ax.set_ylim(0, 130)
ax.legend()

fig.tight_layout()
save_figure_for_paper(fig, f"{name_sim}_cost_breakdown", figures_path)
plt.show()
