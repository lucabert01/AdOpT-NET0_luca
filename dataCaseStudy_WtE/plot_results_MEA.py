import h5py
from pathlib import Path
from adopt_net0.result_management.read_results import (
    extract_datasets_from_h5group,
)
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import warnings
from matplotlib import rcParams
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper
from matplotlib.patches import Patch


# ======================================================
# SETTINGS
# ======================================================

simulations = ["MEA", "MEA_timeless"]

explored_dh_ratio = [0, 0.25, 0.5, 0.75, 1]
explored_dh_ratio_str = [str(r) for r in explored_dh_ratio]

gas_price = 40
carbon_tax = 150

batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B',
                 '#B1E87E', '#F7D03C', '#D491B8', '#012E4D']

figures_path = "../figures"


# ======================================================
# LOAD STATIC DATA
# ======================================================

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


# ======================================================
# READ RESULTS (BOTH SIMULATIONS)
# ======================================================

results_summary = {}

for name_sim in simulations:

    print(f"\nProcessing simulation: {name_sim}")
    results_summary[name_sim] = {}

    raw_results_path = Path("./raw_results/" + name_sim)

    dh_ratio_dirs = [d for d in raw_results_path.iterdir()
                     if d.is_dir() and "dh_ratio" in d.name]

    dir_results_sorted = sorted(dh_ratio_dirs)
    file_names = [d.name for d in dir_results_sorted[-num_cases:]]

    for i in range(len(file_names)):

        dh_ratio_str = explored_dh_ratio_str[i]
        results_summary[name_sim][dh_ratio_str] = {}

        file_path = raw_results_path / f"{file_names[i]}/optimization_results.h5"

        with h5py.File(file_path, 'r') as hdf_file:
            df_operation = pd.DataFrame(
                extract_datasets_from_h5group(hdf_file["operation"]))
            df_design = pd.DataFrame(
                extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
            df_design_network = pd.DataFrame(
                extract_datasets_from_h5group(
                    hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]))

        co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]
        w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCHP')]
        boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG_existing')]

        w2e_output = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCHP')]
        boiler_output = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster',
                                             'Boiler_Industrial_NG_existing')]
        heat_demand = df_operation.loc[:, ('energy_balance', 'period1',
                                           'industrial_cluster', 'heat', 'demand')]

        waste_in = w2e_output['wasteIn_input']
        el_out = w2e_output['electricity_output']
        co2_captured_w2e = w2e_output['CO2captured_var_output_ccs']

        emissions_w2e = waste_in * emission_factor

        pipeline_cost = df_design_network['capex'].values.flatten()[0]
        storage_cost = co2_storage_design['opex_variable']
        transport_stor_cost = storage_cost + pipeline_cost

        baseline_el_prod = ((waste_in * lhv - heat_demand / th_efficiency)
                            * el_efficiency).where((waste_in * lhv - heat_demand / th_efficiency) > 0, 0)

        baseline_boiler_prod = (heat_demand - waste_in * lhv *
                                th_efficiency).where((heat_demand - waste_in *
                                                      lhv * th_efficiency) > 0, 0)

        extra_usage_boiler = sum(
            boiler_output['heat_output'] - baseline_boiler_prod) / th_efficiency_boiler

        # Store results
        results_summary[name_sim][dh_ratio_str]['hourly_wasteProcessed'] = w2e_output['wasteProcessed_output']
        results_summary[name_sim][dh_ratio_str]['hourly_boiler_heat_out'] = boiler_output['heat_output']
        results_summary[name_sim][dh_ratio_str]['heat_demand'] = heat_demand
        results_summary[name_sim][dh_ratio_str]['hourly_wte_heat_for_heat_ccs'] = \
            w2e_output['heat_var_input_ccs'] / th_efficiency
        results_summary[name_sim][dh_ratio_str]['hourly_wte_heat_for_el'] = \
            (el_out - w2e_output['electricity_var_input_ccs']) / el_efficiency
        results_summary[name_sim][dh_ratio_str]['hourly_wte_heat_for_el_ccs'] = \
            w2e_output['electricity_var_input_ccs'] / el_efficiency

        results_summary[name_sim][dh_ratio_str]['capex_tot'] = float(w2e_design["capex_ccs"])
        results_summary[name_sim][dh_ratio_str]['opex_fixed'] = float(w2e_design["opex_fixed_ccs"])
        results_summary[name_sim][dh_ratio_str]['transport_stor_cost'] = float(transport_stor_cost)
        results_summary[name_sim][dh_ratio_str]['loss_el_revenues'] = float(sum(
            (baseline_el_prod - el_out) * el_price))
        results_summary[name_sim][dh_ratio_str]['extra_cost_boiler'] = float(
            extra_usage_boiler * (emission_factor_boiler * carbon_tax + gas_price))

        results_summary[name_sim][dh_ratio_str]['tot_co2_captured'] = float(sum(
            co2_captured_w2e))
        results_summary[name_sim][dh_ratio_str]['tot_co2_avoided'] = float(
            sum(emissions_w2e) - (sum(emissions_w2e - co2_captured_w2e)
                                  + extra_usage_boiler * emission_factor_boiler))


# ======================================================
# PLOT #1 (UNCHANGED STRUCTURE, PER SIMULATION)
# ======================================================

for name_sim in simulations:

    setup_matplotlib_for_paper(column="double")

    stack_colors = [
        batlow_colors[0],
        batlow_colors[1],
        batlow_colors[2],
        batlow_colors[3]
    ]

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

    for i, dh_ratio_str in enumerate(explored_dh_ratio_str):

        ax = axes[i]

        rolling_av_hours = 24
        total_heat_production = (
            results_summary[name_sim][dh_ratio_str]['hourly_wasteProcessed'] * lhv
        )

        heat_for_ccs = []
        wte_heat_to_demand = []
        wte_heat_for_el = []
        wte_heat_for_el_ccs = []
        boiler_output_frac = []

        for j in range(len(total_heat_production)):

            denominator = max(total_heat_production)

            if total_heat_production[j] > 0:
                heat_for_ccs.append(
                    results_summary[name_sim][dh_ratio_str]['hourly_wte_heat_for_heat_ccs'][j] / denominator
                )
                wte_heat_to_demand.append(
                    results_summary[name_sim][dh_ratio_str]['heat_demand'][j] / denominator
                )
                wte_heat_for_el.append(
                    results_summary[name_sim][dh_ratio_str]['hourly_wte_heat_for_el'][j] / denominator
                )
                wte_heat_for_el_ccs.append(
                    results_summary[name_sim][dh_ratio_str]['hourly_wte_heat_for_el_ccs'][j] / denominator
                )
                boiler_output_frac.append(
                    results_summary[name_sim][dh_ratio_str]['hourly_boiler_heat_out'][j] / denominator
                )
            else:
                heat_for_ccs.append(0)
                wte_heat_to_demand.append(0)
                wte_heat_for_el.append(0)
                wte_heat_for_el_ccs.append(0)
                boiler_output_frac.append(0)

        time = range(len(heat_for_ccs))

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

        ax.plot(
            time,
            pd.Series(boiler_output_frac).rolling(rolling_av_hours).mean(),
            color="red",
            linewidth=1.2,
            label="Boiler output"
        )

        if dh_ratio_str in {"0", "0.5", "1"}:
            ax.set_ylabel("Fraction of heat [-]")
        if dh_ratio_str in {"0.75", "1"}:
            ax.set_xlabel("Time [h]")

        ax.set_xlim(0, 8760)

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

    # --- LEGEND IN EMPTY PANEL (EXACTLY LIKE YOUR ORIGINAL) ---
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

# ======================================================
# PLOT #2 (GROUPED STACKED, BATLOW COLORS)
# ======================================================

# -----------------------------
# Compute economics
# -----------------------------
economics_all = {}

for name_sim in simulations:

    capex, opex_f, transport, loss_el, boiler = [], [], [], [], []
    correct_factor = []

    for dh_ratio_str in explored_dh_ratio_str:

        captured = results_summary[name_sim][dh_ratio_str]['tot_co2_captured']
        avoided  = results_summary[name_sim][dh_ratio_str]['tot_co2_avoided']

        if captured == 0 or avoided == 0:
            capex.append(0)
            opex_f.append(0)
            transport.append(0)
            loss_el.append(0)
            boiler.append(0)
            correct_factor.append(0)
        else:
            factor = captured / avoided
            correct_factor.append(factor)

            capex.append(results_summary[name_sim][dh_ratio_str]['capex_tot'] / captured)
            opex_f.append(results_summary[name_sim][dh_ratio_str]['opex_fixed'] / captured)
            transport.append(results_summary[name_sim][dh_ratio_str]['transport_stor_cost'] / captured)
            loss_el.append(results_summary[name_sim][dh_ratio_str]['loss_el_revenues'] / captured)
            boiler.append(results_summary[name_sim][dh_ratio_str]['extra_cost_boiler'] / captured)

    capex     = np.array(capex, dtype=float) * np.array(correct_factor)
    opex_f    = np.array(opex_f, dtype=float) * np.array(correct_factor)
    transport = np.array(transport, dtype=float) * np.array(correct_factor)
    loss_el   = np.array(loss_el, dtype=float) * np.array(correct_factor)
    boiler    = np.array(boiler, dtype=float) * np.array(correct_factor)

    economics_all[name_sim] = {
        "capex": capex,
        "opex_f": opex_f,
        "transport": transport,
        "loss_el": loss_el,
        "boiler": boiler,
        "total": capex + opex_f + transport + loss_el + boiler
    }

# -----------------------------
# Plot
# -----------------------------
fig_width, fig_height = setup_matplotlib_for_paper(column="single")
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

x = np.arange(len(explored_dh_ratio))
bar_width = 0.38

for idx, name_sim in enumerate(simulations):

    offset = -bar_width/2 if idx == 0 else bar_width/2
    eco = economics_all[name_sim]

    # Scenario styling
    hatch = None if idx == 0 else "////"
    edgecolor = "black"
    linewidth = 0.6

    # --- Stacked bars ---
    ax.bar(x + offset, eco["capex"],
           width=bar_width,
           color=batlow_colors[0],
           edgecolor=edgecolor,
           linewidth=linewidth,
           hatch=hatch,
           label="CAPEX" if idx == 0 else "")

    ax.bar(x + offset, eco["opex_f"],
           width=bar_width,
           bottom=eco["capex"],
           color=batlow_colors[1],
           edgecolor=edgecolor,
           linewidth=linewidth,
           hatch=hatch,
           label="OPEX fixed" if idx == 0 else "")

    ax.bar(x + offset, eco["transport"],
           width=bar_width,
           bottom=eco["capex"] + eco["opex_f"],
           color=batlow_colors[3],
           edgecolor=edgecolor,
           linewidth=linewidth,
           hatch=hatch,
           label="Transport & storage" if idx == 0 else "")

    # <-- Swap these two -->
    ax.bar(x + offset, eco["loss_el"],
           width=bar_width,
           bottom=eco["capex"] + eco["opex_f"] + eco["transport"],
           color=batlow_colors[4],
           edgecolor=edgecolor,
           linewidth=linewidth,
           hatch=hatch,
           label="Lost el. revenues" if idx == 0 else "")

    ax.bar(x + offset, eco["boiler"],
           width=bar_width,
           bottom=eco["capex"] + eco["opex_f"] + eco["transport"] + eco["loss_el"],
           color=batlow_colors[5],
           edgecolor=edgecolor,
           linewidth=linewidth,
           hatch=hatch,
           label="Extra boiler cost" if idx == 0 else "")

    # --- Total cost marker ---
    ax.plot(x + offset,
            eco["total"],
            marker="o",
            linestyle="none",
            color="black",
            markersize=4,
            zorder=5)

# -----------------------------
# Axis formatting
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in explored_dh_ratio])
ax.set_xlabel("District heating demand ratio [-]")
ax.set_ylabel("CO$_2$ avoidance cost [€/tCO$_2$]")
ax.set_ylim(0, 150)

ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# -----------------------------
# Legends
# -----------------------------

# Cost component legend (colors)
handles_components, labels_components = ax.get_legend_handles_labels()

legend1 = ax.legend(handles_components,
                    labels_components,
                    ncol=2,
                    frameon=False,
                    loc="upper left")

ax.add_artist(legend1)

# Scenario legend (hatch vs solid)
scenario_handles = [
    Patch(facecolor="white",
          edgecolor="black",
          hatch=None,
          label="Time-resolved"),

    Patch(facecolor="white",
          edgecolor="black",
          hatch="////",
          label="Static")
]

legend2 = ax.legend(handles=scenario_handles,
                    frameon=False,
                    loc="upper right")

fig.tight_layout()

save_figure_for_paper(
    fig,
    "MEA_vs_MEA_timeless_cost_breakdown",
    figures_path
)

plt.show()