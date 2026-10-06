import h5py
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# All paths are relative to this folder, independently of where the script is launched from
os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adopt_net0.result_management.read_results import extract_datasets_from_h5group
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper, draw_tech_selection

# Technology selection as a function of the capex of MEA (x-axis) and of the capex of oxyfuel (y-axis).
# Results of main_cement_capex_sensitivity.py

figures_path = "../figures/cement_tech_selection"

raw_results_path = Path("./raw_results/capex_sensitivity")
capex_matrix = pd.read_csv(raw_results_path / "capex_matrix.csv", sep=";")
cost_extra_fuel = 15

path_processed_data = Path("./dataSources/data_processed.xlsx")
data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
el_price = data["el_price_itNord"]
info_cement = json.loads(Path("./technologies_json/CementEmitter.json").read_text())
emission_factor_clinker_baseline = info_cement["Performance"]["emission_factor"]  # tCO2/tClinker, without oxyfuel calciner
info_heat_pump = json.loads(Path("./technologies_json/HeatPump.json").read_text())
cop_hp = info_heat_pump["Performance"]["performance"]["out"]["heat"][1]


def read_results(case_name):
    """Reads the most recent results of a case"""
    case_dirs = sorted(
        d for d in raw_results_path.glob(f"*_{case_name}") if (d / "optimization_results.h5").exists()
    )
    with h5py.File(case_dirs[-1] / "optimization_results.h5", "r") as hdf_file:
        df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
        df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
        df_design_network = pd.DataFrame(
            extract_datasets_from_h5group(
                hdf_file["design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"]
            )
        )
    return df_operation, df_design, df_design_network


results_summary = []
for _, case in capex_matrix.iterrows():
    df_operation, df_design, df_design_network = read_results(case["case_name"])

    cement_mea_design = df_design.loc[:, ('industrial_cluster', 'CementEmitter')]
    cement_mea_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
    heat_pump_design = df_design.loc[:, ('industrial_cluster', 'HeatPump')]
    cement_oxy_design = df_design.loc[:, ('industrial_cluster', 'CementHybridCCS')]
    cement_oxy_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementHybridCCS')]
    co2_storage_design = df_design.loc[:, ('storage', 'PermanentStorage_CO2_simple')]

    clinker_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster', 'clinker', 'demand')]
    emissions_cement_baseline = sum(clinker_demand * emission_factor_clinker_baseline)

    # economics
    transport_stor_cost = (
        co2_storage_design['opex_variable'].iloc[0] + df_design_network['capex'].values.flatten()[0]
    )

    if cement_mea_design["size_ccs"].iloc[0] > 0:
        type_installed = "MEA"
        capex = cement_mea_design["capex_tot"].iloc[0] + heat_pump_design["capex_tot"].iloc[0]
        opex_fixed = cement_mea_design["opex_fixed"].iloc[0] + heat_pump_design["opex_fixed"].iloc[0]
        opex_variable = cement_mea_design["opex_variable"].iloc[0]
        energy_cost = sum(cement_mea_operation["electricity_var_input_ccs"] * el_price) + sum(
            cement_mea_operation["heat_var_input_ccs"] / cop_hp * el_price)
        co2_captured = cement_mea_operation['CO2captured_var_output_ccs']
        tot_co2_avoided = sum(cement_mea_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
            cement_mea_operation["emissions_pos"])
        ccs_size = cement_mea_design["size_ccs"].iloc[0]

    elif cement_oxy_design["size"].iloc[0] > 0:
        if cement_oxy_design["size_mea"].iloc[0] > 0:
            type_installed = "Oxyfuel + MEA"
        else:
            type_installed = "Oxyfuel"

        capex = cement_oxy_design["capex_tot"].iloc[0]
        opex_fixed = cement_oxy_design["opex_fixed"].iloc[0]
        opex_variable = cement_oxy_design["opex_variable"].iloc[0]
        co2_captured = cement_oxy_operation['CO2captured_output']
        energy_cost = sum(cement_oxy_operation["electricity_input"] * el_price) + sum(
            cement_oxy_operation["extra_fuel_input"] * cost_extra_fuel)
        tot_co2_avoided = sum(cement_oxy_operation["clinker_output"] * emission_factor_clinker_baseline) - sum(
            cement_oxy_operation["emissions_pos"])
        ccs_size = max(co2_captured)

    else:
        type_installed = "none"
        capex = opex_fixed = opex_variable = energy_cost = tot_co2_avoided = ccs_size = 0
        transport_stor_cost = 0
        co2_captured = pd.Series([0])

    results_summary.append(
        {
            "mea_capex_multiplier": case["mea_capex_multiplier"],
            "oxy_capex_multiplier": case["oxy_capex_multiplier"],
            "type_installed": type_installed,
            "size_ccs": ccs_size,
            "capex": capex,
            "opex_fixed": opex_fixed,
            "opex_variable": opex_variable,
            "energy_cost": energy_cost,
            "transport_stor_cost": transport_stor_cost,
            "tot_co2_captured": sum(co2_captured),
            "tot_co2_avoided": tot_co2_avoided,
            "fraction_avoided": tot_co2_avoided / emissions_cement_baseline,
            "cost_of_avoided": (
                (capex + opex_fixed + opex_variable + energy_cost + transport_stor_cost) / tot_co2_avoided
                if type_installed != "none"
                else 0
            ),
        }
    )

results_summary = pd.DataFrame(results_summary)
results_summary.to_csv(raw_results_path / "results_summary.csv", sep=";", index=False)
print(results_summary.to_string())

# Matrices: capex of oxyfuel on the rows (highest on top), capex of MEA on the columns
type_matrix = results_summary.pivot(
    index="oxy_capex_multiplier", columns="mea_capex_multiplier", values="type_installed"
).sort_index(ascending=False).sort_index(axis=1)
cost_matrix = results_summary.pivot(
    index="oxy_capex_multiplier", columns="mea_capex_multiplier", values="cost_of_avoided"
).sort_index(ascending=False).sort_index(axis=1)

types = ["none", "MEA", "Oxyfuel", "Oxyfuel + MEA"]

# SINGLE-COLUMN FIGURE
setup_matplotlib_for_paper("single")
fig, ax = plt.subplots(layout="constrained")
cbar = draw_tech_selection(fig, ax, type_matrix, cost_matrix, types)
# shorter labels in the legend
cbar.ax.set_yticklabels([t.replace(" + ", "\n+ ") for t in types])
ax.set_xticklabels([f"{(m - 1) * 100:+.0f}%" for m in type_matrix.columns])
ax.set_yticklabels([f"{(m - 1) * 100:+.0f}%" for m in type_matrix.index])
ax.set_xlabel("CAPEX increase of MEA [-]")
ax.set_ylabel("CAPEX increase of oxyfuel [-]")

save_figure_for_paper(fig, "cement_tech_selection_capex_sensitivity", figures_path)

plt.show()
