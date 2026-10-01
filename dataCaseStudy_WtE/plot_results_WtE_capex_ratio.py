import h5py
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib import rcParams

# All paths are relative to this folder, independently of where the script is launched from
os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adopt_net0.result_management.read_results import extract_datasets_from_h5group
from utilities.process_results import save_figure_for_paper, setup_matplotlib_for_paper

# Technology selection as a function of the capex of MEA (x-axis) and of the ratio between the capex of CaL and
# MEA (y-axis). Results of main_WtE_capex_ratio.py

batlow_colors = [
    "#222A6A",
    "#4B708A",
    "#6FBC7B",
    "#B1E87E",
    "#F7D03C",
    "#D491B8",
    "#012E4D",
]
figures_path = "../figures"

raw_results_path = Path("./raw_results/capex_ratio")
capex_matrix = pd.read_csv(raw_results_path / "capex_matrix.csv", sep=";")
capex_matrix_info = json.loads((raw_results_path / "capex_matrix_info.json").read_text())
gas_price = 40
import_price_RDF = 20

path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
el_price = data["el_price_itNord"]
emission_factor = data["emission_factor_PAIP"]

info_boiler = json.loads(Path("./technologies_json/Boiler_Industrial_NG.json").read_text())
th_efficiency_boiler = info_boiler["Performance"]["performance"]["out"]["heat"][1]
emission_factor_boiler = info_boiler["Performance"]["emission_factor"]
info_WasteCaL_CCS = json.loads(Path("./technologies_json/WasteCaL_CCS.json").read_text())
emission_factor_rdf = info_WasteCaL_CCS["Performance"]["emission_factor_RDF"]


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


def tec_operation(df_operation, tec):
    return df_operation.loc[:, ("technology_operation", "period1", "industrial_cluster", tec)]


# Benchmark without CCS
df_operation, _, _ = read_results(capex_matrix_info["no_ccs_case_name"])
revenues_no_ccs = sum(tec_operation(df_operation, "WasteCHP")["electricity_output"] * el_price)
tot_boiler_out_no_ccs = sum(tec_operation(df_operation, "Boiler_Industrial_NG_existing")["heat_output"])

results_summary = []
for _, case in capex_matrix.iterrows():
    df_operation, df_design, df_design_network = read_results(case["case_name"])

    waste_in = df_operation.loc[
        :, ("energy_balance", "period1", "industrial_cluster", "wasteProcessed", "demand")
    ]
    boiler_output = tec_operation(df_operation, "Boiler_Industrial_NG_existing")
    w2e_design = df_design.loc[:, ("industrial_cluster", "WasteCHP")]
    w2e_CaL_design = df_design.loc[:, ("industrial_cluster", "WasteCaL_CCS")]
    co2_storage_design = df_design.loc[:, ("storage", "PermanentStorage_CO2_simple")]
    transport_stor_cost = (
        co2_storage_design["opex_variable"].iloc[0] + df_design_network["capex"].values.flatten()[0]
    )
    emissions_boiler = sum(boiler_output["heat_output"]) / th_efficiency_boiler * emission_factor_boiler
    emission_baseline = (
        sum(waste_in * emission_factor)
        + tot_boiler_out_no_ccs / th_efficiency_boiler * emission_factor_boiler
    )

    if w2e_design["size"].iloc[0] > 0 and w2e_design["size_ccs"].iloc[0] > 0:
        w2e_operation = tec_operation(df_operation, "WasteCHP")
        co2_captured = w2e_operation["CO2captured_var_output_ccs"]
        loss_el_revenues = revenues_no_ccs - sum(w2e_operation["electricity_output"] * el_price)
        extra_cost_boiler = (
            (sum(boiler_output["heat_output"]) - tot_boiler_out_no_ccs) / th_efficiency_boiler * gas_price
        )

        type_installed = "MEA"
        size_ccs = w2e_design["size_ccs"].iloc[0]
        capex = w2e_design["capex_ccs"].iloc[0]
        opex_fixed = w2e_design["opex_fixed_ccs"].iloc[0]
        opex_variable = w2e_design["opex_variable_ccs"].iloc[0]
        energy_cost = loss_el_revenues + extra_cost_boiler
        tot_co2_avoided = emission_baseline - (
            sum(w2e_operation["wasteIn_input"] * emission_factor - co2_captured) + emissions_boiler
        )

    elif w2e_CaL_design["size"].iloc[0] > 0 and w2e_CaL_design["size_cal"].iloc[0] > 0:
        w2e_cal_operation = tec_operation(df_operation, "WasteCaL_CCS")
        co2_captured = w2e_cal_operation["CO2captured_output"]
        emissions_w2e = (
            w2e_cal_operation["wasteIn_input"] * emission_factor
            + w2e_cal_operation["wasteInRDF_input"] * emission_factor_rdf
        )

        type_installed = "CaL"
        size_ccs = w2e_CaL_design["size_cal"].iloc[0]
        capex = w2e_CaL_design["capex_tot"].iloc[0]
        opex_fixed = w2e_CaL_design["opex_fixed"].iloc[0]
        opex_variable = w2e_CaL_design["opex_variable"].iloc[0]
        energy_cost = -sum(w2e_cal_operation["el_cal"] * el_price) + sum(
            w2e_cal_operation["wasteInRDF_input"] * import_price_RDF
        )
        tot_co2_avoided = sum(w2e_cal_operation["wasteIn_input"] * emission_factor) - sum(
            emissions_w2e - co2_captured
        )

    else:
        type_installed = "none"
        size_ccs = capex = opex_fixed = opex_variable = energy_cost = tot_co2_avoided = 0
        transport_stor_cost = 0
        co2_captured = pd.Series([0])

    results_summary.append(
        {
            "mea_capex_multiplier": case["mea_capex_multiplier"],
            "capex_ratio": case["capex_ratio"],
            "cal_capex_multiplier": case["cal_capex_multiplier"],
            "type_installed": type_installed,
            "size_ccs": size_ccs,
            "capex": capex,
            "opex_fixed": opex_fixed,
            "opex_variable": opex_variable,
            "energy_cost": energy_cost,
            "transport_stor_cost": transport_stor_cost,
            "tot_co2_captured": sum(co2_captured),
            "tot_co2_avoided": tot_co2_avoided,
            "fraction_avoided": tot_co2_avoided / emission_baseline,
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

# Matrices: capex ratio on the rows (baseline ratio on top), capex of MEA on the columns
type_matrix = results_summary.pivot(
    index="capex_ratio", columns="mea_capex_multiplier", values="type_installed"
).sort_index(ascending=False).sort_index(axis=1)
cost_matrix = results_summary.pivot(
    index="capex_ratio", columns="mea_capex_multiplier", values="cost_of_avoided"
).sort_index(ascending=False).sort_index(axis=1)

types = ["none", "MEA", "CaL"]
type_to_color = {t: batlow_colors[i] for i, t in enumerate(types)}

# SINGLE-COLUMN FIGURE
setup_matplotlib_for_paper("single")
fig, ax = plt.subplots()

for i, ratio in enumerate(type_matrix.index):
    for j, multiplier in enumerate(type_matrix.columns):
        t = type_matrix.loc[ratio, multiplier]
        c = cost_matrix.loc[ratio, multiplier]

        ax.add_patch(
            plt.Rectangle(
                (j, i),
                1,
                1,
                facecolor=type_to_color[t],
                edgecolor="black",
                linewidth=0.8,
            )
        )

        # Cost of CO2 avoided [EUR/tCO2] label
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{c:.1f}" if t != "none" else "-",
            ha="center",
            va="center",
            color="white",
            fontsize=rcParams["axes.labelsize"] - 2,
            fontweight="bold",
        )

# AXES FORMATTING
ax.set_xlim(0, len(type_matrix.columns))
ax.set_ylim(0, len(type_matrix.index))
ax.set_xticks([x + 0.5 for x in range(len(type_matrix.columns))])
ax.set_yticks([y + 0.5 for y in range(len(type_matrix.index))])
ax.set_xticklabels([f"+{(m - 1) * 100:.0f}%" for m in type_matrix.columns])
ax.set_yticklabels(
    [
        f"{r:.1f}*" if abs(r - capex_matrix_info["baseline_capex_ratio"]) < 1e-6 else f"{r:.1f}"
        for r in type_matrix.index
    ]
)
ax.invert_yaxis()
ax.set_xlabel("CAPEX increase of MEA [-]")
ax.set_ylabel("CAPEX ratio CaL/MEA [-]\n(* baseline)")

# LEGEND (TOP, HORIZONTAL, SCALED)
patches = [mpatches.Patch(color=type_to_color[t], label=t) for t in types]
ax.legend(
    handles=patches,
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=len(types),
    fontsize=rcParams["legend.fontsize"],
    frameon=False,
)

fig.tight_layout(pad=0.6)
save_figure_for_paper(fig, "wte_tech_selection_capex_ratio", figures_path)

plt.show()
