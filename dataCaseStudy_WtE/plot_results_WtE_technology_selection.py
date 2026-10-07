import h5py
import json
import pprint
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
from adopt_net0.result_management.read_results import (
    print_h5_tree,
    extract_datasets_from_h5group,
)
from utilities.process_results import (
    save_figure_for_paper,
    setup_matplotlib_for_paper,
    draw_tech_selection,
    draw_heatmap,
    SEQUENTIAL_CMAPS,
)
import io
from pptx import Presentation
from pptx.util import Inches

# Set global styling for the plots
colors = []
figures_path = "../figures"
figures_path_tech_selection = "../figures/wte_tech_selection"

## ----------------- Carbon and electricity price --------------------------
explored_carbon_tax = [100, 150, 200, 250]
explored_el_price = [
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
]  # average el prices explored in the analysis
explored_dh_ratio = [0.5]
gas_price = 40
import_price_RDF = 20

path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
av_el_price = data["el_price_itNord"].mean()
electricity_price_norm = data["el_price_itNord"] / av_el_price
emission_factor = data["emission_factor_PAIP"]
lhv = data["lhv_PAIP"]

def extract_no_ccs_results(
    raw_results_path,
    explored_carbon_tax,
    explored_el_price,
    explored_dh_ratio,
    electricity_price_norm,
):
    """
    Reads simulations from a path without CCS and extracts boiler output and el. revenues.
    """
    results_no_ccs = {}

    # Standardize string lists for path searching
    explored_carbon_tax_str = [str(r) for r in explored_carbon_tax]
    explored_el_price_str = [str(r) for r in explored_el_price]
    explored_dh_ratio_str = [str(r) for r in explored_dh_ratio]

    num_el_prices = len(explored_el_price)
    num_carbon_tax = len(explored_carbon_tax)

    for i_dh in range(len(explored_dh_ratio)):
        dh_ratio_str = f"dh_{explored_dh_ratio_str[i_dh]}"
        results_no_ccs[dh_ratio_str] = {}

        # Filter directories by DH ratio
        dh_ratio_dirs = sorted(
            [
                d
                for d in raw_results_path.iterdir()
                if d.is_dir() and dh_ratio_str in d.name
            ]
        )

        # Get the most recent batch based on total expected simulations
        dh_ratio_names = [d.name for d in dh_ratio_dirs[-num_el_prices * num_carbon_tax :]]

        for j in range(num_carbon_tax):
            ctax_str_key = f"ctax_{explored_carbon_tax_str[j]}"
            results_no_ccs[dh_ratio_str][ctax_str_key] = {}

            # Filter by carbon tax string
            carbon_tax_names = sorted([d for d in dh_ratio_names if ctax_str_key in d])
            current_batch = carbon_tax_names[-num_el_prices:]

            for i in range(num_el_prices):
                # Calculate the hourly price vector for this specific simulation
                current_el_price_vector = explored_el_price[i] * electricity_price_norm

                el_price_key = f"el_price_{explored_el_price_str[i]}"
                file_path = (
                    raw_results_path / current_batch[i] / "optimization_results.h5"
                )

                with h5py.File(file_path, "r") as hdf_file:
                    df_op = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))

                # --- Extraction ---
                # Boiler output
                boiler_op = df_op.loc[
                    :,
                    (
                        "technology_operation",
                        "period1",
                        "industrial_cluster",
                        "Boiler_Industrial_NG_existing",
                    ),
                ]
                # W2E operation
                w2e_op = df_op.loc[
                    :, ("technology_operation", "period1", "industrial_cluster", "WasteCHP")
                ]

                # --- Calculations ---
                el_out = w2e_op["electricity_output"]
                boiler_heat_out = boiler_op["heat_output"]

                results_no_ccs[dh_ratio_str][ctax_str_key][el_price_key] = {
                    "electricity_revenues": sum(el_out * current_el_price_vector),
                    "tot_boiler_out": sum(boiler_heat_out),
                    "el_out": el_out,  # keeping hourly for potential plots
                    "boiler_heat_output": boiler_heat_out,
                }

    return results_no_ccs


path_no_ccs = Path("./raw_results/WtE_withoutCCS")
no_ccs_summary = extract_no_ccs_results(
    path_no_ccs,
    explored_carbon_tax,
    explored_el_price,
    explored_dh_ratio,
    electricity_price_norm,
)

num_el_prices = len(explored_el_price)
num_carbon_tax = len(explored_carbon_tax)
num_dh_ratio = len(explored_dh_ratio)
raw_results_path = Path("./raw_results/technology_selection")
explored_carbon_tax_str = [str(r) for r in explored_carbon_tax]
explored_el_price_str = [str(r) for r in explored_el_price]
explored_dh_ratio_str = [str(r) for r in explored_dh_ratio]
results_summary = {}

for i_dh in range(0, num_dh_ratio):
    dh_ratio = explored_dh_ratio[i_dh]
    dh_ratio_str = f"dh_{explored_dh_ratio_str[i_dh]}"
    results_summary[dh_ratio_str] = {}

    # Get all directories that contain 'dh_ratio_str' in the name
    dh_ratio_dirs = [
        d for d in raw_results_path.iterdir() if d.is_dir() and dh_ratio_str in d.name
    ]

    # Sort directories by name
    dir_results_sorted = sorted(dh_ratio_dirs)

    # Get the most recent ones
    dh_ratio_names = [
        d.name for d in dir_results_sorted[-num_el_prices * num_carbon_tax :]
    ]
    for j in range(0, num_carbon_tax):
        carbon_tax = explored_carbon_tax[j]
        carbon_tax_str = f"ctax_{explored_carbon_tax_str[j]}"
        results_summary[dh_ratio_str][carbon_tax_str] = {}

        # Get all file names that contain 'carbon_tax_str' in the name
        carbon_tax_dirs = [d for d in dh_ratio_names if carbon_tax_str in d]

        # Sort them by name
        dir_results_sorted = sorted(carbon_tax_dirs)

        # Get the most recent ones at the value of carbon tax carbon_tax_str
        carbon_tax_names = [d for d in dir_results_sorted[-num_el_prices:]]
        for i in range(0, num_el_prices):
            el_price = explored_el_price[i] * electricity_price_norm

            file_path = raw_results_path / f"{carbon_tax_names[i]}/optimization_results.h5"

            # Check if each explored_el_price[i] is in file_names[i]
            el_price_str = f"el_price_{explored_el_price[i]}"
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str] = {}

            with h5py.File(file_path, "r") as hdf_file:
                df_operation = pd.DataFrame(
                    extract_datasets_from_h5group(hdf_file["operation"])
                )
                df_design = pd.DataFrame(
                    extract_datasets_from_h5group(hdf_file["design/nodes/period1"])
                )
                df_design_network = pd.DataFrame(
                    extract_datasets_from_h5group(
                        hdf_file[
                            "design/networks/period1/CO2PipelineOnshore/industrial_clusterstorage"
                        ]
                    )
                )
                df_summary = pd.DataFrame(extract_datasets_from_h5group(hdf_file["summary"]))

            net_emissions = df_summary["emissions_pos"]
            boiler_design = df_design.loc[
                :, ("industrial_cluster", "Boiler_Industrial_NG_existing")
            ]
            heat_demand = df_operation.loc[
                :, ("energy_balance", "period1", "industrial_cluster", "heat", "demand")
            ]
            waste_in = df_operation.loc[
                :,
                (
                    "energy_balance",
                    "period1",
                    "industrial_cluster",
                    "wasteProcessed",
                    "demand",
                ),
            ]
            boiler_output = df_operation.loc[
                :,
                (
                    "technology_operation",
                    "period1",
                    "industrial_cluster",
                    "Boiler_Industrial_NG_existing",
                ),
            ]

            w2e_design = df_design.loc[:, ("industrial_cluster", "WasteCHP")]
            co2_storage_design = df_design.loc[:, ("storage", "PermanentStorage_CO2_simple")]
            w2e_CaL_design = df_design.loc[:, ("industrial_cluster", "WasteCaL_CCS")]
            json_boiler = Path("./technologies_json/Boiler_Industrial_NG.json")
            info_boiler = json.loads(json_boiler.read_text())
            th_efficiency_boiler = info_boiler["Performance"]["performance"]["out"]["heat"][
                1
            ]
            emission_factor_boiler = info_boiler["Performance"]["emission_factor"]

            # El. production if CCS didn't exist
            json_wasteCHP = Path("./technologies_json/WasteCHP.json")
            info_wasteCHP = json.loads(json_wasteCHP.read_text())
            th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
            el_efficiency = info_wasteCHP["Performance"]["el_efficiency"]
            no_ccs_entry = no_ccs_summary[dh_ratio_str][carbon_tax_str][el_price_str]
            revenues_no_ccs = no_ccs_entry["electricity_revenues"]
            tot_boiler_out_no_ccs = no_ccs_entry["tot_boiler_out"]
            emission_baseline = (
                sum(waste_in * emission_factor)
                + tot_boiler_out_no_ccs / th_efficiency_boiler * emission_factor_boiler
            )

            if w2e_design["size"].iloc[0] > 0 and w2e_design["size_ccs"].iloc[0] > 0:
                w2e_operation = df_operation.loc[
                    :, ("technology_operation", "period1", "industrial_cluster", "WasteCHP")
                ]
                w2e_output = df_operation.loc[
                    :, ("technology_operation", "period1", "industrial_cluster", "WasteCHP")
                ]
                waste_in = w2e_operation["wasteIn_input"]
                # The electricity used by the MEA plant is not sold
                el_out = w2e_operation["electricity_output"] - w2e_operation["electricity_var_input_ccs"]
                emissions_w2e = waste_in * emission_factor
                emissions_boiler = (
                    sum(boiler_output["heat_output"])
                    / th_efficiency_boiler
                    * emission_factor_boiler
                )
                size_ccs = w2e_design["size_ccs"]

                pipeline_cost = df_design_network["capex"].values.flatten()[0]
                storage_cost = co2_storage_design["opex_variable"]

                # Compute and retrieve necessary parameters
                loss_el_revenues = revenues_no_ccs - sum(el_out * el_price)
                extra_gas_usage_boiler = (
                    sum(boiler_output["heat_output"]) - tot_boiler_out_no_ccs
                ) / th_efficiency_boiler
                extra_cost_boiler = extra_gas_usage_boiler * gas_price
                # Relevant KPIs for economics
                type_installed = "MEA"
                capex = w2e_design["capex_ccs"]
                opex_fixed = w2e_design["opex_fixed_ccs"]
                opex_variable = w2e_design["opex_variable_ccs"]
                energy_cost = (
                    loss_el_revenues
                    + extra_cost_boiler
                )
                co2_captured = w2e_operation["CO2captured_var_output_ccs"]
                tot_co2_avoided = emission_baseline - (
                    sum(emissions_w2e - co2_captured) + emissions_boiler
                )
                transport_stor_cost = storage_cost + pipeline_cost
                load_factor_ccs = sum(co2_captured)/(size_ccs*8760)
                fraction_avoided = tot_co2_avoided/emission_baseline

            elif (
                w2e_CaL_design["size"].iloc[0] > 0
                and w2e_CaL_design["size_cal"].iloc[0] > 0
            ):
                json_WasteCaL_CCS = Path("./technologies_json/WasteCaL_CCS.json")
                info_WasteCaL_CCS = json.loads(json_WasteCaL_CCS.read_text())
                lhv_rdf = info_WasteCaL_CCS["Performance"]["LHV_RDF"]
                th_efficiency = info_WasteCaL_CCS["Performance"]["th_efficiency"]
                el_efficiency = info_WasteCaL_CCS["Performance"]["el_efficiency"]
                emission_factor_rdf = info_WasteCaL_CCS["Performance"]["emission_factor_RDF"]
                w2e_cal_operation = df_operation.loc[
                    :,
                    ("technology_operation", "period1", "industrial_cluster", "WasteCaL_CCS"),
                ]
                waste_in = w2e_cal_operation["wasteIn_input"]
                co2_captured_w2e = w2e_cal_operation["CO2captured_output"]
                waste_in_rdf = w2e_cal_operation["wasteInRDF_input"]
                size_ccs = w2e_CaL_design["size_cal"]
                emissions_w2e = (
                    waste_in * emission_factor + waste_in_rdf * emission_factor_rdf
                )
                revenue_el_cal = sum(w2e_cal_operation["el_cal"] * el_price)
                revenue_el_wte = sum(w2e_cal_operation["electricity_output"] * el_price)

                pipeline_cost = df_design_network["capex"].values.flatten()[0]
                storage_cost = co2_storage_design["opex_variable"]
                tot_co2_captured = sum(co2_captured_w2e)
                tot_co2_avoided = sum(waste_in * emission_factor) - (
                    sum(emissions_w2e - co2_captured_w2e)
                )

                type_installed = "CaL"
                capex = w2e_CaL_design["capex_tot"]
                opex_fixed = w2e_CaL_design["opex_fixed"]
                opex_variable = w2e_CaL_design["opex_variable"]
                co2_captured = w2e_cal_operation["CO2captured_output"]
                energy_cost = (
                    -revenue_el_cal
                    + sum(w2e_cal_operation["wasteInRDF_input"] * import_price_RDF)

                )
                transport_stor_cost = storage_cost + pipeline_cost
                load_factor_ccs = sum(co2_captured_w2e)/(size_ccs*8760)
                fraction_avoided = tot_co2_avoided/emission_baseline
                extra_gas_usage_boiler = (sum(boiler_output["heat_output"]) - tot_boiler_out_no_ccs
                                         ) / th_efficiency_boiler
                # electricity_output already includes el_cal
                loss_el_revenues = revenues_no_ccs - revenue_el_wte
            else:
                no_ccs_entry = no_ccs_summary[dh_ratio_str][carbon_tax_str][el_price_str]
                revenues_no_ccs = no_ccs_entry["electricity_revenues"]
                tot_boiler_out_no_ccs = no_ccs_entry["tot_boiler_out"]
                type_installed = "none"
                capex = 0
                opex_fixed = 0
                opex_variable = 0
                co2_captured = 0
                tot_co2_avoided = 0
                energy_cost = 0
                transport_stor_cost = 0
                load_factor_ccs = 0
                size_ccs = 0
                fraction_avoided = 0
                extra_gas_usage_boiler = 0
                loss_el_revenues = 0

            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["hourly_co2_captured"] = co2_captured
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["capex_tot"] = capex
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][ "opex_fixed"] = opex_fixed
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][
                "opex_variable"] = opex_variable
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][
                "energy_cost"] = energy_cost
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][
                "transport_stor_cost"] = transport_stor_cost
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][
                "tot_co2_captured"] = (sum(co2_captured) if not isinstance(co2_captured, int) else pd.Series([0]))
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][
                "tot_co2_avoided"] = (
                tot_co2_avoided if not isinstance(co2_captured, int) else pd.Series([0])
            )
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str][
                "cost_of_avoided"] = (
                (capex + opex_fixed + opex_variable + energy_cost + transport_stor_cost)
                / tot_co2_avoided
                if not isinstance(co2_captured, int)
                else pd.Series([0])
            )
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["type_installed"] = type_installed
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["net_emissions"] = net_emissions
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["load_factor_ccs"] = load_factor_ccs
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["size_ccs"] = size_ccs
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["fraction_avoided"] = fraction_avoided
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["extra_gas_usage_boiler"] = extra_gas_usage_boiler
            results_summary[dh_ratio_str][carbon_tax_str][el_price_str]["loss_el_revenues"] = loss_el_revenues


# Initialize dictionaries to store the final matrices, keyed by dh
cost_matrix = {}
type_matrix = {}

for dh in explored_dh_ratio_str:
    # 1. Initialize the list to collect rows *for the current dh*
    dh_data_rows = []

    # 2. Loop through results and collect rows
    for ct in explored_carbon_tax_str:
        for ep in explored_el_price_str:
            entry = results_summary[f"dh_{dh}"][f"ctax_{ct}"][f"el_price_{ep}"]
            dh_data_rows.append(
                {
                    "carbon_tax": ct,
                    "el_price": ep,
                    "cost_of_avoided": entry["cost_of_avoided"].iloc[0],
                    "type_installed": entry["type_installed"],
                }
            )

    # 3. Create a DataFrame from the current list of rows
    df = pd.DataFrame(dh_data_rows)  # Use the list of dicts for the current dh

    # Convert columns (as before)
    df["carbon_tax"] = pd.to_numeric(df["carbon_tax"])
    df["el_price"] = pd.to_numeric(df["el_price"])

    # 4. Pivot and store the results in the final dictionaries
    cost_matrix_temporary = df.pivot(
        index="el_price", columns="carbon_tax", values="cost_of_avoided"
    )
    type_matrix_temporary = df.pivot(
        index="el_price", columns="carbon_tax", values="type_installed"
    )

    # 5. Sort indices and store them in the result dictionaries
    cost_matrix[f"dh_{dh}"] = cost_matrix_temporary.sort_index(
        ascending=False
    ).sort_index(axis=1)
    type_matrix[f"dh_{dh}"] = type_matrix_temporary.sort_index(
        ascending=False
    ).sort_index(axis=1)

types = ["none", "MEA", "CaL"]


# --- PLOT SECONDARY VARIABLES ---

# --- 1. DATA PREPARATION ---
results_data = {
    "rev_no_ccs": {},
    "boiler_out_no_ccs": {},
    "boiler_em_no_ccs": {},
    "net_em_ccs": {},
    "load_factor_ccs": {},
    "size_ccs": {},
    "fraction_avoided_ccs": {},
    "extra_gas_ccs": {},
    "loss_el_revenues_ccs": {},
}

for dh in explored_dh_ratio_str:
    dh_ratio_key = f"dh_{dh}"
    rows_no_ccs = []
    rows_ccs = []

    for ct in explored_carbon_tax_str:
        ct_key = f"ctax_{ct}"
        for ep in explored_el_price_str:
            ep_key = f"el_price_{ep}"

            # --- No-CCS Data Extraction ---
            d_no = no_ccs_summary[dh_ratio_key][ct_key][ep_key]
            rev_meur = d_no["electricity_revenues"] / 1e6
            b_em = (d_no["tot_boiler_out"] / th_efficiency_boiler * emission_factor_boiler / 1000)

            rows_no_ccs.append({
                "ct": int(ct), "ep": int(ep),
                "rev": rev_meur, "out": d_no["tot_boiler_out"], "em": b_em
            })

            # --- CCS Data Extraction ---
            d_ccs = results_summary[dh_ratio_key][ct_key][ep_key]

            def get_val(item):
                return float(item.iloc[0]) if hasattr(item, 'iloc') else float(item)

            em_val  = get_val(d_ccs['net_emissions']) / 1000
            lf_val  = get_val(d_ccs['load_factor_ccs'])
            sz_val  = get_val(d_ccs['size_ccs'])
            fa_val  = get_val(d_ccs['fraction_avoided'])
            gas_val = get_val(d_ccs['extra_gas_usage_boiler']) / 1e3
            lel_val = get_val(d_ccs['loss_el_revenues']) / 1e6

            rows_ccs.append({
                "ct": int(ct), "ep": int(ep),
                "net_em": em_val, "lf": lf_val, "sz": sz_val, "fa": fa_val,
                "gas": gas_val, "lel": lel_val,
            })

    def pivot_data(rows, val_col):
        df_temp = pd.DataFrame(rows)
        return df_temp.pivot(index="ep", columns="ct", values=val_col).sort_index(ascending=False).astype(float)

    results_data["rev_no_ccs"][dh_ratio_key]            = pivot_data(rows_no_ccs, "rev")
    results_data["boiler_out_no_ccs"][dh_ratio_key]     = pivot_data(rows_no_ccs, "out")
    results_data["boiler_em_no_ccs"][dh_ratio_key]      = pivot_data(rows_no_ccs, "em")
    results_data["net_em_ccs"][dh_ratio_key]            = pivot_data(rows_ccs, "net_em")
    results_data["load_factor_ccs"][dh_ratio_key]       = pivot_data(rows_ccs, "lf")
    results_data["size_ccs"][dh_ratio_key]              = pivot_data(rows_ccs, "sz")
    results_data["fraction_avoided_ccs"][dh_ratio_key]  = pivot_data(rows_ccs, "fa")
    results_data["extra_gas_ccs"][dh_ratio_key]         = pivot_data(rows_ccs, "gas")
    results_data["loss_el_revenues_ccs"][dh_ratio_key]  = pivot_data(rows_ccs, "lel")


# --- 2. STANDARDIZED PLOTTING FUNCTION ---
def plot_heatmap(df, label, filename, cmap, is_pct=False, zero_color="lightgrey"):
    setup_matplotlib_for_paper("single")
    fig, ax = plt.subplots(layout="constrained")

    data = df.to_numpy()
    n_rows, n_cols = data.shape

    nonzero = data[data != 0]
    curr_vmin = np.nanmin(nonzero) if len(nonzero) > 0 else 0
    curr_vmax = np.nanmax(nonzero) if len(nonzero) > 0 else 1
    norm = plt.Normalize(vmin=curr_vmin, vmax=curr_vmax)
    cmap_obj = plt.get_cmap(cmap)

    for i in range(n_rows):
        for j in range(n_cols):
            val = df.iloc[i, j]

            if val == 0:
                facecolor = zero_color
                text_color = "black"
            else:
                facecolor = cmap_obj(norm(val))
                rel_val = (val - curr_vmin) / (curr_vmax - curr_vmin) if (curr_vmax - curr_vmin) != 0 else 0
                text_color = "white" if rel_val > 0.6 else "black"

            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )

            txt = f"{val:.1%}" if is_pct else f"{val:.1f}"
            ax.text(
                j + 0.5, i + 0.5, txt,
                ha="center", va="center",
                color=text_color,
                fontsize=rcParams["axes.labelsize"] - 2,
                fontweight="bold",
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([x + 0.5 for x in range(n_cols)])
    ax.set_yticks([y + 0.5 for y in range(n_rows)])
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlabel(r"Carbon tax [€/tCO$_2$]")
    ax.set_ylabel("Electricity price [€/MWh]")

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=label)

    if is_pct:
        from matplotlib.ticker import FuncFormatter
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    save_figure_for_paper(fig, filename, figures_path)

# --- COMBINED 2x2 CCS METRICS FIGURE ---
def plot_combined_heatmap(dfs_labels, filename, zero_color="lightgrey", path=figures_path):
    """
    Plot a grid of heatmaps with 2 columns.
    dfs_labels: list of (df, label, cmap, is_pct) tuples, 2 or 4 entries.
    """
    fig_width_in, fig_height_in = setup_matplotlib_for_paper("double")  # wider figure for 2 columns
    n_rows_fig = len(dfs_labels) // 2
    fig, axes = plt.subplots(
        n_rows_fig, 2, figsize=(fig_width_in, fig_height_in * n_rows_fig / 2), layout="constrained", squeeze=False
    )
    for ax, (df, label, cmap, is_pct) in zip(axes.flat, dfs_labels):
        data = df.to_numpy()
        n_rows, n_cols = data.shape

        nonzero = data[data != 0]
        curr_vmin = np.nanmin(nonzero) if len(nonzero) > 0 else 0
        curr_vmax = np.nanmax(nonzero) if len(nonzero) > 0 else 1
        norm = plt.Normalize(vmin=curr_vmin, vmax=curr_vmax)
        cmap_obj = plt.get_cmap(cmap)

        for i in range(n_rows):
            for j in range(n_cols):
                val = df.iloc[i, j]

                if val == 0:
                    facecolor = zero_color
                    text_color = "black"
                else:
                    facecolor = cmap_obj(norm(val))
                    rel_val = (val - curr_vmin) / (curr_vmax - curr_vmin) if (curr_vmax - curr_vmin) != 0 else 0
                    text_color = "white" if rel_val > 0.6 else "black"

                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1,
                        facecolor=facecolor,
                        edgecolor="black",
                        linewidth=0.8,
                    )
                )

                txt = f"{val:.1%}" if is_pct else f"{val:.1f}"
                txt = txt.replace("-0.0", "0.0") if txt in ("-0.0", "-0.0%") else txt
                ax.text(
                    j + 0.5, i + 0.5, txt,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=rcParams["axes.labelsize"] - 2,
                    fontweight="bold",
                )

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks([x + 0.5 for x in range(n_cols)])
        ax.set_yticks([y + 0.5 for y in range(n_rows)])
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        ax.invert_yaxis()
        ax.set_xlabel(r"Carbon tax [€/tCO$_2$]")
        ax.set_ylabel("Electricity price [€/MWh]")

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label=label)

        if is_pct:
            from matplotlib.ticker import FuncFormatter
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

    save_figure_for_paper(fig, filename, path)

# --- TECHNOLOGY SELECTION, FRACTION AVOIDED, SIZE, LOAD FACTOR: COMBINED 2x2 FIGURE AND SINGLE FIGURES ---
def plot_tech_selection_overview(type_df, cost_df, heatmaps, dh_key):
    """
    Plot a 2x2 grid with the technology selection and three heatmaps, and each of the four panels as a single figure.
    heatmaps: list of (df, label, cmap, is_pct, suffix) tuples, exactly 3 entries.
    """
    x_label = r"Carbon tax [€/tCO$_2$]"
    y_label = "Electricity price [€/MWh]"

    # combined figure
    fig_width_in, _ = setup_matplotlib_for_paper("double")
    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_width_in * 0.618), layout="constrained")

    draw_tech_selection(fig, axes[0, 0], type_df, cost_df, types)
    for ax, (df, label, cmap, is_pct, suffix) in zip(axes.flat[1:], heatmaps):
        draw_heatmap(fig, ax, df, label, cmap, is_pct)

    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)

    save_figure_for_paper(fig, f"wte_tech_selection_overview_{dh_key}", figures_path_tech_selection)

    # single figures
    setup_matplotlib_for_paper("single")
    fig, ax = plt.subplots(layout="constrained")
    draw_tech_selection(fig, ax, type_df, cost_df, types)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    save_figure_for_paper(fig, f"wte_tech_selection_{dh_key}", figures_path_tech_selection)

    for df, label, cmap, is_pct, suffix in heatmaps:
        fig, ax = plt.subplots(layout="constrained")
        draw_heatmap(fig, ax, df, label, cmap, is_pct)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        save_figure_for_paper(fig, f"wte_{suffix}_{dh_key}", figures_path_tech_selection)


# --- 3. EXECUTION LOOP ---
for dh in explored_dh_ratio_str:
    dh_key = f"dh_{dh}"

    if dh_key not in results_data["rev_no_ccs"]:
        continue

    metrics = [
        (results_data["rev_no_ccs"][dh_key],              "Revenues [M€/y]",                 "rev_no_ccs",         "YlGn",    False),
        (results_data["boiler_out_no_ccs"][dh_key]/1000,   "Boiler output [GWh/y]",           "boiler_out",         "OrRd",    False),
        (results_data["boiler_em_no_ccs"][dh_key],        r"Boiler emissions [ktCO$_2$/y]",   "boiler_em",          "OrRd",    False),
        (results_data["net_em_ccs"][dh_key],              r"Net emissions [ktCO$_2$/y]",      "net_emissions",      "RdBu_r",  False),
        (results_data["load_factor_ccs"][dh_key],          "Capture unit load factor [%]",             "ccs_lf",             SEQUENTIAL_CMAPS["load_factor"], True),
        (results_data["size_ccs"][dh_key],                 "Capture unit size [t/h]",                  "ccs_size",           SEQUENTIAL_CMAPS["size"], False),
        (results_data["fraction_avoided_ccs"][dh_key],    r"CO$_2$ avoided [%]",             "fraction_avoided",   SEQUENTIAL_CMAPS["fraction_avoided"], True),
        (results_data["extra_gas_ccs"][dh_key],            "Extra gas usage boiler [GWh/y]",  "extra_gas_boiler",   "Blues",   False),
        (results_data["loss_el_revenues_ccs"][dh_key],     "Loss el. revenues [M€/y]",        "loss_el_revenues",   "OrRd",    False),
    ]

    for df, label, suffix, cmap, is_pct in metrics:
        full_name = f"wte_secondary_{suffix}_{dh_key}"
        plot_heatmap(df, label, full_name, cmap, is_pct)

    # NEW: combined 2x2 CCS figure
    combined_metrics = [
        (results_data["size_ccs"][dh_key],              "Capture unit size [t/h]",                 SEQUENTIAL_CMAPS["size"], False),
        (results_data["load_factor_ccs"][dh_key],        "Capture unit load factor [%]",            SEQUENTIAL_CMAPS["load_factor"], True),
        (results_data["loss_el_revenues_ccs"][dh_key],   "Loss el. revenues [M€/y]",       "OrRd",    False),
        (results_data["extra_gas_ccs"][dh_key],          "Extra gas usage boiler [GWh/y]", "Blues",   False),
    ]

    plot_combined_heatmap(
        combined_metrics,
        filename=f"wte_ccs_combined_{dh_key}",
    )

    # energy penalty of the capture: lost electricity revenues and extra gas usage of the boiler
    plot_combined_heatmap(
        [
            (results_data["loss_el_revenues_ccs"][dh_key], "Loss el. revenues [M€/y]",       "OrRd",  False),
            (results_data["extra_gas_ccs"][dh_key],        "Extra gas usage boiler [GWh/y]", "Blues", False),
        ],
        filename=f"wte_energy_penalty_{dh_key}",
        path=figures_path_tech_selection,
    )

    # technology selection with fraction avoided, size and load factor
    plot_tech_selection_overview(
        type_matrix[dh_key],
        cost_matrix[dh_key],
        [
            (results_data["fraction_avoided_ccs"][dh_key], r"CO$_2$ avoided [%]", SEQUENTIAL_CMAPS["fraction_avoided"], True,  "fraction_avoided"),
            (results_data["size_ccs"][dh_key],             "Capture unit size [t/h]",      SEQUENTIAL_CMAPS["size"],             False, "ccs_size"),
            (results_data["load_factor_ccs"][dh_key],      "Capture unit load factor [%]", SEQUENTIAL_CMAPS["load_factor"],      True,  "ccs_lf"),
        ],
        dh_key,
    )


plt.show()
print("All secondary plots (including Net Emissions and CCS Size) have been saved.")