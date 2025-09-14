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
from matplotlib import rcParams
import warnings
from utilities.process_results import save_figure_for_paper, print_h5_structure


colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']


## -----------------  Electricity price --------------------------

explored_el_price = [25, 50, 75, 100,125]
rolling_av_hours = 1

path_processed_data = Path("./dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
av_el_price = data["el_price_itNord"].mean()
electricity_price_norm = data["el_price_itNord"]/av_el_price

json_WasteCaL_CCS = Path("./technologies_json/WasteCaL_CCS.json")
info_WasteCaL_CCS = json.loads(json_WasteCaL_CCS.read_text())
lhv = info_WasteCaL_CCS["Performance"]["LHV"]
lhv_rdf = info_WasteCaL_CCS["Performance"]["LHV_RDF"]
th_efficiency = info_WasteCaL_CCS["Performance"]["th_efficiency"]
el_efficiency = info_WasteCaL_CCS["Performance"]["el_efficiency"]
emission_factor = info_WasteCaL_CCS["Performance"]["emission_factor"]
emission_factor_rdf = info_WasteCaL_CCS["Performance"]["emission_factor_RDF"]
ccr = info_WasteCaL_CCS["Performance"]["capture_rate"]


num_cases = len(explored_el_price)
raw_results_path = Path("./raw_results/CaL")
# Get all directories that contain 'el_price' in the name
el_price_dirs = [d for d in raw_results_path.iterdir()
                 if d.is_dir() and "el_price" in d.name]

# Sort directories by name
dir_results_sorted = sorted(el_price_dirs)

# Get the most recent ones
file_names = [d.name for d in dir_results_sorted[-num_cases:]]
explored_el_price_str = [str(r) for r in explored_el_price]

results_summary = {}
for i in range(0,len(file_names)):
    file_path = raw_results_path / f"{file_names[i]}/optimization_results.h5"

    # Check if each explored_el_price[i] is in file_names[i]
    el_price_str = explored_el_price_str[i]
    results_summary[el_price_str] = {}
    if f"{el_price_str}" in file_names[i]:
        print(f"{el_price_str} found in {file_names[i]}")
    else:
        print(f"{el_price_str} NOT found in {file_names[i]}")

    with h5py.File(file_path, 'r') as hdf_file:
        df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
        df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))
    print(df_operation)

    w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCaL_CCS')]
    boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG_existing')]
    w2e_operation = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCaL_CCS')]
    boiler_operation = df_operation.loc[:,
                    ('technology_operation', 'period1', 'industrial_cluster', 'Boiler_Industrial_NG_existing')]
    heat_demand = df_operation.loc[:, ('energy_balance', 'period1', 'industrial_cluster','heat', 'demand')]

    waste_processed_out = w2e_operation['wasteProcessed_output']
    waste_in_rdf = w2e_operation['wasteInRDF_input']
    heat_out = w2e_operation['heat_output']
    el_out = w2e_operation['electricity_output']
    waste_out = w2e_operation['wasteProcessed_output']
    waste_in = w2e_operation['wasteIn_input']
    co2_captured_w2e = w2e_operation['CO2captured_output']
    emissions_w2e = waste_in * emission_factor +  waste_in_rdf * emission_factor_rdf
    fraction_co2_captured = sum(co2_captured_w2e) / sum(emissions_w2e)
    size_cal = w2e_design["size_cal"]
    fraction_size_cal = size_cal/ (max(emissions_w2e)*ccr)
    ccs_capacity_factor = sum(co2_captured_w2e)/(size_cal*8760)
    boiler_load_factor = sum(boiler_operation['heat_output'])/ sum(heat_demand)

    # economics
    el_price = electricity_price_norm * explored_el_price[i]
    capex_cal = w2e_design["capex_tot"]
    opex_fixed = w2e_design["opex_fixed"]
    opex_variable = w2e_design["opex_variable"]
    revenue_el_cal = sum(w2e_operation['el_cal']*el_price)


    results_summary[el_price_str]['size_cal'] = size_cal
    results_summary[el_price_str]['fraction_size_cal'] = fraction_size_cal
    results_summary[el_price_str]['ccs_capacity_factor'] = ccs_capacity_factor
    results_summary[el_price_str]['hourly_co2_captured'] = co2_captured_w2e
    results_summary[el_price_str]['hourly_el_tot_wte'] = el_out
    results_summary[el_price_str]['hourly_el_cal_only'] = w2e_operation['el_cal']
    results_summary[el_price_str]['hourly_el_wte_only'] = w2e_operation['el_wte_only']
    results_summary[el_price_str]['hourly_wte_heat_for_el'] = w2e_operation['el_wte_only']/el_efficiency
    results_summary[el_price_str]['hourly_emissions'] = emissions_w2e

    results_summary[el_price_str]['capex_tot'] = capex_cal
    results_summary[el_price_str]['opex_fixed'] = opex_fixed
    results_summary[el_price_str]['opex_variable'] = opex_variable
    results_summary[el_price_str]['revenue_el_cal'] = revenue_el_cal
    results_summary[el_price_str]['tot_co2_avoided'] = sum(waste_in*emission_factor)-(sum(emissions_w2e-co2_captured_w2e))
    results_summary[el_price_str]['tot_co2_captured'] = sum(co2_captured_w2e)

    results_summary['hourly_boiler_heat_out'] = boiler_operation['heat_output']
    results_summary['boiler_load_factor'] = boiler_load_factor
    results_summary['hourly_heat_demand'] = heat_demand
    results_summary['hourly_heat_prod_wte'] = heat_out
    results_summary['hourly_wasteProcessed'] = waste_processed_out





# Extract fraction_size_cal values (convert pandas series to scalar)
el_prices = []
fraction_size_cal_values = []

for el_price in explored_el_price_str:
    size_series = results_summary[el_price]['fraction_size_cal']
    if hasattr(size_series, "iloc"):  # ensure it's a scalar
        size_val = float(size_series.iloc[0])
    else:
        size_val = float(size_series)
    el_prices.append(float(el_price))
    fraction_size_cal_values.append(size_val)

# Use exactly your defined colors
colors = batlow_colors[:len(el_prices)]

plt.figure(figsize=(7,5))

# Scatter plot with batlow colors
for i, (x, y) in enumerate(zip(el_prices, fraction_size_cal_values)):
    plt.scatter(x, y, color=colors[i], s=100, edgecolor=colors[i], zorder=3)

# Connect points with a neutral line
plt.plot(el_prices, fraction_size_cal_values, linestyle="--", color="gray", alpha=0.6, zorder=2)

plt.xlabel("Electricity Price [€/MWh]")
plt.ylabel("Fraction CO2 treated")
plt.title("Fraction CaL Size vs Electricity Price")

plt.show()


el_prices = []
capacity_factor_cal_values = []

for el_price in explored_el_price_str:
    size_series = results_summary[el_price]['ccs_capacity_factor']
    if hasattr(size_series, "iloc"):  # ensure it's a scalar
        size_val = float(size_series.iloc[0])
    else:
        size_val = float(size_series)
    el_prices.append(float(el_price))
    capacity_factor_cal_values.append(size_val)

# Use exactly your defined colors
colors = batlow_colors[:len(el_prices)]

plt.figure(figsize=(7,5))

# Scatter plot with batlow colors
for i, (x, y) in enumerate(zip(el_prices, capacity_factor_cal_values)):
    plt.scatter(x, y, color=colors[i], s=100, edgecolor=colors[i], zorder=3)

# Connect points with a neutral line
plt.plot(el_prices, capacity_factor_cal_values, linestyle="--", color="gray", alpha=0.6, zorder=2)

plt.xlabel("Electricity Price [€/MWh]")
plt.ylabel("CaL load factor [-]")
plt.title("Load factor CaL vs Electricity Price")

plt.show()

## Plot the economics

capex_tot = []
opex_fixed = []
opex_variable = []
revenue_el_cal = []
tot_co2_avoided = []
tot_co2_captured = []
abatement_cost = []
capture_cost = []


economics = {
    "capex_tot": capex_tot,
    "opex_fixed": opex_fixed,
    "opex_variable": opex_variable,
    "revenue_el_cal": revenue_el_cal,
    "tot_co2_avoided": tot_co2_avoided,
    "tot_co2_captured": tot_co2_captured,
}

for el_price in explored_el_price_str:
    values = {}
    tot_co2_captured = results_summary[el_price]["tot_co2_captured"]
    # Collect each parameter first
    for economic_param, storage_list in economics.items():
        val = results_summary[el_price][economic_param]/tot_co2_captured
        storage_list.append(val)
        values[economic_param] = val

    # Compute abatement cost
    if results_summary[el_price_str]['tot_co2_avoided'] > 0:
        capture = (
            values["capex_tot"]
            + values["opex_fixed"]
            + values["opex_variable"]
            - values["revenue_el_cal"]
        )

    else:
        capture_cost = 0

    capture_cost.append(capture)



# Scatter points for capture cost
for i, (x, y) in enumerate(zip(el_prices, capture_cost)):
    plt.scatter(x, y, color=colors[i], marker="s", s=100, edgecolor=colors[i], zorder=3, label="Capture Cost" if i == 0 else "")

# Connect points with a dashed line
plt.plot(el_prices, capture_cost, linestyle="--", color="black", alpha=0.6, zorder=2)

# Labels
plt.xlabel("Electricity Price [€/MWh]")
plt.ylabel("Cost [€/tCO₂]")

# Legend
plt.legend()
plt.show()



