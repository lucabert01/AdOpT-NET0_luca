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



def save_figure_for_paper(fig, filename, file_path_results):
    """
    Save a matplotlib figure with settings similar to the provided MATLAB function.

    Parameters:
        fig : matplotlib.figure.Figure
            The figure handle to save.
        filename : str
            The base filename for saving the figure.
        file_path_results : str or Path
            The directory where the figure should be saved.
    """
    from matplotlib import rcParams

    # Convert to Path object if needed
    file_path_results = Path(file_path_results)
    file_path_results.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Set figure size (width x height in inches)
    width_in, height_in = 432 / 72, 288 / 72  # Convert from points (1 pt = 1/72 inch)
    fig.set_size_inches(width_in, height_in)


    # # Save in PDF and JPG formats
    fig.savefig(file_path_results / f"{filename}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(file_path_results / f"{filename}.jpg", format='jpeg', dpi=300, bbox_inches='tight')



file_path = Path(__file__).parent.parent/"userData/20250724191232-1/optimization_results.h5"
file_path_results = r"C:\Users\0954659\OneDrive - Universiteit Utrecht\Documents\PhD Luca\Papers\Linear CCS technologies\Paper\Figures"
json_wasteCHP = Path("./technologies_json/WasteCHP.json")
info_wasteCHP = json.loads(json_wasteCHP.read_text())
lhv = info_wasteCHP["Performance"]["LHV"]
emission_factor = info_wasteCHP["Performance"]["emission_factor"]
json_mea = Path("./technologies_json/MEA_medium.json")
info_mea = json.loads(json_wasteCHP.read_text())
ccr = info_wasteCHP["Performance"]["capture_rate"]

print_h5_tree(file_path)

with h5py.File(file_path, 'r') as hdf_file:
    df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
    df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))

print(df_operation)

w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCHP')]
boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG')]
w2e_output = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCHP')]
boiler_output = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'Boiler_Industrial_NG')]

heat_out = w2e_output['heat_output']
el_out = w2e_output['electricity_output']
waste_out = w2e_output['wasteProcessed_output']
waste_in = w2e_output['wasteFuel_input']
co2_captured_w2e = w2e_output['CO2captured_var_output_ccs']
emissions_total = waste_in*emission_factor
fraction_co2_captured = co2_captured_w2e/emissions_total


# Printing the values
print("WtE CCS Size:", w2e_design["size_ccs"])
path_plot = Path(__file__).parent




# Set global styling for the plots
rcParams.update({
    'font.size': 16,
    'font.family': 'Arial',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Centralized figure size
figsize = (10, 8)
colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']



# sharex=True links the x-axes of the two plots.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)


# --- Top Plot (ax1): Energy Output ---
ax1.plot(heat_out, color=batlow_colors[1], linewidth=2, label='Heat')
ax1.plot(el_out, color=batlow_colors[2], linewidth=2, label='Electricity')
ax1.set_ylabel('Energy [MW]')
ax1.set_ylim(0, 250)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Bottom Plot (ax2): CO2 Capture ---
ax2.plot(fraction_co2_captured * 100, color=batlow_colors[3], linewidth=2, label='Fraction CO2 Captured')
ax2.set_xlabel('Time [h]')
ax2.set_ylabel('Capture Rate [%]')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)


# --- Display Plot ---
# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
plt.show()

## -----------------  DH ratio --------------------------



def plot_results_dh_ratio(explored_dh_ratio):
    num_cases = len(explored_dh_ratio)
    raw_results_path = Path("./raw_results")
    # Get all directories that contain 'dh_ratio' in the name
    dh_ratio_dirs = [d for d in raw_results_path.iterdir()
                     if d.is_dir() and "dh_ratio" in d.name]

    # Sort directories by name (or modify to sort by timestamp if needed)
    dir_results_sorted = sorted(dh_ratio_dirs)

    # Get the most recent ones
    file_names = [d.name for d in dir_results_sorted[-num_cases:]]
    explored_dh_ratio_str = [str(r) for r in explored_dh_ratio]

    results_summary = {}
    for i in len(file_names):
        file_path = Path(__file__).parent.parent / f"userData/{file_names[i]}/optimization_results.h5"
        # Check if each explored_dh_ratio[i] is in file_names[i]
        dh_ratio_str = explored_dh_ratio_str[i]
        if str(dh_ratio_str) in file_names[i]:
            print(f"{dh_ratio_str} found in {file_names[i]}")
        else:
            print(f"{dh_ratio_str} NOT found in {file_names[i]}")

        with h5py.File(file_path, 'r') as hdf_file:
            df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
            df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))

        print(df_operation)

        w2e_design = df_design.loc[:, ('industrial_cluster', 'WasteCHP')]
        boiler_design = df_design.loc[:, ('industrial_cluster', 'Boiler_Industrial_NG')]
        w2e_output = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCHP')]
        boiler_output = df_operation.loc[:,
                        ('technology_operation', 'period1', 'industrial_cluster', 'Boiler_Industrial_NG')]

        heat_out = w2e_output['heat_output']
        el_out = w2e_output['electricity_output']
        waste_out = w2e_output['wasteProcessed_output']
        waste_in = w2e_output['wasteFuel_input']
        co2_captured_w2e = w2e_output['CO2captured_var_output_ccs']
        emissions_total = waste_in * emission_factor
        fraction_co2_captured = co2_captured_w2e / emissions_total
        size_ccs = w2e_design["size_ccs"]
        fraction_size_ccs = size_ccs/ccr/ (max(waste_in)*emission_factor)
        ccs_utilization_rate = fraction_co2_captured/(fraction_size_ccs*ccr)
        heat_supplied_by_boiler = sum(boiler_output['heat_output'])/ (sum(heat_out) + sum(boiler_output['heat_output']))

        results_summary[dh_ratio_str]['size_ccs'] = size_ccs
        results_summary[dh_ratio_str]['fraction_size_ccs'] = fraction_size_ccs
        results_summary[dh_ratio_str]['ccs_utilization_rate'] = ccs_utilization_rate
        results_summary[dh_ratio_str]['heat_supplied_by_boiler'] = heat_supplied_by_boiler


    # Plot
    fraction_size_ccs = [results_summary[dh]['fraction_size_ccs'] for dh in explored_dh_ratio_str]
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fraction_size_ccs*100, color=batlow_colors[4], linewidth=2, label='CCS Size')
    ax.set_xlabel('time [h]')
    ax.set_ylabel('fraction CCS size [%]')  # Adjust if units are known
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()