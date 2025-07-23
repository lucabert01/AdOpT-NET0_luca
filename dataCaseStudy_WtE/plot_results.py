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



file_path = Path(__file__).parent.parent/"userData/20250723201601-1/optimization_results.h5"
file_path_results = r"C:\Users\0954659\OneDrive - Universiteit Utrecht\Documents\PhD Luca\Papers\Geological CO2 storage\Paper\Figures"


print_h5_tree(file_path)

with h5py.File(file_path, 'r') as hdf_file:
    df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
    df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))

print(df_operation)

w2e_output_df = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteCHP')]

heat_out = w2e_output_df['heat_output']
el_out = w2e_output_df['electricity_output']
waste_out = w2e_output_df['wasteProcessed_output']
waste_in = w2e_output_df['wasteFuel_input']
# co2_captured_w2e = w2e_output_df['CO2captured_var_output_ccs']



# # Printing the values
# print("Pump Size:", size_pump)
# print("Cement CCS Size:", size_ccs_cement)
# print("Waste to Energy CCS Size:", size_ccs_w2e)
# path_plot = Path(__file__).parent




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
figsize = (10, 6)
colors = []
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C', '#D491B8','#012E4D']


# Plot emission cement and emission WtE
fig1 = plt.figure(figsize=figsize)
plt.plot(heat_out, color=batlow_colors[1], linewidth=2, label='Heat')
plt.plot(el_out, color=batlow_colors[2], linewidth=2, label='Electricity')
plt.xlabel('Time [h]')
plt.ylabel('Energy [MW]')
plt.ylim(0, 250)
# plt.title('Quality of the pump fit')
plt.legend()
plt.tight_layout()
# save_figure_for_paper(fig1, "emissionsCementWtE_appendix", path_plot)
plt.show()


# Plotting power pump
file_path = Path(__file__).parent/ "pump_coefficients.json"

# Read the data from the JSON file
with open(file_path, "r") as file:
    data_loaded = json.load(file)

# Access the data
a = data_loaded["a"]
b = data_loaded["b"]
p_pump_out_min = data_loaded["p_pump_out_min"]
p_loss_offshore = data_loaded["p_loss_offshore"]
nu = data_loaded["nu"]
eta_pump = data_loaded["eta"]
p_pump_in = data_loaded["p_pump_in"]
p_pump_out = whp + p_loss_offshore
pump_unfitted_power = tot_co2_captured * nu * (p_pump_out - p_pump_in) / eta_pump * 0.1 / 3.6  # power in MWh/day
ratio_fit_pump = power_pump/pump_unfitted_power
fixed_pump_power = tot_co2_captured * nu * (93.563 - p_pump_in) / eta_pump * 0.1 / 3.6
pump_ratio = power_pump/fixed_pump_power
specific_power_pump = power_pump/tot_co2_captured*1000 # in kWh/tCO2


# fig3 = plt.figure(figsize=(10, 6))
# plt.plot(days/365, specific_power_pump, color='#66B2A5', linewidth=2)
# #plt.xlabel('Time [y]')
# plt.ylabel('Pump consumption [kWh/tCO$_2$]')
# plt.ylim(0, max(specific_power_pump)*1.1)
# # plt.title('Impact of bhp variations on the pump power consumption')
# plt.tight_layout()
# plt.show(block=False)
# plt.xlim(0, 1800/365)
# save_figure_for_paper(fig3, "power_pump", path_plot)


# Quality of fit
fig2 = plt.figure(figsize=(10, 6))
plt.plot(days/365, ratio_fit_pump, color='#66B2A5', linewidth=2)
plt.axhline(y=1.0, color="black", linestyle='--', linewidth=1, label='Perfect fit')
plt.xlabel('Time [y]')
plt.ylabel('Ratio')
plt.ylim(0.6, max(ratio_fit_pump) * 1.1)
# plt.title('Quality of the pump fit')
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.xlim(0, 1800/365)
save_figure_for_paper(fig2, "fit_pump", path_plot)

# Parity plot: Predicted vs Actual

# fig4 = plt.figure(figsize=(10, 6))
# plt.scatter(pump_unfitted_power, power_pump, color=batlow_colors[2], s=30, alpha=0.7, label='Data Points')
# plt.plot([0, max(pump_unfitted_power)], [0, max(pump_unfitted_power)], color="black", linestyle='--', linewidth=1.5, label='Perfect Fit')
# plt.plot([0, max(pump_unfitted_power)], [0, max(pump_unfitted_power) * 1.1], color='gray', linestyle='--', linewidth=1)
# plt.plot([0, max(pump_unfitted_power)], [0, max(pump_unfitted_power) * 0.9], color='gray', linestyle='--', linewidth=1)
#
# # Text annotations for uncertainty lines
# plt.text(max(pump_unfitted_power) * 0.8, max(pump_unfitted_power) * 1.1, '+10%', color='gray', fontsize=12, ha='left')
# plt.text(max(pump_unfitted_power) * 0.8, max(pump_unfitted_power) * 0.9, '-10%', color='gray', fontsize=12, ha='left')
# plt.xlabel('Actual Pump Power Consumption')
# plt.ylabel('Predicted Pump Power Consumption')
# plt.title('Parity Plot: Predicted vs Actual Pump Power Consumption with ±10% Uncertainty')
# plt.legend()
# plt.tight_layout()
# save_figure_for_paper(fig4, "parity_pump_fit", path_plot)


# plt.show()



