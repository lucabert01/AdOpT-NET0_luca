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

    # Apply font settings directly to the figure (local only, doesn't affect global state)
    rcParams.update({'font.size': 16, 'font.family': 'Arial'})

    # Ensure high-quality rendering
    fig.set_dpi(300)  # Higher DPI for better quality
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})  # Embed fonts in vector formats

    # Save in PDF and JPG formats
    fig.savefig(file_path_results / f"{filename}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(file_path_results / f"{filename}.jpg", format='jpeg', dpi=300, bbox_inches='tight')



file_path = Path(__file__).parent.parent/"userData/FullRun20241210/optimization_results.h5"


print_h5_tree(file_path)

with h5py.File(file_path, 'r') as hdf_file:
    df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
    df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))

print(df_operation)
rho_co2_surface = 876.5
convert_inj_rate = 1/(rho_co2_surface*3.6)
pmax = 210
cement_output_df = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
w2e_output_df = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteToEnergyEmitter')]
co2stor_results_df = df_operation.loc[:, ("technology_operation", "period1","storage", "PermanentStorage_CO2_detailed")]

emission_cement = cement_output_df['cement_output']
emission_w2e = w2e_output_df['waste_output']
co2_captured_cement = cement_output_df['CO2captured_var_output_ccs']
co2_captured_w2e = w2e_output_df['CO2captured_var_output_ccs']
bhp = co2stor_results_df['bhp']
whp = co2stor_results_df['whp']
average_inj_rate = co2stor_results_df['average_inj_rate']*24/convert_inj_rate
storage_level = co2stor_results_df['storage_level']
power_pump = co2stor_results_df['electricity_input']
size_pump = df_design[('storage','PermanentStorage_CO2_detailed','size_pump')]
size_ccs_cement = df_design[('industrial_cluster','CementEmitter','size_ccs')]
size_ccs_w2e = df_design[('industrial_cluster','WasteToEnergyEmitter','size_ccs')]

emission_tot = emission_w2e + emission_cement
tot_co2_captured = co2_captured_cement +co2_captured_w2e
# Create a range of days
days = np.array(range(0, len(cement_output_df) ))
value_average_inj_rate = np.array([average_inj_rate[i*180+1] for i in range(0,int(len(average_inj_rate)/180))])
print("average_inj_rate:",value_average_inj_rate)

# Printing the values
print("Pump Size:", size_pump)
print("Cement CCS Size:", size_ccs_cement)
print("Waste to Energy CCS Size:", size_ccs_w2e)
path_plot = Path(__file__).parent

# Plotting CO2 emissions and capture
file_path_results = r"C:\Users\0954659\OneDrive - Universiteit Utrecht\Documents\PhD Luca\Papers\Geological CO2 storage\Paper\Figures"


# Calculate the ratios
cement_ratio = co2_captured_cement / emission_cement
w2e_ratio = co2_captured_w2e / emission_w2e

# # Plot the ratios
# plt.figure(figsize=(12, 6))
# plt.plot(cement_ratio, label='Cement: CO2 Captured/Emissions', marker='o', markersize=3, linewidth=0.7)
# plt.plot(w2e_ratio, label='W2E: CO2 Captured/Emissions', marker='x', linestyle='--', markersize=3, linewidth=0.7)
# plt.xlabel('Time')
# plt.ylabel('CO2 Captured/Emissions Ratio')
# plt.title('Comparison of CO2 Captured to Emissions Ratios')
# plt.xlim(0, 1800)
# plt.legend()
#
# plt.figure(figsize=(12, 6))
# fig, ax1 = plt.subplots(figsize=(12, 6))
# line1 = ax1.plot((co2_captured_cement + co2_captured_w2e) / (emission_cement + emission_w2e),
#                  label='CO2 Captured/Total Emissions', color='blue', marker='o', markersize=3, linewidth=0.7)
# ax1.set_xlabel('Time')
# ax1.set_ylabel('CO2 Captured/Total Emissions', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
# ax2 = ax1.twinx()
# line2 = ax2.plot((co2_captured_cement + co2_captured_w2e),
#                  label='Total CO2 Captured', color='green', marker='x', markersize=3, linewidth=0.7, linestyle='--')
# ax2.set_ylabel('Total CO2 Captured', color='green')
# ax2.tick_params(axis='y', labelcolor='green')
# plt.title('Share of Total CO2 Captured')
# ax1.grid(True)
# lines = line1 + line2
# labels = [line.get_label() for line in lines]
# ax1.legend(lines, labels, loc='lower right')
# ax1.set_xlim(0, 1800)
# plt.show()


# Set global font properties
rcParams.update({'font.size': 16, 'font.family': 'Arial'})

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Use plt.subplots to link axes directly to the figure
ax.fill_between(days / 365, 0, tot_co2_captured, color='#D491B8', alpha=0.7, label='Captured CO$_2$')
ax.fill_between(days / 365, tot_co2_captured, emission_tot, color='#012E4D', alpha=0.7, label='Emitted CO$_2$')

# Set labels, limits, and legend
ax.set_ylabel('CO$_2$ emissions [t/day]', fontsize=16)
ax.set_xlabel('Time [years]', fontsize=16)
ax.set_ylim(0, max(emission_tot) * 1.1)
ax.set_xlim(0, max(days / 365))
ax.legend(fontsize=14)

# Adjust layout for clarity
plt.tight_layout()

# Show the plot without blocking
plt.show(block=False)
save_figure_for_paper(fig, "emissions", path_plot)




# Plotting BHP

batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C']
# First plot: Average injection rate
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(days / 365, average_inj_rate, color='#D491B8', linewidth=2, label='Average injection Rate')
ax1.set_ylabel('Av. inj. rate [t/day]')
ax1.set_ylim(max(average_inj_rate) * 0.6, max(average_inj_rate) * 1.1)
#ax1.set_xlabel('Time [y]')
plt.tight_layout()
plt.xlim(0, 1800 / 365)
save_figure_for_paper(fig1, "average_injection_rate", path_plot)

# Second plot: BHP
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(days / 365, bhp, color=batlow_colors[2], linewidth=2, label='BHP')
ax2.axhline(y=pmax, color=batlow_colors[0], linestyle='--', linewidth=1, label='p$_{max}$')
ax2.set_xlabel('Time [y]')
ax2.set_ylabel('Pressure [bar]')
ax2.set_ylim(min(bhp) * 0.88, pmax * 1.04)
ax2.legend(loc='lower right')
plt.tight_layout()
plt.xlim(0, 1800 / 365)
save_figure_for_paper(fig2, "bhp_case_study", path_plot)




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


fig3 = plt.figure(figsize=(10, 6))
plt.plot(days/365, specific_power_pump, color='#66B2A5', linewidth=2)
#plt.xlabel('Time [y]')
plt.ylabel('Pump consumption [kWh/tCO$_2$]')
plt.ylim(0, max(specific_power_pump)*1.1)
# plt.title('Impact of bhp variations on the pump power consumption')
plt.tight_layout()
plt.show(block=False)
plt.xlim(0, 1800/365)
save_figure_for_paper(fig3, "power_pump", path_plot)


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


plt.show()



