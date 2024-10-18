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



file_path = Path(__file__).parent.parent/"userData/20241018183752-1/optimization_results.h5"


print_h5_tree(file_path)

with h5py.File(file_path, 'r') as hdf_file:
    df_operation = pd.DataFrame(extract_datasets_from_h5group(hdf_file["operation"]))
    df_design = pd.DataFrame(extract_datasets_from_h5group(hdf_file["design/nodes/period1"]))

print(df_operation)

cement_output_df = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'CementEmitter')]
w2e_output_df = df_operation.loc[:, ('technology_operation', 'period1', 'industrial_cluster', 'WasteToEnergyEmitter')]
co2stor_results_df = df_operation.loc[:,("technology_operation", "period1","storage", "PermanentStorage_CO2_detailed")]

emission_cement = cement_output_df['cement_output']
emission_w2e = w2e_output_df['waste_output']
co2_captured_cement = cement_output_df['CO2captured_var_output_ccs']
co2_captured_w2e = w2e_output_df['CO2captured_var_output_ccs']
bhp = co2stor_results_df['bhp']
whp = co2stor_results_df['whp']
average_inj_rate = co2stor_results_df['average_inj_rate']
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
path_plot = Path(__file__).parent.parent.parent/"PhD Luca/Papers/Geological CO2 storage/Paper/Figures"

# Plotting CO2 emissions and capture
plt.figure(figsize=(10, 6))
# Fill the area under the captured CO2 curve (Light Pink from Crameri Batlow)
plt.fill_between(days/365, 0, tot_co2_captured, color='#F6C6D6', alpha=0.7, label='Captured CO2')
# Fill the area between captured and emitted CO2 (Dark Blue from Crameri Batlow)
plt.fill_between(days/365, tot_co2_captured, emission_tot, color='#012E4D', alpha=0.7, label='Emitted CO2')
plt.xlabel('Time [year]')
plt.ylabel('CO2 emissions [t/day]')
plt.ylim(0, max(emission_tot) * 1.1)
plt.xlim(0, max(days/365))
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.savefig(path_plot/"emissions.jpg", format='jpeg', dpi=500)



# Plotting BHP
rho_co2_surface = 505.71
convert_inj_rate = 1/(rho_co2_surface*3.6)
pmax = 175
batlow_colors = ['#222A6A', '#4B708A', '#6FBC7B', '#B1E87E', '#F7D03C']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(days/365, average_inj_rate/convert_inj_rate, color=batlow_colors[1], linewidth=2, label='Injection Rate')
ax1.set_ylabel('Average injection rate [t/day]')  # Replace 'units' with the appropriate unit for injection rate
ax1.legend()
ax1.set_ylim(max(average_inj_rate/convert_inj_rate)*0.6, max(average_inj_rate/convert_inj_rate) * 1.1)
ax2.plot(days/365, bhp, color=batlow_colors[2], linewidth=2, label='Bottomhole pressure')
ax2.axhline(y=pmax, color=batlow_colors[0], linestyle='--', linewidth=1, label='Caprock fracture pressure')
ax2.set_xlabel('Time [year]')
ax2.set_ylabel('Bottomhole pressure [bar]')
ax2.set_ylim(min(bhp)*0.98, pmax * 1.02)
ax2.legend()
plt.tight_layout()
plt.savefig(path_plot/"bhp_case_study.jpg", format='jpeg', dpi=500)



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
fixed_pump_power = pump_unfitted_power[0]


plt.figure(figsize=(10, 6))
plt.plot(days/365, ratio_fit_pump, color='#294B6C', linewidth=2, label='Fitted pump power over real one')
plt.axhline(y=1.0, color="#DCE391", linestyle='--', linewidth=1, label='Perfect fit')
plt.xlabel('Time [year]')
plt.ylabel('Ratio')
plt.ylim(0.6, max(ratio_fit_pump) * 1.1)
plt.title('Quality of the pump fit')
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.savefig(path_plot/"fit_pump.jpg", format='jpeg', dpi=500)

pump_ratio = power_pump/fixed_pump_power
averaged_pump_ratio = [np.mean(pump_ratio[i:i+180]) for i in range(0, len(pump_ratio), 180)]
expanded_pump_ratio = np.repeat(averaged_pump_ratio, 180)


plt.figure(figsize=(10, 6))
plt.plot(days/365, expanded_pump_ratio, color='#66B2A5', linewidth=2, label='With model')
plt.axhline(y=1.0, color="#082D48", linestyle='--', linewidth=1, label='Without model')
plt.xlabel('Time [year]')
plt.ylabel('Normalized specific pump consumption')
plt.ylim(0, max(power_pump/fixed_pump_power)*1.1)
plt.title('Impact of bhp variations on the pump power consumption')
plt.legend()
plt.tight_layout()
plt.show(block=False)
plt.savefig(path_plot/"power_pump.jpg", format='jpeg', dpi=500)

plt.show()
