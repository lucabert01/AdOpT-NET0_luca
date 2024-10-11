import h5py
from pathlib import Path
from adopt_net0.result_management.read_results import (
    print_h5_tree,
    extract_dataset_from_h5,
    extract_datasets_from_h5group,
)
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np


file_path = Path(__file__).parent.parent/"userData/20241011172455-1/optimization_results.h5"


print_h5_tree(file_path)

with h5py.File(file_path, 'r') as hdf_file:
    df_operation = extract_datasets_from_h5group(hdf_file["operation"])
    df_design = extract_datasets_from_h5group(hdf_file["design/nodes/period1/storage"])

print(df_operation)

cement_output_df = df_operation[('technology_operation', 'period1', 'storage', 'CementEmitter')]
w2e_output_df = df_operation[('technology_operation', 'period1', 'storage', 'WasteToEnergyEmitter')]
co2stor_results_df = df_operation[("technology_operation", "period1","storage", "PermanentStorage_CO2_detailed")]

emission_cement = cement_output_df['cement_output']
emission_w2e = w2e_output_df['waste_output']
co2_captured_cement = cement_output_df['CO2captured_var_output_ccs']
co2_captured_w2e = w2e_output_df['CO2captured_var_output_ccs']
bhp = co2stor_results_df['bhp']
whp = co2stor_results_df['whp']
storage_level = co2stor_results_df['storage_level']
power_pump = co2stor_results_df['electricity_input']
size_ccs = df_design[('CementEmitter','size_ccs')]

emission_tot = emission_w2e + emission_cement
tot_co2_captured = co2_captured_cement +co2_captured_w2e
# Create a range of days
days = range(0, len(cement_output_df) )

# Plotting CO2 emissions and capture
plt.figure(figsize=(10, 6))
# Fill the area under the captured CO2 curve (Light Pink from Crameri Batlow)
plt.fill_between(days, 0, tot_co2_captured, color='#F6C6D6', alpha=0.7, label='Captured CO2')
# Fill the area between captured and emitted CO2 (Dark Blue from Crameri Batlow)
plt.fill_between(days, tot_co2_captured, emission_tot, color='#012E4D', alpha=0.7, label='Emitted CO2')
plt.xlabel('Time [day]')
plt.ylabel('CO2 emissions [t/day]')
plt.ylim(0, max(emission_tot) * 1.1)
plt.xlim(0, max(days))
plt.legend()
plt.tight_layout()
plt.show(block=False)



# Plotting BHP
plt.figure(figsize=(10, 6))
plt.plot(days, bhp, color='#782D5D', linewidth=2, label='Bottomhole pressure')
plt.xlabel('Time [day]')
plt.ylabel('Bottomhole pressure [bar]')
plt.ylim(155, max(bhp) * 1.02)
plt.legend()
plt.tight_layout()
plt.show(block=False)

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
plt.plot(days, ratio_fit_pump, color='#294B6C', linewidth=2, label='Fitted pump power over real one')
plt.axhline(y=1.0, color="#DCE391", linestyle='--', linewidth=1, label='Perfect fit')
plt.xlabel('Time [day]')
plt.ylabel('Ratio')
plt.ylim(0.6, max(ratio_fit_pump) * 1.1)
plt.title('Quality of the pump fit')
plt.legend()
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10, 6))
plt.plot(days, power_pump/fixed_pump_power, color='#66B2A5', linewidth=2, label='With model')
plt.axhline(y=1.0, color="#082D48", linestyle='--', linewidth=1, label='Without model')
plt.xlabel('Time [day]')
plt.ylabel('Normalized specific pump consumption')
plt.ylim(0, max(power_pump/fixed_pump_power)*1.1)
plt.title('Impact of bhp variations on the pump power consumption')
plt.legend()
plt.tight_layout()
plt.show(block=False)

plt.show()
