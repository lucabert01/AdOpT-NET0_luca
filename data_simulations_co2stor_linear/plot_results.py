import h5py
from pathlib import Path
from adopt_net0.result_management.read_results import (
    print_h5_tree,
    extract_dataset_from_h5,
    extract_datasets_from_h5group,
)
import pandas as pd
import matplotlib.pyplot as plt


file_path = Path(__file__).parent.parent/"userData/20241008103619-1/optimization_results.h5"


print_h5_tree(file_path)

with h5py.File(file_path, 'r') as hdf_file:
    df_case = extract_datasets_from_h5group(hdf_file["operation"])

print(df_case)


cement_output_df = df_case[('technology_operation', 'period1', 'storage', 'CementEmitter')]
co2stor_results_df = df_case[("technology_operation", "period1","storage", "PermanentStorage_CO2_detailed")]

emission_cement = cement_output_df['cement_output']
co2_captured = cement_output_df['CO2captured_var_output_ccs']
bhp = co2stor_results_df['bhp']
storage_level = co2stor_results_df['storage_level']
power_pump = co2stor_results_df['electricity_input']


# Create a range of days
days = range(0, len(cement_output_df) )

# Plotting CO2 emissions and capture
plt.figure(figsize=(10, 6))
emitted_emissions = emission_cement - co2_captured
# Fill the area under the captured CO2 curve (Light Pink from Crameri Batlow)
plt.fill_between(days, 0, co2_captured, color='#F6C6D6', alpha=0.7, label='Captured CO2')
# Fill the area between captured and emitted CO2 (Dark Blue from Crameri Batlow)
plt.fill_between(days, co2_captured, emitted_emissions, color='#012E4D', alpha=0.7, label='Emitted CO2')
plt.xlabel('Time [day]')
plt.ylabel('CO2 emissions [t/day]')
plt.ylim(0, max(emission_cement) * 1.1)
plt.xlim(0, max(days))
plt.legend()
plt.tight_layout()
plt.show()



# Plotting BHP
plt.figure(figsize=(10, 6))
plt.plot(days, bhp, color='#C1D3E0', linewidth=2, label='Bottomhole pressure')
plt.xlabel('Time [day]')
plt.ylabel('Bottomhole pressure [bar]')
plt.ylim(0, max(bhp) * 1.1)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting power pump
plt.figure(figsize=(10, 6))
plt.plot(days, power_pump, color='#66B2A5', linewidth=2, label='Energy use pump')
plt.xlabel('Time [day]')
plt.ylabel('Power [MW]')
plt.ylim(0, max(power_pump) * 1.1)
plt.legend()
plt.tight_layout()
plt.show()


