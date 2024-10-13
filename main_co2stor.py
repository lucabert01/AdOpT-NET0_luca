# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import pyomo.environ as pyo
import scipy.io as sci

# Function for later
def prepare_data_series(array, type, end_period):
    array = array[:end_period].flatten()
    full_series = np.zeros(8760)
    full_series[:len(array)] = array
    full_series = pd.DataFrame(full_series, columns=[type])
    return full_series



# Specify the path to your input data
path = Path("./detailedCO2storage_test2")

# Create template files (comment these lines if already defined)
adopt.create_optimization_templates(path)

# Load json template
with open(path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = ["storage"]
# Carriers:
topology["carriers"] = ["electricity", "CO2captured", "cement", "heat", "waste"]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

end_period = 15*30


# Load json template
with open(path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
# Change objective
configuration["optimization"]["objective"]["value"] = "costs"
# Set MILP gap
configuration["solveroptions"]["mipgap"]["value"] = 0.02
# Save json template
with open(path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

adopt.create_input_data_folder_template(path)

# Add technologies
with open(path / "period1" / "node_data" / "storage" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["PermanentStorage_CO2_detailed", "CementEmitter", "WasteToEnergyEmitter"]

with open(path / "period1" / "node_data" / "storage" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Copy over technology files
adopt.copy_technology_data(path)

# Load the .mat file for ROM model of saline aquifer
general_performance_data_path = Path(__file__).parent
aquifer_performance_data_path = (
        general_performance_data_path
        / "adopt_net0/data/technology_data/Sink/SalineAquifer_data/matrices_for_ROM.mat"
)


co2stor_data = sci.loadmat(aquifer_performance_data_path)
inj_rate = co2stor_data["uNew"]
inj_rate = inj_rate[:end_period, 0]
conversion_factor = 550*3.6
tot_inj_rate = np.zeros(8760)
tot_inj_rate[:len(inj_rate)] = inj_rate*conversion_factor
tot_inj_rate = pd.DataFrame(tot_inj_rate, columns=['Import limit'])

# Load and prepare data series
data_path = Path(__file__).parent/"data_simulations_co2stor_linear"
cement_emissions = np.load(data_path/"fullprofile_emissions_cement.npy", allow_pickle=True)
cement_emissions = prepare_data_series(cement_emissions, "Demand", end_period)
waste_emissions = np.load(data_path/"fullprofile_emissions_w2e.npy", allow_pickle=True)
waste_emissions = prepare_data_series(waste_emissions, "Demand", end_period)
carbon_price_timeseries = np.load(data_path/'fullprofile_co2_price.npy', allow_pickle=True)
carbon_price_timeseries = prepare_data_series(carbon_price_timeseries, "price", end_period)

# Set import limits/cost
adopt.fill_carrier_data(path, value_or_data=0, columns=['Import limit'], carriers=['CO2captured'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=-1500, columns=['Import price'], carriers=['CO2captured'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=5000, columns=['Import limit'], carriers=['electricity'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=20000, columns=['Import limit'], carriers=['heat'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=30, columns=['Import price'], carriers=['electricity'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=10, columns=['Import price'], carriers=['heat'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=cement_emissions/3, columns=['Demand'], carriers=['cement'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=waste_emissions/2, columns=['Demand'], carriers=['waste'], nodes=['storage'])

carbon_price = carbon_price_timeseries['price'].values
carbon_cost_path = path / "period1" / "node_data" / "storage" /"CarbonCost.csv"
carbon_cost_template = pd.read_csv(carbon_cost_path, sep=';', index_col=0, header=0)
carbon_cost_template['price'] = carbon_price
carbon_cost_template = carbon_cost_template.reset_index()
carbon_cost_template.to_csv(carbon_cost_path, sep=';', index=False)

# Construct and solve the model
m = adopt.ModelHub()
m.read_data(path, start_period=0, end_period=end_period)
m.construct_model()
m.construct_balances()

# # Force output of CementEmitter
# model = m.model[m.info_solving_algorithms["aggregation_model"]].periods["period1"]
# set_t_full = model.set_t_full
# emission_profile_cement = np.ones((8760,1))*50
# b_tec = m.model[m.info_solving_algorithms[
#     "aggregation_model"]].periods["period1"].node_blocks["storage"].tech_blocks_active[
#     "CementEmitter"]
#
# def init_cement_emissions(const,t):
#     return b_tec.var_output[t, "cement"] == emission_profile_cement[t]
# b_tec.const_emission_cement = pyo.Constraint(set_t_full, rule=init_cement_emissions)

m.solve()
# m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
#     "PermanentStorage_CO2_detailed"].var_distance.pprint()
# m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
#     "PermanentStorage_CO2_detailed"].var_bhp.pprint()
# m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
#     "PermanentStorage_CO2_detailed"].var_d_min.pprint()


compare_co2_stored = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_detailed"].var_input[t, "CO2captured"].value for t in range(1,end_period))
compare_el_storage = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_detailed"].var_input[t, "electricity"].value for t in range(1,end_period))
compare_heat_ccs = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_input_ccs[t, "heat"].value for t in range(1,end_period))
compare_el_ccs = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_input_ccs[t, "electricity"].value for t in range(1,end_period))
compare_co2_ccs = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_output_ccs[t, "CO2captured"].value for t in range(1,end_period))
compare_size_ccs = m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_size_ccs.value
compare_capex_ccs = m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_capex_ccs.value

print("Some results:")
print(f"CO2 Stored: {compare_co2_stored:.2f}")
print(f"Electricity for Storage: {compare_el_storage:.2f}")
print(f"Heat for CCS: {compare_heat_ccs:.2f}")
print(f"Electricity for CCS: {compare_el_ccs:.2f}")
print(f"CO2 captured CCS: {compare_co2_ccs:.2f}")
print(f"CapEx for CCS: {compare_capex_ccs:.2f}")
print(f"Size CCS: {compare_size_ccs:.2f}")


