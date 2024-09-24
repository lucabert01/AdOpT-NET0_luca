# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import pyomo.environ as pyo


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
topology["carriers"] = ["electricity", "CO2captured", "cement", "heat", "gas", "hydrogen"]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)


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
#technologies["new"] = ["PermanentStorage_CO2_detailed", "CementEmitter"]
technologies["new"] = ["PermanentStorage_CO2_simple", "CementEmitter", "GasTurbine_simple_CCS"]

with open(path / "period1" / "node_data" / "storage" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Copy over technology files
adopt.copy_technology_data(path)

# Set import limits/cost
adopt.fill_carrier_data(path, value_or_data=0.0345*0, columns=['Import limit'], carriers=['CO2captured'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=-1500, columns=['Import price'], carriers=['CO2captured'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=50, columns=['Import limit'], carriers=['electricity'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=20000, columns=['Import limit'], carriers=['heat'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=20000, columns=['Import limit'], carriers=['gas'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=480/24, columns=['Demand'], carriers=['cement'], nodes=['storage'])

carbon_price = np.ones(8760)*500
carbon_cost_path = path / "period1" / "node_data" / "storage" /"CarbonCost.csv"
carbon_cost_template = pd.read_csv(carbon_cost_path, sep=';', index_col=0, header=0)
carbon_cost_template['price'] = carbon_price
carbon_cost_template = carbon_cost_template.reset_index()
carbon_cost_template.to_csv(carbon_cost_path, sep=';', index=False)

# Construct and solve the model
m = adopt.ModelHub()
m.read_data(path, start_period=0, end_period=72)
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


t_end = 73
compare_co2_stored = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_simple"].var_input[t, "CO2captured"].value for t in range(1,t_end))
compare_el_storage = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_simple"].var_input[t, "electricity"].value for t in range(1,t_end))
compare_heat_ccs = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_input_ccs[t, "heat"].value for t in range(1,t_end))
compare_el_ccs = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_input_ccs[t, "electricity"].value for t in range(1,t_end))
compare_capex_ccs = m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "CementEmitter"].var_capex_ccs.value

print("Comparison of Values:")
print(f"CO2 Stored (Period 1-73): {compare_co2_stored:.2f}")
print(f"Electricity for Storage (Period 1-73): {compare_el_storage:.2f}")
print(f"Heat for CCS (Period 1-73): {compare_heat_ccs:.2f}")
print(f"Electricity for CCS (Period 1-73): {compare_el_ccs:.2f}")
print(f"CapEx for CCS: {compare_capex_ccs:.2f}")