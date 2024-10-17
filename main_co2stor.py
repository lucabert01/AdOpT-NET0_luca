# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import pyomo.environ as pyo
import scipy.io as sci
import os

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
topology["nodes"] = ["storage", "industrial_cluster"]
# Carriers:
topology["carriers"] = ["electricity", "CO2captured", "cement", "heat", "waste"]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

end_period = 3*180


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
technologies["new"] = ["PermanentStorage_CO2_detailed"]

with open(path / "period1" / "node_data" / "storage" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

with open(path / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["CementEmitter", "WasteToEnergyEmitter"]

with open(path / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Copy over technology files
adopt.copy_technology_data(path)


# Add networks
with open(path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = ["CO2PipelineOnshore"]

with open(path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# Make a new folder for the new network
os.makedirs(path / "period1" / "network_topology" / "new" / "CO2PipelineOnshore", exist_ok=True)
# max size arc
arc_size = pd.read_csv(path / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";", index_col=0)
arc_size.loc["industrial_cluster", "storage"] = 10000
arc_size.to_csv(path / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "size_max_arcs.csv", sep=";")
print("Max size per arc:", arc_size)

# Use the templates, fill and save them to the respective directory
# Connection
connection = pd.read_csv(path / "period1" / "network_topology" / "new" / "connection.csv", sep=";", index_col=0)
connection.loc["industrial_cluster", "storage"] = 1
connection.to_csv(path / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "connection.csv", sep=";")
print("Connection:", connection)

# Delete the template
os.remove(path / "period1" / "network_topology" / "new" / "connection.csv")

# Distance
distance = pd.read_csv(path / "period1" / "network_topology" / "new" / "distance.csv", sep=";", index_col=0)
distance.loc["industrial_cluster", "storage"] = 100
distance.to_csv(path / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "distance.csv", sep=";")
print("Distance:", distance)

# Delete the template
os.remove(path / "period1" / "network_topology" / "new" / "distance.csv")

# Delete the max_size_arc template
os.remove(path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")






# Load the .mat file for ROM model of saline aquifer
# general_performance_data_path = Path(__file__).parent
# aquifer_performance_data_path = (
#         general_performance_data_path
#         / "adopt_net0/data/technology_data/Sink/SalineAquifer_data/matrices_for_ROM.mat"
# )
# co2stor_data = sci.loadmat(aquifer_performance_data_path)
# inj_rate = co2stor_data["uNew"]
# inj_rate = inj_rate[:end_period, 0]
# conversion_factor = 550*3.6
# tot_inj_rate = np.zeros(8760)
# tot_inj_rate[:len(inj_rate)] = inj_rate*conversion_factor
# tot_inj_rate = pd.DataFrame(tot_inj_rate, columns=['Import limit'])

# Load and prepare data series
data_path = Path(__file__).parent/"data_simulations_co2stor_linear"
cement_emissions = np.load(data_path/"fullprofile_emissions_cement.npy", allow_pickle=True)
cement_emissions = prepare_data_series(cement_emissions, "Demand", end_period)
waste_emissions = np.load(data_path/"fullprofile_emissions_w2e.npy", allow_pickle=True)
waste_emissions = prepare_data_series(waste_emissions, "Demand", end_period)
carbon_price_timeseries = np.load(data_path/'fullprofile_co2_price.npy', allow_pickle=True)
carbon_price_timeseries = prepare_data_series(carbon_price_timeseries, "price", end_period)

# Set import limits/cost
adopt.fill_carrier_data(path, value_or_data=5000, columns=['Import limit'], carriers=['electricity'], nodes=['industrial_cluster','storage'])
adopt.fill_carrier_data(path, value_or_data=20000, columns=['Import limit'], carriers=['heat'], nodes=['industrial_cluster','storage'])
adopt.fill_carrier_data(path, value_or_data=100, columns=['Import price'], carriers=['electricity'], nodes=['industrial_cluster','storage'])
adopt.fill_carrier_data(path, value_or_data=40, columns=['Import price'], carriers=['heat'], nodes=['industrial_cluster','storage'])
adopt.fill_carrier_data(path, value_or_data=cement_emissions/3, columns=['Demand'], carriers=['cement'], nodes=['industrial_cluster'])
adopt.fill_carrier_data(path, value_or_data=waste_emissions/2, columns=['Demand'], carriers=['waste'], nodes=['industrial_cluster'])

carbon_price = carbon_price_timeseries['price'].values
carbon_price = np.linspace(110, 300, 8760)
carbon_cost_path = path / "period1" / "node_data" / "industrial_cluster" /"CarbonCost.csv"
carbon_cost_template = pd.read_csv(carbon_cost_path, sep=';', index_col=0, header=0)
carbon_cost_template['price'] = carbon_price
carbon_cost_template = carbon_cost_template.reset_index()
carbon_cost_template.to_csv(carbon_cost_path, sep=';', index=False)

# Construct and solve the model
m = adopt.ModelHub()
m.read_data(path, start_period=0, end_period=end_period)
m.construct_model()
m.construct_balances()
m.solve()



co2_stored = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_detailed"].var_input[t, "CO2captured"].value for t in range(1,end_period))
el_storage = sum(m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_detailed"].var_input[t, "electricity"].value for t in range(1,end_period))
co2_ccs_cement = sum(m.model["full"].periods["period1"].node_blocks["industrial_cluster"].tech_blocks_active[
    "CementEmitter"].var_output_ccs[t, "CO2captured"].value for t in range(1,end_period))
size_ccs_cement = m.model["full"].periods["period1"].node_blocks["industrial_cluster"].tech_blocks_active[
    "CementEmitter"].var_size_ccs.value
co2_ccs_w2e = sum(m.model["full"].periods["period1"].node_blocks["industrial_cluster"].tech_blocks_active[
    "WasteToEnergyEmitter"].var_output_ccs[t, "CO2captured"].value for t in range(1,end_period))
size_ccs_w2e = m.model["full"].periods["period1"].node_blocks["industrial_cluster"].tech_blocks_active[
    "WasteToEnergyEmitter"].var_size_ccs.value

print("Some results:")
print(f"CO2 Stored: {co2_stored:.2f}")
print(f"Electricity for Storage: {el_storage:.2f}")
print(f"CO2 captured cement: {co2_ccs_cement:.2f}")
print(f"Size CCS cement: {size_ccs_cement:.2f}")
print(f"CO2 captured W2E: {co2_ccs_w2e:.2f}")
print(f"Size CCS W2E: {size_ccs_w2e:.2f}")


