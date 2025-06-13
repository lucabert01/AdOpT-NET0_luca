# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import pyomo.environ as pyo
import scipy.io as sci
import os


# Specify the path to your input data
path = Path("testNetwork")

# Create template files (comment these lines if already defined)
adopt.create_optimization_templates(path)

# Load json template
with open(path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = ["storage", "industrial_cluster"]
# Carriers:
topology["carriers"] = ["electricity", "CO2captured", "hydrogen", "heat", "gas"]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

end_period = 1


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

node_location = pd.read_csv(path / "NodeLocations.csv", sep=";", index_col=0, header=0)
node_location.at["industrial_cluster", "lon"] = 10
node_location.at["industrial_cluster", "lat"] = 10
node_location.at["industrial_cluster", "alt"] = 10
node_location.at["storage", "lon"] = 12
node_location.at["storage", "lat"] = 10
node_location.at["storage", "alt"] = 10
node_location = node_location.reset_index()
node_location.to_csv(path / "NodeLocations.csv", sep=";", index=False)

# Add technologies
with open(
    path / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "r"
) as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["CementEmitter"]

with open(
    path / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
) as json_file:
    json.dump(technologies, json_file, indent=4)

# # Copy over technology files
# adopt.copy_technology_data(path)


# Add networks
with open(path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = ["CO2_Pipeline"]

with open(path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# Make a new folder for the new network
os.makedirs(
    path / "period1" / "network_topology" / "new" / "CO2_Pipeline", exist_ok=True
)
# max size arc
arc_size = pd.read_csv(
    path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
    sep=";",
    index_col=0,
)
arc_size.loc["industrial_cluster", "storage"] = 10000
arc_size.to_csv(
    path
    / "period1"
    / "network_topology"
    / "new"
    / "CO2_Pipeline"
    / "size_max_arcs.csv",
    sep=";",
)
print("Max size per arc:", arc_size)

# Use the templates, fill and save them to the respective directory
# Connection
connection = pd.read_csv(
    path / "period1" / "network_topology" / "new" / "connection.csv",
    sep=";",
    index_col=0,
)
connection.loc["industrial_cluster", "storage"] = 1
connection.to_csv(
    path / "period1" / "network_topology" / "new" / "CO2_Pipeline" / "connection.csv",
    sep=";",
)
print("Connection:", connection)

# Delete the template
os.remove(path / "period1" / "network_topology" / "new" / "connection.csv")

# Distance
distance = pd.read_csv(
    path / "period1" / "network_topology" / "new" / "distance.csv", sep=";", index_col=0
)
distance.loc["industrial_cluster", "storage"] = 40
distance.to_csv(
    path / "period1" / "network_topology" / "new" / "CO2_Pipeline" / "distance.csv",
    sep=";",
)
print("Distance:", distance)

# Delete the template
os.remove(path / "period1" / "network_topology" / "new" / "distance.csv")

# Delete the max_size_arc template
os.remove(path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")


adopt.copy_network_data(path)


# Set import limits/cost
adopt.fill_carrier_data(
    path,
    value_or_data=5000,
    columns=["Import limit"],
    carriers=["electricity"],
    nodes=["industrial_cluster", "storage"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=20000,
    columns=["Import limit"],
    carriers=["heat"],
    nodes=["industrial_cluster", "storage"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=80,
    columns=["Import price"],
    carriers=["electricity"],
    nodes=["industrial_cluster", "storage"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=30,
    columns=["Import price"],
    carriers=["heat"],
    nodes=["industrial_cluster", "storage"],
)
adopt.fill_carrier_data(
    path,
    value_or_data=10,
    columns=["Demand"],
    carriers=["CO2captured"],
    nodes=["storage"],
)


carbon_price = np.linspace(0, 0, 8760)
carbon_cost_path = (
    path / "period1" / "node_data" / "industrial_cluster" / "CarbonCost.csv"
)
carbon_cost_template = pd.read_csv(carbon_cost_path, sep=";", index_col=0, header=0)
carbon_cost_template["price"] = carbon_price
carbon_cost_template = carbon_cost_template.reset_index()
carbon_cost_template.to_csv(carbon_cost_path, sep=";", index=False)

# Construct and solve the model
m = adopt.ModelHub()
m.read_data(path, start_period=0, end_period=end_period)
m.construct_model()
m.construct_balances()
m.solve()


capex_pipeline = m.model["full"].periods["period1"].network_block["CO2_Pipeline"].var_capex.value


print("Some results:")
print(f"CO2 Stored: {co2_stored:.2f}")
