# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np


# Specify the path to your input data
path = Path("./detailedCO2storage_test1")

# Create template files (comment these lines if already defined)
adopt.create_optimization_templates(path)

# Load json template
with open(path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = ["storage"]
# Carriers:
topology["carriers"] = ["electricity", "CO2captured"]
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
technologies["new"] = ["PermanentStorage_CO2_detailed"]

with open(path / "period1" / "node_data" / "storage" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Copy over technology files
adopt.copy_technology_data(path)

# Set import limits/cost
adopt.fill_carrier_data(path, value_or_data=0.0345, columns=['Import limit'], carriers=['CO2captured'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=-1500, columns=['Import price'], carriers=['CO2captured'], nodes=['storage'])
adopt.fill_carrier_data(path, value_or_data=20000, columns=['Import limit'], carriers=['electricity'], nodes=['storage'])


# Construct and solve the model
m = adopt.ModelHub()
m.read_data(path, start_period=0, end_period=6)
m.quick_solve()
m.model["full"].periods["period1"].node_blocks["storage"].tech_blocks_active[
    "PermanentStorage_CO2_detailed"].var_bhp.pprint()
a =1