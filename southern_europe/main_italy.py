import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from data_process.utilities.defined_functions import (
    calculate_annual_emission_values,
    calculate_emitter_capacities,
    assign_carriers_to_nodes,
    assign_mea_technology,
    assign_ccs_technologies_debug,
    update_network_distance_matrix_debug,
    update_network_connection_matrix,
    update_network_size_max_arcs,
    load_climate_data_from_api_robust,
    update_carrier_data,
    process_gamma_sheets_to_csv,
    copy_technology_data_custom,
    update_emitter_ccs_references,
    convert_network_data_indices_to_names,
    apply_carbon_pricing_to_all_nodes,
    compute_opex_var_arcs
)


#----- Scenario parameterization -----#
ref_year = 2024
discount_rate = 0.08 # default
co2_intensity_electricity = 0 # default (kg CO2/kWh)
cop_hp = 2.6 # default
electricity_import_limit = 1000 # default
heat_import_limit = 3000 # default
max_transport_capacity = 3000
carbon_tax = 200  # euro per tonne CO2
enable_carbon_pricing = True
nr_DD_days = 15
node_metrics_suffix = 150  # or "150", "200", "" for the base case. Refers to the cutoff size for truck connections
node_metrics_file = f"node_metrics_{node_metrics_suffix}.xlsx"
#----- Create folder for results -----#
result_path = "./Results_CCSchainOptimization"
# Create input data path and optimisation templates
input_data_path = Path("Italy_CaseStudy")
input_data_path.mkdir(parents=True, exist_ok=True)
adopt.create_optimization_templates(input_data_path)
objective = "emissions_net"
#----- Import data-----#
path_data_case_study = Path("./italy_data")

path_files_technologies = path_data_case_study / "technologies"
path_files_networks = path_data_case_study / "networks"
path_files_node_flux = path_data_case_study / "geographical_feature"
path_files_electricity = path_data_case_study / "electricity_metrics"
path_files_network_capex = path_data_case_study / "network_capex_metrics"

network_location = pd.read_excel(path_files_node_flux/node_metrics_file, index_col=0, sheet_name='nodes') # nodes
network_emission_flux = pd.read_excel(path_files_node_flux/node_metrics_file, index_col=0, sheet_name='nodes') # annual emission fluxes
network_pipeline = pd.read_excel(path_files_node_flux/node_metrics_file, index_col=0, sheet_name='pipeline') # pipeline connection and distance
network_truck = pd.read_excel(path_files_node_flux/node_metrics_file, index_col=0, sheet_name='truck') # truck connection and distance
network_railway = pd.read_excel(path_files_node_flux/node_metrics_file, index_col=0, sheet_name='railway') # train connection and distance

electricity_price = pd.read_csv(path_files_electricity/"electricity_prices_hourly_2024.csv")
electricity_price = electricity_price.drop(index=range(1416, 1440)).reset_index(drop=True) #2024 is leap year


network_location['node_type'] = network_location['node_type'].str.strip()
network_emission_flux['node_type'] = network_emission_flux['node_type'].str.strip()

node_names = network_location['node_name'].unique().tolist()

# Print all node names to spot any remaining issues
print("All node names after stripping:")
for i, name in enumerate(node_names):
    print(f"  [{i}] repr: {repr(name)}")  # repr() reveals hidden spaces/chars

#----- Calculate annual emission values -----#
# Calculate the actual annual emission values using the specified formula logic
network_emission_flux = calculate_annual_emission_values(network_emission_flux, path_files_node_flux)

#----- Calculate emitter capacities -----#
# Calculate initial capacities for emitter technologies based on annual emissions and emission factors
# Using tonnes/hour units (appropriate for emitters that produce physical products)
print("Calculating emitter capacities based on annual emissions and emission factors...")
network_emission_flux = calculate_emitter_capacities(network_emission_flux)

#----- Update topology json with carriers assignment -----#
adopt.create_input_data_folder_template(input_data_path)

# Assign carriers to nodes based on their types
assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux)

# Update configmodel json
with open(input_data_path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
configuration["optimization"]["objective"]["value"] = objective # set optimization objective
configuration["solveroptions"]["mipgap"]["value"] = 0.02 # set MILP gap
configuration['optimization']['typicaldays']['N']['value'] = nr_DD_days
configuration['reporting']['save_summary_path']['value'] = result_path
configuration['reporting']['save_path']['value'] = result_path
configuration['reporting']['case_name']['value'] = f"{objective}"

with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

#----- Define node locations -----#
node_location = pd.read_csv(input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)

for node in node_names:
    node_row = network_location[network_location['node_name'] == node]
    if not node_row.empty:
        node_location.at[node, 'lon'] = node_row['longitude'].values[0]
        node_location.at[node, 'lat'] = node_row['latitude'].values[0]
        node_location.at[node, 'alt'] = node_row['altitude'].values[0]
    else:
        print(f"Warning: Node {node} not found in network_location dataframe")

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

#----- Add technologies for nodes -----#
# Assign MEA technology to network_emission_flux (now using calculated annual_emission values)
network_emission_flux = assign_mea_technology(network_emission_flux, path_data_case_study)

# Then assign CCS technologies, passing both DataFrames
# Note: This now uses the calculated capacities from calculate_emitter_capacities()
assign_ccs_technologies_debug(network_location, network_emission_flux, path_data_case_study, input_data_path)


# ===== DEBUG 1: After assign_ccs_technologies() =====
print("\n🔍 DEBUG: Checking Technologies.json files after assignment...")
import json
from pathlib import Path

# Check a few key nodes to see what's in their Technologies.json files
debug_nodes = ["Piacenza", "LOMELLINA ENERGIA", "Eni S.p.A Casalborsetti "]  # Mix of emitter and storage nodes

for node in debug_nodes:
    tech_file_path = input_data_path / "period1" / "node_data" / node / "Technologies.json"
    if tech_file_path.exists():
        with open(tech_file_path, "r") as f:
            tech_data = json.load(f)
        print(f"  📄 {node}:")
        print(f"    existing: {tech_data.get('existing', 'NOT_FOUND')}")
        print(f"    existing types: {[type(v).__name__ for v in tech_data.get('existing', {}).values()]}")
        print(f"    new: {tech_data.get('new', 'NOT_FOUND')}")
        print(f"    new type: {type(tech_data.get('new', 'NOT_FOUND')).__name__}")

        # Check if there are any binary/bytes objects
        for section, data in tech_data.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, bytes):
                        print(f"    ⚠️  FOUND BYTES: {section}.{key} = {value}")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, bytes):
                        print(f"    ⚠️  FOUND BYTES: {section}[{i}] = {item}")
    else:
        print(f"  ❌ {node}: Technologies.json not found")

# Copy over technology files using our custom function
copy_technology_data_custom(input_data_path, path_files_technologies, network_emission_flux)

# Update CCS references in emitter technologies to match determined MEA sizes
update_emitter_ccs_references(input_data_path, network_emission_flux)

#----- Add networks -----#
new_network_types = ["CO2_Pipeline", "CO2Truck", "CO2Railway"]

with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = new_network_types

with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# Since there are no existing networks, simply remove all template from the network folder
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "size.csv")

# Make folders for the new networks
for network_type in new_network_types:
    os.makedirs(input_data_path / "period1" / "network_topology" / "new" / network_type, exist_ok=True)

# Prepare network data dictionary for the updated functions
# Each matrix contains values where: 0 = no connection, >0 = connected with distance value
network_data_dict = {
    'pipeline': network_pipeline,
    'truck': network_truck,
    'railway': network_railway
}

print("\n🔍 Converting network data indices to match topology...")
network_data_dict = convert_network_data_indices_to_names(network_data_dict, network_location)


# Distance matrices (use actual values from network data)
update_network_distance_matrix_debug(input_data_path, network_data_dict, new_network_types)

# Connection matrices (convert >0 to 1, keep 0 as 0)
update_network_connection_matrix(input_data_path, network_data_dict)

# Max size arc (all networks) - using predefined size_max value
print(f"🔍 Network sizing: Using predefined transport capacity = {max_transport_capacity} tonnes/hour")
update_network_size_max_arcs(input_data_path, network_data_dict, max_transport_capacity)

# ===== DEBUG 2: After network matrix generation =====
print("\n🔍 DEBUG: Checking network matrices after generation...")

# Check the generated CSV files for each network type
for network_type in new_network_types:
    print(f"  📊 Network: {network_type}")

    # Check distance matrix
    distance_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "distance.csv"
    if distance_path.exists():
        try:
            distance_df = pd.read_csv(distance_path, sep=";", index_col=0)
            print(f"    Distance matrix shape: {distance_df.shape}")
            print(f"    Distance matrix dtypes: {distance_df.dtypes.unique()}")
            print(f"    Distance matrix sample values: {distance_df.iloc[0, 0]} (type: {type(distance_df.iloc[0, 0])})")

            # Check for any string/object values
            object_cols = distance_df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                print(f"    ⚠️  Object columns in distance: {object_cols.tolist()}")
                for col in object_cols:
                    print(f"      Sample values in {col}: {distance_df[col].head(3).tolist()}")
        except Exception as e:
            print(f"    ❌ Error reading distance matrix: {e}")

    # Check connection matrix
    connection_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "connection.csv"
    if connection_path.exists():
        try:
            connection_df = pd.read_csv(connection_path, sep=";", index_col=0)
            print(f"    Connection matrix shape: {connection_df.shape}")
            print(f"    Connection matrix dtypes: {connection_df.dtypes.unique()}")
            print(f"    Connection matrix unique values: {connection_df.values.flatten()[:10]}")
        except Exception as e:
            print(f"    ❌ Error reading connection matrix: {e}")

    # Check size_max_arcs matrix
    size_max_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "size_max_arcs.csv"
    if size_max_path.exists():
        try:
            size_max_df = pd.read_csv(size_max_path, sep=";", index_col=0)
            print(f"    Size max matrix shape: {size_max_df.shape}")
            print(f"    Size max matrix dtypes: {size_max_df.dtypes.unique()}")
            print(f"    Size max matrix sample values: {size_max_df.iloc[0, 0]} (type: {type(size_max_df.iloc[0, 0])})")
        except Exception as e:
            print(f"    ❌ Error reading size_max matrix: {e}")

# Delete the templates
os.remove(input_data_path / "period1" / "network_topology" / "new" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

# Copy network data and change costs
adopt.copy_network_data(input_data_path, path_files_networks)

#----- Process gamma sheets from capex_defined_per_arc.xlsx -----#
# Process gamma sheets and save as CSV files in network folder
gamma_pipeline_per_arc = process_gamma_sheets_to_csv(
    path_files_network_capex,
    input_data_path,
    network_location,
    transport_mode="pipeline"
)




compute_opex_var_arcs(
    path_node_metrics=path_files_node_flux / node_metrics_file,
    path_output_root=input_data_path / "period1" / "network_topology" / "new",
)

#----- Update carrier data with pricing, emission factors, and demands -----#
print("Updating carrier data with hourly demand profiles...")
update_carrier_data(
    input_data_path,
    electricity_price,
    network_emission_flux,
    path_files_technologies,
    node_names,
    co2_intensity_electricity,
    cop_hp,
    path_files_node_flux,
    electricity_import_limit,
    heat_import_limit
)

# ----- Apply Carbon Pricing -----#
if enable_carbon_pricing and carbon_tax > 0:
    print(f"\n" + "=" * 60)
    print(f"💰 APPLYING CARBON PRICING: €{carbon_tax}/tonne CO2")
    print(f"=" * 60)

    # Apply carbon pricing to all nodes
    carbon_pricing_success = apply_carbon_pricing_to_all_nodes(
        input_data_path,
        carbon_tax,
        node_names
    )

    if carbon_pricing_success:
        print(f"✅ Carbon pricing successfully applied to all {len(node_names)} nodes")
    else:
        print(f"⚠️  Some nodes failed carbon pricing application")
else:
    print(f"\n💡 Carbon pricing disabled (carbon_tax={carbon_tax}, enabled={enable_carbon_pricing})")

#----- Define climate data -----#
load_climate_data_from_api_robust(input_data_path)

#----- Build and solve optimization problem -----#
print("Building and solving optimization problem...")

# ===== DEBUG 3: Before optimization =====
print("\n🔍 DEBUG: Final checks before optimization...")

# Check topology consistency
with open(input_data_path / "Topology.json", "r") as f:
    topology = json.load(f)

print(f"  🗺️  Topology:")
print(f"    Nodes ({len(topology['nodes'])}): {topology['nodes']}")
print(f"    Carriers ({len(topology['carriers'])}): {topology['carriers']}")
print(f"    Investment periods: {topology['investment_periods']}")

# Check if all nodes have proper directory structure
missing_files = []
for node in topology["nodes"]:
    node_dir = input_data_path / "period1" / "node_data" / node
    if not node_dir.exists():
        missing_files.append(f"{node}/directory")
        continue

    # Check required files
    required_files = ["Technologies.json", "ClimateData.csv", "CarbonCost.csv"]
    for req_file in required_files:
        if not (node_dir / req_file).exists():
            missing_files.append(f"{node}/{req_file}")

    # Check carrier data directory
    carrier_dir = node_dir / "carrier_data"
    if not carrier_dir.exists():
        missing_files.append(f"{node}/carrier_data/directory")
    else:
        for carrier in topology["carriers"]:
            carrier_file = carrier_dir / f"{carrier}.csv"
            if not carrier_file.exists():
                missing_files.append(f"{node}/carrier_data/{carrier}.csv")

if missing_files:
    print(f"  ❌ Missing files/directories: {missing_files[:10]}...")  # Show first 10
else:
    print(f"  ✅ All required files present")

import adopt_net0 as adopt
import pyomo.environ as pyo
from pyomo.environ import SolverFactory

m = adopt.ModelHub()
m.read_data(input_data_path, start_period=0, end_period=8759)

m.construct_model()
m.construct_balances()

# 1. Print the keys to see how adopt_net0 stores the model(s)
print(f"Available keys in m.model: {m.model.keys()}")

# 2. Extract the actual Pyomo model from the dictionary
# This grabs the first (and likely only) key in the dictionary
model_key = list(m.model.keys())[0]
pyomo_model = m.model[model_key]

# 3. Verify it is now a Pyomo ConcreteModel
print(f"Type of pyomo_model is now: {type(pyomo_model)}")

# ----- DIAGNOSTIC: force CCS installation at ROBILANTE to test true (in)feasibility -----
# Robilante's only available CCS size class (MEA_large) requires size_ccs >= 142.6 t/h
# if installed at all (para_size_min_ccs). The pure emissions_net objective run put
# size_ccs=0 there; this forces the "installed" branch of the disjunction to see
# whether that's a real infeasibility or just an unexploited optimization opportunity.
b_robilante = (
    pyomo_model.periods["period1"]
    .node_blocks["ROBILANTE cement plant"]
    .tech_blocks_active["CementEmitter_existing"]
)
b_robilante.const_force_ccs_installed = pyo.Constraint(
    expr=b_robilante.var_size_ccs >= b_robilante.para_size_min_ccs
)
print(f"Forced CCS at Robilante: size_ccs >= {b_robilante.para_size_min_ccs.value}")

m.solve()

if m.solution.solver.termination_condition in [
    pyo.TerminationCondition.infeasible,
    pyo.TerminationCondition.infeasibleOrUnbounded,
]:
    gurobi_model = m.solver._solver_model
    gurobi_model.computeIIS()
    gurobi_model.write("infeasible.ilp")
    print("IIS written to infeasible.ilp")

    print("\n--- IIS CONSTRAINTS (auto-translated to Pyomo names) ---")
    for g_con in gurobi_model.getConstrs():
        if g_con.IISConstr:
            p_con = m.solver._solver_con_to_pyomo_con_map.get(g_con)
            print(f"  {g_con.ConstrName} -> {p_con.name if p_con is not None else '(no pyomo mapping)'}")

    print("\n--- IIS VARIABLE BOUNDS (auto-translated to Pyomo names) ---")
    for g_var in gurobi_model.getVars():
        if g_var.IISLB or g_var.IISUB:
            p_var = m.solver._solver_var_to_pyomo_var_map.get(g_var)
            bound_type = "/".join(b for b, flag in [("LB", g_var.IISLB), ("UB", g_var.IISUB)] if flag)
            print(f"  {g_var.VarName} ({bound_type}) -> {p_var.name if p_var is not None else '(no pyomo mapping)'}")
else:
    print(f"\nSolve status: {m.solution.solver.termination_condition}")
    print(f"emissions_net with CCS forced at Robilante: {pyomo_model.periods['period1'].var_emissions_net.value}")
    print("(compare against the unconstrained stage-1 minimum of ~4,699,803 t)")