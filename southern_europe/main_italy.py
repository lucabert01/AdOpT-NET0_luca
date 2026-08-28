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
    load_sector_reference_values,
    update_cement_hybrid_ccs_capacities,
    update_wastecal_ccs_capacities,
    convert_network_data_indices_to_names,
    apply_carbon_pricing_to_all_nodes,
    compute_opex_var_arcs,
    update_capex_gamma2_per_arc,
    load_pipeline_class_connection_matrix,
    PIPELINE_SIZE_CLASSES
)


#----- Scenario parameterization (shared across all runs) -----#
ref_year = 2024
discount_rate = 0.08 # default
co2_intensity_electricity = 0 # default (kg CO2/kWh)
cop_hp = 2.6 # default
levelized_capex_hp = 12.7 #computed from DanishEnergyAgency large HP excess heat
electricity_import_limit = 1000 # default
heat_import_limit = 3000 # default
wasteIn_import_limit = 1000 # default -- must exceed WasteCaL_CCS.json's size_max (140 t/h per node) with headroom, otherwise infeasible
max_transport_capacity = 3000  # default ceiling for truck/railway/generic networks (t/h)
# Per-arc capacity ceiling for the pipeline size classes - MUST match each
# class's calibrated kg/s range (converted to t/h, x3.6) in
# data_process/updated_network/pipeline_capex_per_arc_calculator.py::
# SIZE_CLASS_MASSFLOW_RANGES_KG_S, and the size_max in italy_data/networks/
# CO2_Pipeline_{small,medium,large}.json. Since size_max_defined_per_arc=1 in
# those JSONs, this is the ONLY thing that actually bounds a built arc's size
# - leaving it at the flat max_transport_capacity would let e.g. a 'small'
# pipe get built far outside its calibrated range, using extrapolated (wrong)
# economics.
pipeline_size_class_max_capacity_t_h = {
    "CO2_Pipeline_small": 104.4,
    "CO2_Pipeline_medium": 478.8,
    "CO2_Pipeline_large": 1692.0,
}
carbon_tax = 200  # euro per tonne CO2
enable_carbon_pricing = True
nr_DD_days = 15
node_metrics_suffix = 150  # or "150", "200", "" for the base case. Refers to the cutoff size for truck connections
node_metrics_file = f"node_metrics_{node_metrics_suffix}.xlsx"
objective = "emissions_minC"

# Refining/Other stay at their baseline technology in every scenario -- only
# cement/waste vary (see SCENARIOS below).
tech_for_refining = ["RefineryEmitter"]
tech_for_other = ["UnspecifiedEmitter"]

# If True, each sector's technology is modeled as "existing"(this requires
# exactly one technology selected per sector). If False, ALL technologies selected for a sector are modeled
# as "new"
tech_as_existing = True

# Used just for the conversion from emissions to actual product demand -- fixed,
# independent of which technology is actually selected per scenario (see
# load_sector_reference_values' docstring).
REFERENCE_EMITTER_TECHNOLOGIES = {
    "Cement": ["CementEmitter"],
    "Waste": ["WasteToEnergyEmitter"],
    "Refining": ["RefineryEmitter"],
    "Other": ["UnspecifiedEmitter"],
}

#----- Multi-run scenario matrix -----#
# 2x2 factorial: cement {baseline MEA retrofit, oxyfuel hybrid} x waste {baseline MEA
# retrofit, calcium looping}. Each entry's "name" becomes both the case-study working
# folder (Italy_CaseStudy/<name>) and the results folder
# (Results_CCSchainOptimization/<name>), numbered so they sort in a stable, readable
# order.
# NOTE: entries below are matched by FILENAME (via find_technology_file), not by the
# JSON's own "tec_type" field -- for CementHybridCCS those happen to be the same
# string, but WasteCaL_CCS.json's tec_type is "WasteToEnergyCaLCCS" while its filename
# (and therefore what must go here) is "WasteCaL_CCS".
SCENARIOS = [
    {"name": "mea",    "tech_for_cement": ["CementEmitter"],   "tech_for_waste": ["WasteToEnergyEmitter"]},
    {"name": "oxy",    "tech_for_cement": ["CementHybridCCS"], "tech_for_waste": ["WasteToEnergyEmitter"]},
    {"name": "cal",    "tech_for_cement": ["CementEmitter"],   "tech_for_waste": ["WasteCaL_CCS"]},
    {"name": "oxyCal", "tech_for_cement": ["CementHybridCCS"], "tech_for_waste": ["WasteCaL_CCS"]},
    # "Timeless" variant of "mea": every hourly demand/electricity/heat profile is
    # replaced by its own annual average (still 8760 values, so annual totals -- and
    # therefore annual emissions -- are unchanged), and only 1 typical/design day is
    # used instead of nr_DD_days. Since the flattened profiles have zero variance, 1
    # day reproduces the full year exactly, isolating the effect of removing hourly
    # variance with minimal change from the "mea" base case. Emitter/CCS capacity
    # sizing is untouched -- it's computed from the raw (non-flattened) profile.
    {"name": "mea_timeless", "tech_for_cement": ["CementEmitter"], "tech_for_waste": ["WasteToEnergyEmitter"],
     "flatten_profiles": True, "nr_dd_days": 1},
]

#----- Import data-----#
path_data_case_study = Path("./italy_data")

path_files_technologies = path_data_case_study / "technologies"
path_files_networks = path_data_case_study / "networks"
path_files_node_flux = path_data_case_study / "geographical_feature"
path_files_electricity = path_data_case_study / "electricity_metrics"
path_files_network_capex = path_data_case_study / "network_capex_metrics"

#----- Reference emission factors per sector (t CO2 emitted / t product output) -----#
# emission_profile_emitters.xlsx holds hourly CO2 EMISSION rates (t CO2/h) for every
# emitter. These factors convert that into each sector's real PRODUCT output rate.
# Computed once -- independent of scenario, see REFERENCE_EMITTER_TECHNOLOGIES above.
print("\nLoading sector emission factors from emitter technology JSONs...")
sector_emission_factor = load_sector_reference_values(
    path_files_technologies, REFERENCE_EMITTER_TECHNOLOGIES, ("Performance", "emission_factor")
)

#----- Flue-gas CO2 concentration per sector -----#
# Used to size the generic MEA CCS size range (assign_mea_technology). Also computed
# once, independent of scenario.
print("Loading sector CO2 concentrations from emitter technology JSONs...")
co2_concentration_by_type = load_sector_reference_values(
    path_files_technologies, REFERENCE_EMITTER_TECHNOLOGIES, ("Performance", "ccs", "co2_concentration")
)


def run_scenario(scenario_name: str, tech_for_cement: list, tech_for_waste: list,
                  flatten_profiles: bool = False, nr_dd_days: int = None):
    """
    Runs the full CCS-chain data-preparation + optimization pipeline for one
    emitter-technology scenario (a choice of cement and waste technology).

    Writes its case-study input data to Italy_CaseStudy/<scenario_name>/ and its
    results to Results_CCSchainOptimization/<scenario_name>/, so multiple scenarios
    can be run back to back without overwriting each other.

    :param str scenario_name: folder-safe name identifying this scenario
    :param list tech_for_cement: technology name(s) to use for the Cement sector
        (e.g. ["CementEmitter"] or ["CementHybridCCS"])
    :param list tech_for_waste: technology name(s) to use for the Waste sector,
        matched by FILENAME under italy_data/technologies/Emitter/, not by the JSON's
        own "tec_type" (e.g. ["WasteToEnergyEmitter"] or ["WasteCaL_CCS"] -- the
        latter's tec_type is "WasteToEnergyCaLCCS", a different string from its
        filename)
    :param bool flatten_profiles: if True, replace every hourly demand and
        electricity/heat price profile with its own annual average (see
        update_carrier_data's docstring) to remove hourly variance while preserving
        annual totals.
    :param int nr_dd_days: overrides the module-level nr_DD_days for this scenario's
        number of typical/design days, if given.
    """
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario_name}  (cement={tech_for_cement}, waste={tech_for_waste})")
    print("=" * 80)

    technology_selection = {
        "Cement": tech_for_cement,
        "Waste": tech_for_waste,
        "Refining": tech_for_refining,
        "Other": tech_for_other,
    }

    if tech_as_existing:
        for sector, techs in technology_selection.items():
            if len(techs) != 1:
                raise ValueError(
                    f"tech_as_existing=True requires exactly one technology per sector "
                    f"(a node's 'existing' plant can't be two technologies at once), but "
                    f"'{sector}' has {len(techs)}: {techs}. Either trim it to one "
                    f"technology or set tech_as_existing=False."
                )

    #----- Create folder for results -----#
    result_path = f"./Results_CCSchainOptimization/{scenario_name}"
    Path(result_path).mkdir(parents=True, exist_ok=True)
    # Create input data path and optimisation templates
    input_data_path = Path("Italy_CaseStudy") / scenario_name
    input_data_path.mkdir(parents=True, exist_ok=True)
    adopt.create_optimization_templates(input_data_path)

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
    network_emission_flux = calculate_emitter_capacities(network_emission_flux, sector_emission_factor)

    #----- Update topology json with carriers assignment -----#
    # Assign carriers to nodes based on their types
    assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux,
                              technology_selection)

    adopt.create_input_data_folder_template(input_data_path)

    # Update configmodel json
    with open(input_data_path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)
    configuration["optimization"]["objective"]["value"] = objective # set optimization objective
    configuration["solveroptions"]["mipgap"]["value"] = 0.01 # set MILP gap
    configuration['optimization']['typicaldays']['N']['value'] = nr_dd_days if nr_dd_days is not None else nr_DD_days
    configuration['optimization']['typicaldays']['method']['value'] = 1  # cluster demand/balances too (see IIS diagnosis: method 2 forced identical clinker output across hours mapped to the same typical day despite differing demand)
    configuration['reporting']['save_summary_path']['value'] = result_path
    configuration['reporting']['save_path']['value'] = result_path
    configuration['reporting']['case_name']['value'] = f"{objective}_{scenario_name}"

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
    network_emission_flux = assign_mea_technology(network_emission_flux, path_data_case_study, co2_concentration_by_type)

    # Then assign CCS technologies, passing both DataFrames
    # Note: This now uses the calculated capacities from calculate_emitter_capacities()
    assign_ccs_technologies_debug(network_location, network_emission_flux, path_data_case_study, input_data_path,
                                   technology_selection, tech_as_existing)


    # ===== DEBUG 1: After assign_ccs_technologies() =====
    print("\n🔍 DEBUG: Checking Technologies.json files after assignment...")

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

    # Copy over technology files using our custom function -- each copied JSON already
    # carries its own real Performance.emission_factor / ccs.co2_concentration, since
    # those are read (not overwritten) from the source templates; see
    # sector_emission_factor / co2_concentration_by_type above.
    copy_technology_data_custom(input_data_path, path_files_technologies, network_emission_flux)

    # Update CCS references in emitter technologies to match determined MEA sizes
    update_emitter_ccs_references(input_data_path, network_emission_flux, technology_selection, co2_concentration_by_type)

    # Write each cement node's own fixed clinker capacity into its copied CementHybridCCS.json
    # (only has an effect at nodes where CementHybridCCS was actually selected, i.e. it's a
    # no-op unless "CementHybridCCS" is in tech_for_cement)
    update_cement_hybrid_ccs_capacities(input_data_path, network_emission_flux)

    # Same, for WasteCaL_CCS's fixed waste-processing capacity (no-op unless
    # "WasteCaL_CCS" is in tech_for_waste)
    update_wastecal_ccs_capacities(input_data_path, network_emission_flux)

    #----- Add networks -----#
    # Three pipeline "size class" network technologies instead of one
    # CO2_Pipeline - each has its own JSON (italy_data/networks/
    # CO2_Pipeline_{small,medium,large}.json) and its own gamma1/gamma2 CSVs
    # (from gamma_defined_per_arc_pipeline_{small,medium,large}.xlsx below),
    # but they all reuse the same physical pipeline connectivity/distance
    # matrix (see the 'pipeline' data key mappings in defined_functions.py).
    new_network_types = ["CO2_Pipeline_small", "CO2_Pipeline_medium", "CO2_Pipeline_large", "CO2Truck", "CO2Railway"]

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

    #----- Apply per-pipeline-size-class connection overrides -----#
    # Curated via italy_data/geographical_feature/network_connections_dashboard.py
    # (Pipeline size classes tab) - lets
    # you remove specific arcs from an individual pipeline size class (e.g.
    # never build 'large' on an arc that only ever carries a small emitter's
    # flow), instead of every class sharing the exact same flat connectivity.
    # A no-op per class until an override is actually saved for it (falls
    # back to the shared 'pipeline' connectivity).
    pipeline_class_connections_path = path_files_network_capex / "pipeline_size_class_connections.xlsx"
    for size_class in PIPELINE_SIZE_CLASSES:
        network_data_dict[f"pipeline_{size_class}"] = load_pipeline_class_connection_matrix(
            pipeline_class_connections_path, size_class, network_pipeline
        )

    print("\n🔍 Converting network data indices to match topology...")
    network_data_dict = convert_network_data_indices_to_names(network_data_dict, network_location)


    # Distance matrices (use actual values from network data)
    update_network_distance_matrix_debug(input_data_path, network_data_dict, new_network_types)

    # Connection matrices (convert >0 to 1, keep 0 as 0)
    update_network_connection_matrix(input_data_path, network_data_dict)

    # Max size arc (all networks) - using predefined size_max value, with the
    # three pipeline size classes capped at their own calibrated ceiling
    print(f"🔍 Network sizing: default = {max_transport_capacity} t/h, "
          f"pipeline size classes = {pipeline_size_class_max_capacity_t_h}")
    update_network_size_max_arcs(input_data_path, network_data_dict, max_transport_capacity,
                                  per_network_type_capacity=pipeline_size_class_max_capacity_t_h)

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

    #----- Process gamma sheets from gamma_defined_per_arc_pipeline_{size}.xlsx -----#
    # Process gamma sheets and save as CSV files in each pipeline size class's network folder
    gamma_pipeline_small_per_arc = process_gamma_sheets_to_csv(
        path_files_network_capex,
        input_data_path,
        network_location,
        transport_mode="pipeline_small"
    )
    gamma_pipeline_medium_per_arc = process_gamma_sheets_to_csv(
        path_files_network_capex,
        input_data_path,
        network_location,
        transport_mode="pipeline_medium"
    )
    gamma_pipeline_large_per_arc = process_gamma_sheets_to_csv(
        path_files_network_capex,
        input_data_path,
        network_location,
        transport_mode="pipeline_large"
    )

    compute_opex_var_arcs(
        path_node_metrics=path_files_node_flux / node_metrics_file,
        path_output_root=input_data_path / "period1" / "network_topology" / "new",
        discount_rate=discount_rate,
        target_year=ref_year,
    )

    #----- Update truck/railway gamma2 (capex, per arc) with the capacity-based
    #      cost model from CostsFun_Share (Oeuvray et al. 2024), escalated from
    #      their EUR_2021 basis to EUR_{ref_year} -----#
    update_capex_gamma2_per_arc(
        path_node_metrics=path_files_node_flux / node_metrics_file,
        path_output_root=input_data_path / "period1" / "network_topology" / "new",
        path_network_data=input_data_path / "period1" / "network_data",
        discount_rate=discount_rate,
        target_year=ref_year,
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
        levelized_capex_hp,
        path_files_node_flux,
        electricity_import_limit,
        heat_import_limit,
        sector_emission_factor=sector_emission_factor,
        wasteIn_import_limit=wasteIn_import_limit,
        flatten_profiles=flatten_profiles,
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

    m = adopt.ModelHub()
    m.read_data(input_data_path, start_period=0, end_period=8759)
    m.quick_solve()


if __name__ == "__main__":
    import sys

    # Optional CLI filter: `python main_italy.py mea_timeless [other_name ...]` runs
    # only the named scenario(s) instead of the full SCENARIOS list.
    requested_names = sys.argv[1:]
    scenarios_to_run = (
        [s for s in SCENARIOS if s["name"] in requested_names] if requested_names else SCENARIOS
    )
    if requested_names:
        missing = set(requested_names) - {s["name"] for s in scenarios_to_run}
        if missing:
            raise ValueError(f"Unknown scenario name(s): {sorted(missing)}. "
                              f"Available: {[s['name'] for s in SCENARIOS]}")

    for scenario in scenarios_to_run:
        run_scenario(
            scenario["name"],
            scenario["tech_for_cement"],
            scenario["tech_for_waste"],
            flatten_profiles=scenario.get("flatten_profiles", False),
            nr_dd_days=scenario.get("nr_dd_days"),
        )
