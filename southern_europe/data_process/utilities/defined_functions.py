import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime


def calculate_annual_emission_values(network_emission_flux):
    """
    Calculate the actual annual emission values for each emitter based on Excel formula logic.

    This function recreates the Excel formula: =IF(ISNUMBER(computed_annual_flux), computed_annual_flux, annual_flux)
    If 'computed_annual_flux' has a valid number, use it; otherwise use 'annual_flux'.

    Parameters:
        - network_emission_flux: DataFrame containing emission data with 'annual_flux' and 'computed_annual_flux' columns

    Returns:
        - network_emission_flux: Updated DataFrame with calculated 'annual_emission' column
    """

    def apply_emission_formula(row):
        """
        Apply the Excel formula logic: =IF(ISNUMBER(computed_annual_flux),computed_annual_flux,annual_flux)
        If 'computed_annual_flux' has a valid number, use it; otherwise use 'annual_flux'
        """
        computed_flux = row.get('computed_annual_flux', None)
        annual_flux = row.get('annual_flux', 0)

        # Check if computed_flux is a valid number (not NaN, not None, not empty, not zero)
        if pd.notna(computed_flux) and computed_flux != 0:
            return computed_flux
        else:
            return annual_flux

    # Apply the logic to create the annual_emission column
    network_emission_flux['annual_emission'] = network_emission_flux.apply(apply_emission_formula, axis=1)

    return network_emission_flux


def calculate_emitter_capacities(network_emission_flux, path_data_case_study, path_files_node_flux,
                                 capacity_unit="tonnes_per_hour"):
    """
    Calculate initial capacities for emitter technologies based on:
    1. Maximum demand value from Excel sheet (if available)
    2. Annual emissions and emission factors (fallback method)

    ENHANCED: Uses real demand data when available, handles multiple emitters per node properly.

    Parameters:
        - network_emission_flux: DataFrame containing emission data with 'annual_emission' column
        - path_data_case_study: Path to the case study data directory
        - path_files_node_flux: Path to the directory containing real hourly demand profiles Excel file
        - capacity_unit: Unit for capacity calculation - "MW" or "tonnes_per_hour" (default)

    Returns:
        - network_emission_flux: Updated DataFrame with 'emitter_capacity' column
    """
    # Ensure annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        raise ValueError("'annual_emission' column not found. Please run calculate_annual_emission_values() first.")

    # Define mapping from node_type to technology file and emission factor key path
    node_type_mapping = {
        'Waste': ('Emitter/WasteToEnergyEmitter.json', ['Performance', 'emission_factor']),
        'Cement': ('Emitter/CementEmitter.json', ['Performance', 'emission_factor']),
        'Refining': ('Emitter/RefineryEmitter.json', ['Performance', 'emission_factor']),
        'Other': ('Emitter/UnspecifiedEmitter.json', ['Performance', 'emission_factor'])
    }

    # Load emission factors from JSON files
    emission_factors = {}
    for node_type, (filename, factor_key_path) in node_type_mapping.items():
        tech_file_path = path_data_case_study / "technologies" / filename
        if tech_file_path.exists():
            with open(tech_file_path, 'r') as f:
                tech_data = json.load(f)

                # Navigate through the nested structure
                try:
                    current_data = tech_data
                    for key in factor_key_path:
                        current_data = current_data[key]

                    emission_factors[node_type] = current_data
                    print(f"✅ Loaded emission factor for {node_type}: {current_data}")
                except KeyError as e:
                    print(f"Warning: Key path {factor_key_path} not found in {filename} (missing: {e})")
                    emission_factors[node_type] = 1.0  # Default value
        else:
            print(f"Warning: Technology file {filename} not found at {tech_file_path}")
            emission_factors[node_type] = 1.0  # Default value

    # Initialize capacity column
    network_emission_flux['emitter_capacity'] = 0.0

    print(f"🔍 DEBUG: DataFrame shape before processing: {network_emission_flux.shape}")
    print(f"🔍 DEBUG: DataFrame index: {network_emission_flux.index.tolist()}")

    # Define Excel file path for real demand data
    excel_file_path = path_files_node_flux / "emitter_hourly_profile.xlsx"
    excel_exists = excel_file_path.exists()

    if excel_exists:
        print(f"📊 Excel file found: {excel_file_path}")
    else:
        print(f"📊 Excel file not found: {excel_file_path} - using fallback method for all nodes")

    # Create carrier name mapping
    carrier_to_node_type = {
        'waste': 'Waste',
        'cement': 'Cement',
        'refined_product': 'Refining',
        'industrial_product': 'Other'
    }

    node_type_to_carrier = {v: k for k, v in carrier_to_node_type.items()}

    # Calculate capacities for each emitter - Use enumerate to get unique row positions
    for row_position, (idx, row) in enumerate(network_emission_flux.iterrows()):
        node_name = row['node_name']
        node_type = row['node_type']
        annual_emission = row['annual_emission']  # kg CO2/year

        print(
            f"🔍 DEBUG: Processing row position {row_position} (index {idx}) - Node: {node_name}, Type: {node_type}, Emission: {annual_emission}")

        # Skip non-emitter nodes
        if node_type in ['Storage', 'Transport'] or annual_emission == 0:
            print(f"🔍 DEBUG: Skipping row position {row_position} - non-emitter or zero emission")
            continue

        capacity = 0.0
        method_used = "unknown"

        # Method 1: Try to get capacity from Excel sheet (if available)
        if excel_exists and node_type in node_type_to_carrier:
            carrier_name = node_type_to_carrier[node_type]
            sheet_name = f"{node_name} - {node_type}"

            try:
                print(f"  📊 Attempting to load sheet: '{sheet_name}'")
                demand_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

                if 'Demand' in demand_df.columns:
                    # Convert to numeric, replacing any non-numeric values with NaN
                    demand_numeric = pd.to_numeric(demand_df['Demand'], errors='coerce')

                    # Check if we have valid data
                    valid_count = demand_numeric.notna().sum()
                    total_count = len(demand_numeric)

                    if valid_count > total_count * 0.8:  # At least 80% valid data
                        # Get maximum demand value (capacity is the peak demand)
                        max_demand_value = demand_numeric.max()

                        if pd.notna(max_demand_value) and max_demand_value > 0:
                            if capacity_unit == "tonnes_per_hour":
                                capacity = round(max_demand_value, 2)  # Assume data is already in tonnes/hour
                                unit_label = "tonnes/hour"
                            elif capacity_unit == "MW":
                                # Convert from tonnes/hour to MW (rough estimate)
                                capacity_mw = max_demand_value * 0.001  # Adjust conversion factor as needed
                                capacity = round(capacity_mw, 2)
                                unit_label = "MW"
                            else:
                                raise ValueError(f"Unsupported capacity_unit: {capacity_unit}")

                            method_used = f"excel_max_demand ({max_demand_value:.2f} from sheet)"
                            print(
                                f"      ✅ Using Excel max demand: {max_demand_value:.2f} -> capacity: {capacity:.2f} {unit_label}")
                        else:
                            print(f"      ⚠️  Excel sheet has no valid positive demand values")
                    else:
                        print(
                            f"      ⚠️  Excel sheet has too many invalid values ({total_count - valid_count} invalid)")
                else:
                    print(f"      ⚠️  Excel sheet has no 'Demand' column")

            except Exception as e:
                print(f"      ⚠️  Could not load Excel sheet '{sheet_name}': {e}")

        # Method 2: Fallback to calculation from annual emissions (original method)
        if capacity == 0.0 and node_type in emission_factors:
            print(f"      🔄 Using fallback method (annual emissions)")

            # Calculate annual demand in kg product/year using the emission factor
            annual_demand_kg = annual_emission / emission_factors[node_type]  # kg product/year

            print(
                f"🔍 DEBUG: Row position {row_position} calculation - Emission: {annual_emission}, Emission Factor: {emission_factors[node_type]}, Annual Demand: {annual_demand_kg}")

            if capacity_unit == "tonnes_per_hour":
                # Convert to tonnes/hour: kg/year -> tonnes/hour
                # 1 year = 8760 hours, 1 tonne = 1000 kg
                capacity_tonnes_per_hour = annual_demand_kg / (8760 * 1000)
                capacity = round(capacity_tonnes_per_hour, 2)  # Round to 2 decimal places
                unit_label = "tonnes/hour"

            elif capacity_unit == "MW":
                # Convert to MW assuming typical industrial process energy intensity
                # Rough estimate: 1 kg product/hour ≈ 0.001 MW (adjustable based on process)
                # Annual demand kg/year -> hourly demand kg/hour -> MW
                hourly_demand_kg = annual_demand_kg / 8760  # kg product/hour
                capacity_mw = hourly_demand_kg * 0.001  # Convert to MW (rough estimate)
                capacity = round(capacity_mw, 2)  # Round to 2 decimal places
                unit_label = "MW"

            else:
                raise ValueError(f"Unsupported capacity_unit: {capacity_unit}")

            method_used = f"annual_emissions_calculation"

        # Set capacity using iloc to handle duplicate indices properly
        if capacity > 0:
            network_emission_flux.iloc[
                row_position, network_emission_flux.columns.get_loc('emitter_capacity')] = capacity

            print(
                f"🔍 DEBUG: Set capacity {capacity} for row position {row_position} (index {idx}, Node: {node_name}, Type: {node_type})")
            print(f"Node {node_name} ({node_type}): "
                  f"Method: {method_used}, "
                  f"Capacity: {capacity:.2f} {unit_label}")
        else:
            print(f"      ❌ Could not determine capacity for {node_name} ({node_type})")

    # Debug: Check final capacities to verify different values for different emitters
    print(f"🔍 FINAL DEBUG: Capacity verification by node and type:")
    capacity_by_node_type = {}
    method_summary = {"excel_max_demand": 0, "annual_emissions_calculation": 0, "failed": 0}

    for row_position, (idx, row) in enumerate(network_emission_flux.iterrows()):
        if row['node_type'] not in ['Storage', 'Transport'] and row['annual_emission'] > 0:
            key = f"{row['node_name']}_{row['node_type']}"
            capacity_by_node_type[key] = row['emitter_capacity']
            print(f"  {key}: {row['emitter_capacity']:.2f} (row_pos: {row_position}, idx: {idx})")

            # Count methods used (approximate based on capacity values)
            if row['emitter_capacity'] > 0:
                # Check if this looks like it came from Excel (typically higher values) or calculation
                if excel_exists:
                    method_summary["excel_max_demand"] += 1
                else:
                    method_summary["annual_emissions_calculation"] += 1
            else:
                method_summary["failed"] += 1

    # Check for duplicate capacities at the same node
    nodes_with_multiple_emitters = {}
    for key, capacity in capacity_by_node_type.items():
        node_name = key.split('_')[0]
        if node_name not in nodes_with_multiple_emitters:
            nodes_with_multiple_emitters[node_name] = []
        nodes_with_multiple_emitters[node_name].append((key, capacity))

    for node_name, emitters in nodes_with_multiple_emitters.items():
        if len(emitters) > 1:
            capacities = [cap for _, cap in emitters]
            if len(set(capacities)) == 1:  # All capacities are the same
                print(f"⚠️  WARNING: Node {node_name} has multiple emitters with identical capacities:")
                for emitter_key, capacity in emitters:
                    print(f"    {emitter_key}: {capacity}")
                print(f"    This may indicate that Excel sheets weren't found for individual emitters")
            else:
                print(f"✅ SUCCESS: Node {node_name} has multiple emitters with different capacities:")
                for emitter_key, capacity in emitters:
                    print(f"    {emitter_key}: {capacity}")

    # Summary
    print(f"\n📊 CAPACITY CALCULATION SUMMARY:")
    print(f"  Excel max demand method: {method_summary['excel_max_demand']} emitters")
    print(f"  Annual emissions calculation: {method_summary['annual_emissions_calculation']} emitters")
    print(f"  Failed to calculate: {method_summary['failed']} emitters")
    print(f"  Total processed: {sum(method_summary.values())} emitters")

    return network_emission_flux


def assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux):
    """
    Assign appropriate carriers to each node based on their type(s).

    Carrier assignment rules:
    - All nodes get: electricity, heat, CO2captured (except Transport nodes)
    - Transport nodes get: electricity, CO2captured only (no heat)
    - Cement nodes also get: cement
    - Waste nodes also get: waste
    - Refining nodes also get: refined_product
    - Other nodes also get: industrial_product
    - Nodes with multiple emitters get all relevant carriers from both emitter types

    Parameters:
        - input_data_path: Path to the input data directory
        - network_location: DataFrame containing node information with node_name and node_type
        - network_emission_flux: DataFrame containing emission data with node_name and node_type

    Returns:
        - None (updates Topology.json file)
    """

    # Get all unique nodes
    all_nodes = network_location['node_name'].unique().tolist()

    # Base carriers that most nodes get
    base_carriers = ["electricity", "heat", "CO2captured"]
    transport_carriers = ["electricity", "CO2captured"]  # Transport nodes don't get heat

    # Mapping from emitter node_type to specific carriers
    emitter_carriers = {
        'Cement': 'cement',
        'Waste': 'waste',
        'Refining': 'refined_product',
        'Other': 'industrial_product'
    }

    # Collect all unique carriers needed
    all_carriers = set(base_carriers)

    # Add emitter-specific carriers
    for node_name in all_nodes:
        # Check if this node has emitters in network_emission_flux
        node_emission_rows = network_emission_flux[network_emission_flux['node_name'] == node_name]

        for _, emission_row in node_emission_rows.iterrows():
            emitter_type = emission_row['node_type']
            if emitter_type in emitter_carriers:
                all_carriers.add(emitter_carriers[emitter_type])

    # Convert to sorted list for consistent output
    all_carriers = sorted(list(all_carriers))

    # Update Topology.json
    with open(input_data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    topology["nodes"] = all_nodes
    topology["carriers"] = all_carriers
    topology["investment_periods"] = ["period1"]

    with open(input_data_path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

    # Log carrier assignment summary
    print(f"Carrier assignment completed:")
    print(f"  - Total nodes: {len(all_nodes)}")
    print(f"  - Total carriers: {all_carriers}")

    # Show detailed assignment for verification
    node_carrier_summary = {}
    for node_name in all_nodes:
        # Get node type(s) from network_location
        node_location_rows = network_location[network_location['node_name'] == node_name]
        node_emission_rows = network_emission_flux[network_emission_flux['node_name'] == node_name]

        # Determine carriers for this node
        node_carriers = set()

        # Check if any location row is Transport type
        is_transport = any(row['node_type'] == 'Transport' for _, row in node_location_rows.iterrows())

        if is_transport:
            node_carriers.update(transport_carriers)
        else:
            node_carriers.update(base_carriers)

        # Add emitter-specific carriers
        for _, emission_row in node_emission_rows.iterrows():
            emitter_type = emission_row['node_type']
            if emitter_type in emitter_carriers:
                node_carriers.add(emitter_carriers[emitter_type])

        node_carrier_summary[node_name] = sorted(list(node_carriers))

    # Show summary by node type
    transport_nodes = []
    emitter_nodes = []
    storage_nodes = []

    for node_name in all_nodes:
        node_types = set()
        # Get types from both DataFrames
        for _, row in network_location[network_location['node_name'] == node_name].iterrows():
            node_types.add(row['node_type'])
        for _, row in network_emission_flux[network_emission_flux['node_name'] == node_name].iterrows():
            node_types.add(row['node_type'])

        if 'Transport' in node_types:
            transport_nodes.append(node_name)
        elif any(t in ['Cement', 'Waste', 'Refining', 'Other'] for t in node_types):
            emitter_nodes.append(node_name)
        elif 'Storage' in node_types:
            storage_nodes.append(node_name)

    print(f"  - Transport nodes ({len(transport_nodes)}): {transport_nodes}")
    print(f"  - Emitter nodes ({len(emitter_nodes)}): {emitter_nodes}")
    print(f"  - Storage nodes ({len(storage_nodes)}): {storage_nodes}")

    return True


def assign_mea_technology(network_emission_flux, path_data_case_study):
    """
    Determines appropriate MEA (Monoethanolamine) carbon capture technology scale
    for emitter nodes based on their annual CO2 emissions.

    This function analyzes emission data for each node and determines the appropriate
    MEA technology scale (small, medium, large), adding it to a new column 'mea_technology'.

    Parameters:
        - network_emission_flux: DataFrame containing node information and emission data with 'annual_emission' column
        - path_data_case_study: Path to the case study data directory

    Returns:
        - network_emission_flux: Updated DataFrame with mea_technology column added
    """
    # Ensure annual_emission column exists
    if 'annual_emission' not in network_emission_flux.columns:
        raise ValueError("'annual_emission' column not found. Please run calculate_annual_emission_values() first.")

    # Define paths to different MEA technology scales
    mea_paths = {
        "large": path_data_case_study / "technologies/CCSTechnologies/MEA_large.json",
        "medium": path_data_case_study / "technologies/CCSTechnologies/MEA_medium.json",
        "small": path_data_case_study / "technologies/CCSTechnologies/MEA_small.json"
    }

    # Load MEA technology specifications from JSON files
    mea_data = {}
    for scale, path in mea_paths.items():
        with open(path, "r") as f:
            mea_data[scale] = json.load(f)

    # Add column for MEA technology if it doesn't exist
    network_emission_flux['mea_technology'] = None

    # Process each row in the network_emission_flux DataFrame
    for idx, row in network_emission_flux.iterrows():
        node_name = row['node_name']
        node_type = row['node_type']

        # Skip non-emitter nodes (Storage and Transport)
        if node_type in ["Storage", "Transport"]:
            continue

        # Get the node's calculated annual CO2 emission (kg/year)
        annual_emission = row["annual_emission"]

        # Determine CO2 concentration based on emitter type
        if node_type in ["Waste"]:
            co2_concentration = 0.07
        elif node_type in ["Cement"]:
            co2_concentration = 0.20
        elif node_type in ["Refining"]:
            co2_concentration = 0.25
        else:
            co2_concentration = 0.15

        # Calculate CO2 ranges for each MEA scale based on technology specs
        # Convert MEA scale from t/h to kg/year for comparison
        conversion_factor = 1000 * 24 * 365  # t/h to kg/year

        mea_ranges = {}
        for scale, data in mea_data.items():
            min_co2 = co2_concentration * data["size_min"] * conversion_factor
            max_co2 = co2_concentration * data["size_max"] * conversion_factor
            mea_ranges[scale] = (min_co2, max_co2)

        # Find the MEA scale that matches the node's emission range
        suitable_mea = None
        for scale, (min_co2, max_co2) in mea_ranges.items():
            if min_co2 <= annual_emission <= max_co2:
                suitable_mea = scale
                break

        # If no exact match found, choose the closest scale
        if suitable_mea is None:
            distances = {}
            for scale, (min_co2, max_co2) in mea_ranges.items():
                if annual_emission < min_co2:
                    distances[scale] = min_co2 - annual_emission
                elif annual_emission > max_co2:
                    distances[scale] = annual_emission - max_co2

            suitable_mea = min(distances, key=distances.get)

        # Store the suitable MEA technology in the mea_technology column
        mea_tech_path = str(path_data_case_study / f"technologies/CCSTechnologies/MEA_{suitable_mea}.json")
        network_emission_flux.at[idx, 'mea_technology'] = mea_tech_path

    return network_emission_flux


def assign_ccs_technologies(network_location, network_emission_flux, path_data_case_study, input_data_path):
    """
    Assigns appropriate technologies to nodes based on their type and previously determined MEA technology.
    Handles nodes with multiple emitters by accumulating all required technologies.

    FIXED: Ensures proper data types and JSON serialization for DataHandle compatibility.

    Parameters:
        - network_location: DataFrame containing node information
        - network_emission_flux: DataFrame containing emission data, MEA technology assignments, and calculated capacities
        - path_data_case_study: Path to the case study data directory
        - input_data_path: Path to the input data directory

    Returns:
        - None
    """
    # Ensure capacity column exists
    if 'emitter_capacity' not in network_emission_flux.columns:
        raise ValueError("'emitter_capacity' column not found. Please run calculate_emitter_capacities() first.")

    # Group by unique node names to handle multiple emitters per node
    unique_nodes = network_location['node_name'].unique()

    for node_name in unique_nodes:
        # Get all rows for this node
        node_rows = network_location[network_location['node_name'] == node_name]

        # Initialize technology dictionaries with proper data types
        existing_techs_dict = {}
        new_techs_list = []

        # Process each row for this node
        for idx, row in node_rows.iterrows():
            node_type = row['node_type']

            if node_type == "Storage":
                # Storage nodes get permanent CO2 storage technology as "new"
                storage_tech_path = path_data_case_study / "technologies/Sink/PermanentStorage_CO2_simple.json"
                if storage_tech_path.exists():
                    new_techs_list.append("PermanentStorage_CO2_simple")
                    print(f"Found storage technology at: {storage_tech_path}")
                else:
                    print(f"Warning: Storage technology file not found at {storage_tech_path}")
                    # Check if it's in the main technologies folder
                    alt_storage_path = path_data_case_study / "technologies/PermanentStorage_CO2_simple.json"
                    if alt_storage_path.exists():
                        new_techs_list.append("PermanentStorage_CO2_simple")
                        print(f"Found storage technology at alternative path: {alt_storage_path}")
                    else:
                        new_techs_list.append("PermanentStorage_CO2_simple")  # Add anyway, let system handle

            elif node_type == "Transport":
                # Transport nodes don't require specific technologies
                pass

            else:  # Emitter nodes (Waste, Cement, Refining, Other)
                # For emitter nodes, we add the emitter technology with calculated capacity as "existing"

                # Get the calculated capacity for this specific emitter
                emitter_row = network_emission_flux[
                    (network_emission_flux['node_name'] == node_name) &
                    (network_emission_flux['node_type'] == node_type)
                    ]

                if not emitter_row.empty:
                    capacity = float(emitter_row['emitter_capacity'].iloc[0])  # Ensure it's a Python float
                else:
                    capacity = 0.0
                    print(f"Warning: No capacity data found for {node_name} ({node_type})")

                # Assign appropriate emitter technology based on node type with calculated capacity
                if node_type == "Waste":
                    existing_techs_dict["WasteToEnergyEmitter"] = capacity
                elif node_type == "Cement":
                    existing_techs_dict["CementEmitter"] = capacity
                elif node_type == "Refining":
                    existing_techs_dict["RefineryEmitter"] = capacity
                elif node_type == "Other":
                    existing_techs_dict["UnspecifiedEmitter"] = capacity

                print(f"Assigned {node_type} emitter to {node_name} with capacity: {capacity:.2f}")

        # Remove duplicates from new_techs_list
        new_techs_list = list(set(new_techs_list))

        # Read the node's current Technology.json file
        tech_file_path = input_data_path / "period1" / "node_data" / node_name / "Technologies.json"

        # FIXED: Ensure all values are proper Python types (not numpy types) for JSON serialization
        # Convert any numpy types to native Python types
        existing_techs_clean = {}
        for tech_name, capacity in existing_techs_dict.items():
            existing_techs_clean[str(tech_name)] = float(capacity)  # Ensure string keys and float values

        new_techs_clean = [str(tech) for tech in new_techs_list]  # Ensure string elements

        technologies = {
            "existing": existing_techs_clean,
            "new": new_techs_clean,
        }

        # Write updated technologies to the file with proper JSON serialization
        with open(tech_file_path, "w") as json_file:
            json.dump(technologies, json_file, indent=4, ensure_ascii=False)

        print(
            f"Technologies assigned to {node_name}: existing={list(existing_techs_clean.keys())} (with capacities), new={new_techs_clean}")

def find_technology_file(tech_name, search_path):
    """Find a technology file in the search path and its subdirectories"""
    for root, dirs, files in os.walk(search_path):
        if f"{tech_name}.json" in files:
            return Path(root) / f"{tech_name}.json"
    return None


def copy_technology_data_custom(input_data_path, path_files_technologies, network_emission_flux=None):
    """
    Custom function to copy technology data files, handling both dict and list formats for "new" technologies.
    Also copies MEA files as CCS components based on MEA assignments from assign_mea_technology.

    Parameters:
        - input_data_path: Path to the input data directory
        - path_files_technologies: Path to the source technology files directory
        - network_emission_flux: DataFrame with MEA technology assignments (optional)
    """
    # Read topology to get nodes and periods
    with open(input_data_path / "Topology.json", "r") as f:
        topology = json.load(f)

    # Copy technology files for each node in each period
    for period in topology["investment_periods"]:
        for node in topology["nodes"]:
            # Read Technologies.json for this node
            tech_file_path = input_data_path / period / "node_data" / node / "Technologies.json"

            if tech_file_path.exists():
                with open(tech_file_path, "r") as f:
                    technologies_at_node = json.load(f)

                # Get all technology names from Technologies.json
                existing_techs = list(technologies_at_node["existing"].keys()) if technologies_at_node[
                    "existing"] else []

                # Handle "new" being either dict or list
                if isinstance(technologies_at_node["new"], dict):
                    new_techs = list(technologies_at_node["new"].keys())
                elif isinstance(technologies_at_node["new"], list):
                    new_techs = technologies_at_node["new"]
                else:
                    new_techs = []

                all_techs = existing_techs + new_techs

                # Create technology_data directory if it doesn't exist
                tech_data_dir = input_data_path / period / "node_data" / node / "technology_data"
                tech_data_dir.mkdir(parents=True, exist_ok=True)

                # Copy each technology file listed in Technologies.json
                for tech_name in all_techs:
                    source_file = find_technology_file(tech_name, path_files_technologies)
                    if source_file:
                        dest_file = tech_data_dir / f"{tech_name}.json"
                        shutil.copy2(source_file, dest_file)
                        print(f"Copied {tech_name}.json to {dest_file}")
                    else:
                        print(f"Warning: Technology file {tech_name}.json not found in {path_files_technologies}")

                # Copy MEA technologies for this node based on assign_mea_technology results
                if network_emission_flux is not None:
                    node_emission_rows = network_emission_flux[network_emission_flux['node_name'] == node]

                    for _, emission_row in node_emission_rows.iterrows():
                        if emission_row['node_type'] in ['Waste', 'Cement', 'Refining', 'Other']:
                            mea_tech = emission_row.get('mea_technology')
                            if pd.notna(mea_tech):
                                mea_tech_name = Path(mea_tech).stem  # e.g., 'MEA_medium' or 'MEA_large'

                                # Copy the MEA technology file
                                mea_source_file = find_technology_file(mea_tech_name, path_files_technologies)
                                if mea_source_file:
                                    mea_dest_file = tech_data_dir / f"{mea_tech_name}.json"
                                    if not mea_dest_file.exists():  # Only copy if not already copied
                                        shutil.copy2(mea_source_file, mea_dest_file)
                                        print(f"Copied MEA component {mea_tech_name}.json to {mea_dest_file}")
                                else:
                                    print(
                                        f"Warning: MEA technology file {mea_tech_name}.json not found in {path_files_technologies}")

    print("Technology data copying completed.")


def update_emitter_ccs_references(input_data_path, network_emission_flux):
    """
    Updates the CCS references in emitter technology files to match the determined MEA technology size.
    This function ensures that each emitter's CCS section points to the correct MEA technology
    as determined by the assign_mea_technology function.

    Parameters:
        - input_data_path: Path to the input data directory
        - network_emission_flux: DataFrame containing emission data and MEA technology assignments
    """
    # Read topology to get nodes and periods
    with open(input_data_path / "Topology.json", "r") as f:
        topology = json.load(f)

    print("Updating CCS references in emitter technologies...")

    for period in topology["investment_periods"]:
        for node in topology["nodes"]:
            # Get MEA technology assignments for this node
            node_emission_rows = network_emission_flux[network_emission_flux['node_name'] == node]

            # Check each emitter type at this node
            for _, emission_row in node_emission_rows.iterrows():
                node_type = emission_row['node_type']
                if node_type in ['Waste', 'Cement', 'Refining', 'Other']:
                    mea_tech = emission_row.get('mea_technology')
                    if pd.notna(mea_tech):
                        mea_tech_name = Path(mea_tech).stem  # e.g., 'MEA_medium' or 'MEA_large'

                        # Determine the emitter technology file to update
                        emitter_tech_name = None
                        if node_type == "Waste":
                            emitter_tech_name = "WasteToEnergyEmitter"
                        elif node_type == "Cement":
                            emitter_tech_name = "CementEmitter"
                        elif node_type == "Refining":
                            emitter_tech_name = "RefineryEmitter"
                        elif node_type == "Other":
                            emitter_tech_name = "UnspecifiedEmitter"

                        if emitter_tech_name:
                            tech_file_path = input_data_path / period / "node_data" / node / "technology_data" / f"{emitter_tech_name}.json"

                            if tech_file_path.exists():
                                # Read the technology file
                                with open(tech_file_path, 'r') as f:
                                    tech_data = json.load(f)

                                # Ensure CCS section exists and update it
                                if "Performance" not in tech_data:
                                    tech_data["Performance"] = {}

                                if "ccs" not in tech_data["Performance"]:
                                    tech_data["Performance"]["ccs"] = {}

                                # Update the CCS section with the determined MEA technology
                                tech_data["Performance"]["ccs"]["possible"] = 1
                                tech_data["Performance"]["ccs"]["ccs_type"] = mea_tech_name

                                # Set CO2 concentration based on emitter type (if not already set)
                                if "co2_concentration" not in tech_data["Performance"]["ccs"]:
                                    if node_type == "Waste":
                                        tech_data["Performance"]["ccs"]["co2_concentration"] = 0.07
                                    elif node_type == "Cement":
                                        tech_data["Performance"]["ccs"]["co2_concentration"] = 0.20
                                    elif node_type == "Refining":
                                        tech_data["Performance"]["ccs"]["co2_concentration"] = 0.25
                                    elif node_type == "Other":
                                        tech_data["Performance"]["ccs"]["co2_concentration"] = 0.15

                                # Write back the updated file
                                with open(tech_file_path, 'w') as f:
                                    json.dump(tech_data, f, indent=2)

                                print(f"  ✅ Updated {emitter_tech_name} at {node} to use CCS: {mea_tech_name}")
                            else:
                                print(f"  ❌ Technology file not found: {tech_file_path}")

    print("CCS reference updates completed.")


def update_network_distance_matrix(input_data_path, network_data_dict, network_types, decimal_places=2):
    """
    Update distance matrices for multiple network types using network data where values > 0 represent distances.

    FIXED: Ensures proper matrix dimensions and data types to prevent scalar dataspace errors.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary mapping network types to their data (values > 0 are distances, 0 means no connection)
    - network_types: List of network type folder names (e.g., ['CO2_Pipeline', 'CO2Truck', 'CO2Railway', 'CO2Ship'])
    - decimal_places: Number of decimal places to round to (default: 2)
    """
    # Load the template distance CSV (empty matrix with node names)
    template_distance = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "distance.csv",
                                    sep=";", index_col=0)

    print(f"🔍 DEBUG: Template distance matrix shape: {template_distance.shape}")
    print(f"🔍 DEBUG: Template distance matrix columns: {list(template_distance.columns)}")
    print(f"🔍 DEBUG: Template distance matrix index: {list(template_distance.index)}")

    # Process each network type with its corresponding data
    for network_type in network_types:
        # Get the corresponding network data
        if network_type == 'CO2_Pipeline':
            network_data = network_data_dict.get('pipeline')
        elif network_type == 'CO2Truck':
            network_data = network_data_dict.get('truck')
        elif network_type == 'CO2Railway':
            network_data = network_data_dict.get('railway')
        elif network_type == 'CO2Ship':
            network_data = network_data_dict.get('ship')
        else:
            print(f"Warning: No data mapping found for network type {network_type}")
            continue

        if network_data is None:
            print(f"Warning: No data found for network type {network_type}")
            continue

        print(f"🔍 DEBUG: Processing {network_type} - Network data shape: {network_data.shape}")
        print(f"🔍 DEBUG: Network data columns: {list(network_data.columns)}")
        print(f"🔍 DEBUG: Network data index: {list(network_data.index)}")

        # Create updated distance matrix
        updated_distance = template_distance.copy()

        # FIXED: Ensure proper data alignment and matrix format
        # Check if dimensions match
        if template_distance.shape != network_data.shape:
            print(f"❌ WARNING: Shape mismatch for {network_type}")
            print(f"  Template shape: {template_distance.shape}")
            print(f"  Network data shape: {network_data.shape}")

            # Try to align the data by reindexing
            try:
                # Reindex network_data to match template structure
                network_data_aligned = network_data.reindex(index=template_distance.index,
                                                            columns=template_distance.columns,
                                                            fill_value=0.0)
                print(f"  ✅ Successfully aligned network data for {network_type}")
                network_data = network_data_aligned
            except Exception as e:
                print(f"  ❌ Failed to align network data for {network_type}: {e}")
                continue

        # Ensure all values are numeric and handle any string/object types
        try:
            # Convert to float, replacing any non-numeric values with 0
            updated_distance = updated_distance.astype(float)
            network_data_numeric = pd.DataFrame(network_data).astype(float)

            # Update the template matrix with distance values
            updated_distance.iloc[:, :] = network_data_numeric.values

            # Round to specified decimal places
            updated_distance = updated_distance.round(decimal_places)

            # FIXED: Ensure matrix has proper minimum dimensions (at least 1x1)
            if updated_distance.empty or updated_distance.shape[0] == 0 or updated_distance.shape[1] == 0:
                print(f"❌ Error: Empty matrix for {network_type}")
                continue

            print(f"✅ Successfully processed distance matrix for {network_type}: shape {updated_distance.shape}")

        except Exception as e:
            print(f"❌ Error processing distance matrix for {network_type}: {e}")
            continue

        # Save the updated distance matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "distance.csv"

        # FIXED: Ensure proper CSV formatting with explicit parameters
        try:
            updated_distance.to_csv(output_path, sep=";", float_format=f'%.{decimal_places}f',
                                    lineterminator='\n', encoding='utf-8')
            print(f"✅ Successfully saved distance matrix for {network_type} to {output_path}")
        except Exception as e:
            print(f"❌ Error saving distance matrix for {network_type}: {e}")
            continue

    return True


def update_network_connection_matrix(input_data_path, network_data_dict):
    """
    Update connection matrices for multiple network types.
    Values > 0 in network data indicate connection (converted to 1), values = 0 indicate no connection.

    FIXED: Ensures proper matrix dimensions and data types to prevent scalar dataspace errors.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary with keys 'pipeline', 'truck', 'railway', 'ship' containing network data
    """
    # Load the template connection CSV (empty matrix with node names)
    template_connection = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "connection.csv",
                                      sep=";", index_col=0)

    print(f"🔍 DEBUG: Template connection matrix shape: {template_connection.shape}")

    # Mapping from network data keys to network type folders
    network_type_mapping = {
        'pipeline': 'CO2_Pipeline',
        'truck': 'CO2Truck',
        'railway': 'CO2Railway',
        'ship': 'CO2Ship'
    }

    # Process each network data type
    for data_key, network_type in network_type_mapping.items():
        if data_key not in network_data_dict:
            print(f"Warning: {data_key} data not found in network_data_dict")
            continue

        network_data = network_data_dict[data_key]
        print(f"🔍 DEBUG: Processing connection matrix for {network_type}")

        # Create updated connection matrix for this network type
        updated_connection = template_connection.copy()

        # FIXED: Ensure proper data alignment and matrix format
        try:
            # Check if dimensions match
            if template_connection.shape != network_data.shape:
                print(f"❌ WARNING: Shape mismatch for {network_type} connection matrix")
                # Try to align the data
                network_data_aligned = network_data.reindex(index=template_connection.index,
                                                            columns=template_connection.columns,
                                                            fill_value=0.0)
                network_data = network_data_aligned

            # Convert network data to connection matrix: values > 0 become 1, values = 0 stay 0
            # Ensure numeric data type first
            network_data_numeric = pd.DataFrame(network_data).astype(float)
            connection_values = (network_data_numeric.values > 0).astype(int)
            updated_connection.iloc[:, :] = connection_values

            # FIXED: Ensure matrix has proper minimum dimensions
            if updated_connection.empty or updated_connection.shape[0] == 0 or updated_connection.shape[1] == 0:
                print(f"❌ Error: Empty connection matrix for {network_type}")
                continue

            print(f"✅ Successfully processed connection matrix for {network_type}: shape {updated_connection.shape}")

        except Exception as e:
            print(f"❌ Error processing connection matrix for {network_type}: {e}")
            continue

        # Save the updated connection matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "connection.csv"

        # FIXED: Ensure proper CSV formatting
        try:
            updated_connection.to_csv(output_path, sep=";", lineterminator='\n', encoding='utf-8')
            print(f"✅ Successfully saved connection matrix for {network_type} to {output_path}")
        except Exception as e:
            print(f"❌ Error saving connection matrix for {network_type}: {e}")
            continue

    return True


def update_network_size_max_arcs(input_data_path, network_data_dict, max_transport_capacity):
    """
    Update size_max_arcs matrices using a predefined transport capacity value.
    Uses the max_transport_capacity parameter from scenario parameterization.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary with keys 'pipeline', 'truck', 'railway', 'ship' containing network data
    - max_transport_capacity: Predefined maximum transport capacity in tonnes/hour
    """
    # Load the template size_max_arcs CSV
    template_size_max = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
                                    sep=";", index_col=0)

    print(f"🔍 PREDEFINED Network sizing:")
    print(f"  Using predefined transport capacity: {max_transport_capacity:.2f} tonnes/hour")
    print(f"  Applied to all connected arcs in all network types")

    # Mapping from network data keys to network type folders
    network_type_mapping = {
        'pipeline': 'CO2_Pipeline',
        'truck': 'CO2Truck',
        'railway': 'CO2Railway',
        'ship': 'CO2Ship'
    }

    # Process each network data type with the same predefined capacity
    for data_key, network_type in network_type_mapping.items():
        if data_key not in network_data_dict:
            print(f"Warning: {data_key} data not found in network_data_dict")
            continue

        network_data = network_data_dict[data_key]

        print(f"🔍 Processing {network_type} with capacity {max_transport_capacity:.2f} tonnes/hour")

        # Create updated size_max_arcs matrix for this network type
        updated_size_max = template_size_max.copy()

        try:
            # Ensure proper data alignment
            if template_size_max.shape != network_data.shape:
                print(f"❌ WARNING: Shape mismatch for {network_type} size_max matrix")
                network_data_aligned = network_data.reindex(index=template_size_max.index,
                                                            columns=template_size_max.columns,
                                                            fill_value=0.0)
                network_data = network_data_aligned

            # Ensure numeric data type
            network_data_numeric = pd.DataFrame(network_data).astype(float)

            # Create connection matrix: values > 0 become 1, values = 0 stay 0
            connection_matrix = (network_data_numeric.values > 0).astype(int)

            # Create size_max_arcs matrix: connection_matrix * predefined capacity
            size_max_values = connection_matrix * max_transport_capacity

            # Update the template matrix
            updated_size_max = updated_size_max.astype(float)
            updated_size_max.iloc[:, :] = size_max_values

            # Count connections for this network type
            num_connections = np.count_nonzero(connection_matrix)
            print(f"   {num_connections} connections found with capacity {max_transport_capacity:.2f} tonnes/hour each")

            # Ensure matrix has proper minimum dimensions
            if updated_size_max.empty or updated_size_max.shape[0] == 0 or updated_size_max.shape[1] == 0:
                print(f"❌ Error: Empty size_max matrix for {network_type}")
                continue

            print(f"✅ Successfully processed size_max matrix for {network_type}: shape {updated_size_max.shape}")

        except Exception as e:
            print(f"❌ Error processing size_max matrix for {network_type}: {e}")
            continue

        # Save the updated size_max_arcs matrix to the specific network type folder
        output_path = input_data_path / "period1" / "network_topology" / "new" / network_type / "size_max_arcs.csv"

        try:
            updated_size_max.to_csv(output_path, sep=";", float_format='%.2f',
                                    lineterminator='\n', encoding='utf-8')
            print(f"✅ Successfully saved size_max matrix for {network_type} to {output_path}")
        except Exception as e:
            print(f"❌ Error saving size_max matrix for {network_type}: {e}")
            continue

    print(f"\n✅ PREDEFINED: All networks use {max_transport_capacity:.2f} tonnes/hour capacity")
    print(f"   - Pipeline arcs: {max_transport_capacity:.2f} t/h")
    print(f"   - Truck arcs: {max_transport_capacity:.2f} t/h")
    print(f"   - Railway arcs: {max_transport_capacity:.2f} t/h")

    return True




def compute_opex_var_arcs(path_node_metrics: Path, path_output_root: Path) -> None:
    """
    Reads truck and railway distance matrices from node_metrics.xlsx and writes
    opex_var_arcs.csv to the respective CO2Truck and CO2Railway output folders.

    Formulas are taken from Ouvray 2024:
        Truck:   opex_var = (5.58  / distance + 0.15) * distance
        Railway: opex_var = (28.9  / distance + 0.07) * distance

    Zero-distance entries (no arc / same node) are left as 0.

    Parameters
    ----------
    path_node_metrics : Path
        Full path to node_metrics.xlsx.
    path_output_root : Path
        Root folder that contains CO2Truck/ and CO2Railway/ sub-folders.
    """

    def _opex_truck(d: float) -> float:
        return (5.58 / d + 0.15) * d if d > 0 else 0.0

    def _opex_railway(d: float) -> float:
        return (28.9 / d + 0.07) * d if d > 0 else 0.0

    truck_dist = pd.read_excel(path_node_metrics, sheet_name="truck", index_col=0)
    rail_dist  = pd.read_excel(path_node_metrics, sheet_name="railway", index_col=0)

    truck_opex = truck_dist.applymap(_opex_truck)
    rail_opex  = rail_dist.applymap(_opex_railway)

    path_truck_out   = path_output_root / "CO2Truck"   / "opex_var_arcs.csv"
    path_railway_out = path_output_root / "CO2Railway" / "opex_var_arcs.csv"

    path_truck_out.parent.mkdir(parents=True, exist_ok=True)
    path_railway_out.parent.mkdir(parents=True, exist_ok=True)

    truck_opex.to_csv(path_truck_out)
    rail_opex.to_csv(path_railway_out)

    print(f"Saved truck   opex_var_arcs -> {path_truck_out}")
    print(f"Saved railway opex_var_arcs -> {path_railway_out}")
def process_gamma_sheets_to_csv(path_files_network_capex, input_data_path, network_location_df,
                                transport_mode="pipeline"):
    """
    Process gamma sheets from capex_defined_per_arc_{transport_mode}.xlsx and save them as separate CSV files
    in the CO2_{Transport_Mode} folder. Maps node_id (from Excel) to node_name (for topology).

    Parameters:
        - path_files_network_capex: Path to the network capex metrics directory
        - input_data_path: Path to the input data directory
        - network_location_df: DataFrame containing node_id to node_name mapping
        - transport_mode: Transport mode ("pipeline", "truck", "railway")

    Returns:
        - gamma_data_dict: Dictionary containing the gamma data for potential future use
    """

    # Validate transport mode
    valid_modes = ["pipeline"]
    if transport_mode.lower() not in valid_modes:
        raise ValueError(f"transport_mode must be one of {valid_modes}, got: {transport_mode}")

    transport_mode = transport_mode.lower()

    # Define the path to the capex file based on transport mode
    capex_file_path = path_files_network_capex / f"gamma_defined_per_arc_{transport_mode}.xlsx"

    # Check if the file exists
    if not capex_file_path.exists():
        print(f"Warning: gamma_defined_per_arc_{transport_mode}.xlsx not found at {capex_file_path}")
        return None

    # Create node_id to node_name mapping from network_location_df
    # Assuming network_location_df has node_id as index and 'node_name' as column
    if 'node_name' not in network_location_df.columns:
        raise ValueError("network_location_df must have a 'node_name' column")

    # Create the mapping dictionary: node_id -> node_name
    node_id_to_name = network_location_df['node_name'].to_dict()

    print(f"Node ID to Name mapping for {transport_mode.upper()}:")
    for node_id, node_name in node_id_to_name.items():
        print(f"  {node_id} -> {node_name}")

    # Read topology nodes for validation
    with open(input_data_path / "Topology.json", "r") as f:
        topology = json.load(f)
    expected_nodes = set(topology["nodes"])

    # Define the output directory based on transport mode
    transport_mode_mapping = {
        "pipeline": "CO2_Pipeline",
        "truck": "CO2Truck",
        "railway": "CO2Railway"
    }

    output_dir_name = transport_mode_mapping[transport_mode]
    output_dir = input_data_path / "period1" / "network_topology" / "new" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dictionary to store gamma data
    gamma_data_dict = {}

    # List of gamma sheets to process
    gamma_sheets = ['gamma1', 'gamma2', 'gamma3', 'gamma4']

    print(f"\nProcessing {transport_mode.upper()} gamma sheets from {capex_file_path}...")

    # Process each gamma sheet
    for sheet_name in gamma_sheets:
        try:
            # Read the gamma sheet from Excel (indexed by node_id)
            gamma_df = pd.read_excel(capex_file_path, sheet_name=sheet_name, index_col=0)

            print(f"  Original {sheet_name} shape: {gamma_df.shape}")
            print(f"  Original columns (node_ids): {list(gamma_df.columns)}")
            print(f"  Original index (node_ids): {list(gamma_df.index)}")

            # Map column names from node_id to node_name
            new_columns = []
            missing_node_ids_cols = []
            for node_id in gamma_df.columns:
                if node_id in node_id_to_name:
                    new_columns.append(node_id_to_name[node_id])
                else:
                    new_columns.append(node_id)  # Keep original if mapping not found
                    missing_node_ids_cols.append(node_id)

            # Map index names from node_id to node_name
            new_index = []
            missing_node_ids_idx = []
            for node_id in gamma_df.index:
                if node_id in node_id_to_name:
                    new_index.append(node_id_to_name[node_id])
                else:
                    new_index.append(node_id)  # Keep original if mapping not found
                    missing_node_ids_idx.append(node_id)

            # Update the DataFrame with node_names
            gamma_df.columns = new_columns
            gamma_df.index = new_index

            if missing_node_ids_cols:
                print(f"  ⚠️  Warning: Could not map these column node_ids: {missing_node_ids_cols}")
            if missing_node_ids_idx:
                print(f"  ⚠️  Warning: Could not map these index node_ids: {missing_node_ids_idx}")

            print(f"  Mapped columns (node_names): {list(gamma_df.columns)}")
            print(f"  Mapped index (node_names): {list(gamma_df.index)}")

            # Validate that all expected nodes are present
            gamma_nodes_cols = set(gamma_df.columns)
            gamma_nodes_idx = set(gamma_df.index)

            missing_cols = expected_nodes - gamma_nodes_cols
            missing_idx = expected_nodes - gamma_nodes_idx

            if missing_cols:
                print(f"  ❌ Warning: Missing nodes in {sheet_name} columns: {missing_cols}")
            if missing_idx:
                print(f"  ❌ Warning: Missing nodes in {sheet_name} index: {missing_idx}")

            if not missing_cols and not missing_idx:
                print(f"  ✅ All topology nodes found in {sheet_name}")

            # Store in dictionary
            gamma_data_dict[sheet_name] = gamma_df

            # Define output CSV path
            csv_output_path = output_dir / f"{sheet_name}.csv"

            # Round numeric values to 4 decimal places before saving
            gamma_df_formatted = gamma_df.round(4)

            # Save as CSV with semicolon separator (consistent with other files)
            # Use float_format to ensure 4 decimal places are shown
            gamma_df_formatted.to_csv(csv_output_path, sep=";", float_format="%.4f")

            print(f"✅ Successfully processed {sheet_name}: {gamma_df.shape} -> {csv_output_path}")

        except Exception as e:
            print(f"❌ Error processing sheet '{sheet_name}': {e}")
            continue

    # Summary
    successful_sheets = len(gamma_data_dict)
    print(f"\n{transport_mode.upper()} Gamma sheets processing completed:")
    print(f"  - Successfully processed: {successful_sheets}/{len(gamma_sheets)} sheets")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Files created: {', '.join([f'{sheet}.csv' for sheet in gamma_data_dict.keys()])}")

    return gamma_data_dict


def process_all_transport_modes_gamma(path_files_network_capex, input_data_path, network_location_df):
    """
    Process gamma sheets for all transport modes (pipeline, truck, railway).

    Parameters:
        - path_files_network_capex: Path to the network capex metrics directory
        - input_data_path: Path to the input data directory
        - network_location_df: DataFrame containing node_id to node_name mapping

    Returns:
        - all_gamma_data: Dictionary with transport modes as keys and gamma data as values
    """

    transport_modes = ["pipeline", "truck", "railway"]
    all_gamma_data = {}

    print("=" * 80)
    print("PROCESSING GAMMA SHEETS FOR ALL TRANSPORT MODES")
    print("=" * 80)

    for mode in transport_modes:
        print(f"\n{'=' * 20} PROCESSING {mode.upper()} {'=' * 20}")
        gamma_data = process_gamma_sheets_to_csv(
            path_files_network_capex,
            input_data_path,
            network_location_df,
            transport_mode=mode
        )

        if gamma_data is not None:
            all_gamma_data[mode] = gamma_data
            print(f"✅ {mode.upper()} gamma processing completed successfully")
        else:
            print(f"❌ {mode.upper()} gamma processing failed")

    print(f"\n{'=' * 80}")
    print("SUMMARY OF ALL TRANSPORT MODES")
    print(f"{'=' * 80}")
    successful_modes = len(all_gamma_data)
    print(f"Successfully processed: {successful_modes}/{len(transport_modes)} transport modes")
    print(f"Processed modes: {list(all_gamma_data.keys())}")

    return all_gamma_data


def load_climate_data_from_api_robust(folder_path: str | Path, dataset: str = "JRC"):
    """
    Reads in climate data for a full year from a folder containing node data and writes it to the respective file.
    Enhanced to handle offshore nodes and other API failures gracefully.

    Parameters:
    - folder_path: Path to the folder containing node data and NodeLocations.csv
    - dataset: Dataset to import from, can be JRC (only onshore)

    Returns:
    - Tuple of (successful_nodes, failed_nodes, offshore_nodes)
    """
    # Convert to Path
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    # Import inside function to avoid import issues
    from adopt_net0.data_preprocessing.data_loading import import_jrc_climate_data

    # Read NodeLocations.csv with node column as index
    node_locations_path = os.path.join(folder_path, "NodeLocations.csv")
    node_locations_df = pd.read_csv(
        node_locations_path, sep=";", names=["node", "lon", "lat", "alt"], header=0
    )

    if node_locations_df.isnull().values.any():
        raise Exception("Please specify longitude, latitude and altitude for each node")

    # Read nodes and investment_periods from the JSON file
    json_file_path = os.path.join(folder_path, "Topology.json")
    with open(json_file_path, "r") as json_file:
        topology = json.load(json_file)

    year = (
        int(topology["start_date"].split("-")[0])
        if topology["start_date"]
        else "typical_year"
    )

    failed_nodes = []
    successful_nodes = []
    offshore_nodes = []

    for period in topology["investment_periods"]:
        for node_name in topology["nodes"]:
            # Read lon, lat, and alt for this node name from node_locations_df
            node_data = node_locations_df[node_locations_df["node"] == node_name]
            lon = node_data["lon"].values[0]
            lat = node_data["lat"].values[0]
            alt = node_data["alt"].values[0]

            if dataset == "JRC":
                try:
                    print(f"Importing Climate Data for {node_name}...")
                    # Fetch climate data for the node
                    data = import_jrc_climate_data(lon, lat, year, alt)
                    print(f"Importing Climate Data for {node_name} successful")
                    successful_nodes.append(node_name)
                except Exception as e:
                    error_msg = str(e)
                    print(f"Failed to import climate data for {node_name}: {e}")

                    # Check if it's likely an offshore location
                    if "400" in error_msg or "offshore" in error_msg.lower():
                        print(f"  -> {node_name} appears to be offshore (coordinates: {lon}, {lat})")
                        offshore_nodes.append(node_name)
                    else:
                        print(f"  -> Other API issue for {node_name}")

                    failed_nodes.append(node_name)
                    continue
            else:
                raise Exception("Other APIs are not available")

            # Write data to CSV file
            output_folder = os.path.join(folder_path, period, "node_data", node_name)
            output_file = os.path.join(output_folder, "ClimateData.csv")
            existing_data = pd.read_csv(output_file, sep=";")

            # Fill in existing data with data from the fetched DataFrame based on column names
            for column, value in data["dataframe"].items():
                existing_data[column] = value.values[: len(existing_data)]

            # Save the updated data back to ClimateData.csv
            existing_data.to_csv(output_file, index=False, sep=";")

    # Enhanced summary
    print(f"\nSummary:")
    print(f"Successfully processed {len(successful_nodes)} nodes: {successful_nodes}")
    if offshore_nodes:
        print(f"Failed to process {len(failed_nodes)} nodes: {failed_nodes}")
        print(f"\n💡 Offshore nodes detected: {offshore_nodes}")
        print(f"   This is expected behavior for offshore locations.")
        print(f"   The optimization can proceed without climate data for these nodes.")

    return successful_nodes, failed_nodes, offshore_nodes


def update_carrier_data(input_data_path, electricity_price_data, network_emission_flux,
                        path_files_technologies, node_names, co2_intensity_electricity,
                        cop_hp, path_files_node_flux,
                        electricity_import_limit=100, heat_import_limit=200):

    import adopt_net0 as adopt

    co2_intensity_heat = round(co2_intensity_electricity / cop_hp, 4)

    # --- Import limits ---
    adopt.fill_carrier_data(input_data_path, value_or_data=electricity_import_limit,
                            columns=['Import limit'], carriers=['electricity'], nodes=node_names)
    adopt.fill_carrier_data(input_data_path, value_or_data=heat_import_limit,
                            columns=['Import limit'], carriers=['heat'], nodes=node_names)

    # --- Emission factors ---
    adopt.fill_carrier_data(input_data_path, value_or_data=co2_intensity_electricity,
                            columns=['Import emission factor'], carriers=['electricity'], nodes=node_names)
    adopt.fill_carrier_data(input_data_path, value_or_data=co2_intensity_heat,
                            columns=['Import emission factor'], carriers=['heat'], nodes=node_names)

    # --- Electricity & heat prices ---
    electricity_prices = electricity_price_data['Day-ahead Price (EUR/MWh)'].values
    heat_prices = np.round(electricity_prices / cop_hp, 2)

    adopt.fill_carrier_data(input_data_path, value_or_data=electricity_prices,
                            columns=['Import price'], carriers=['electricity'], nodes=node_names)
    adopt.fill_carrier_data(input_data_path, value_or_data=heat_prices,
                            columns=['Import price'], carriers=['heat'], nodes=node_names)

    # --- Load emission factors from technology JSON files ---
    node_type_mapping = {
        'Waste':    ('Emitter/WasteToEnergyEmitter.json', 'waste'),
        'Cement':   ('Emitter/CementEmitter.json',        'cement'),
        'Refining': ('Emitter/RefineryEmitter.json',      'refined_product'),
        'Other':    ('Emitter/UnspecifiedEmitter.json',   'industrial_product'),
    }

    emission_factors = {}
    for node_type, (filename, _) in node_type_mapping.items():
        tech_file_path = path_files_technologies / filename
        try:
            with open(tech_file_path, 'r') as f:
                emission_factors[node_type] = json.load(f)['Performance']['emission_factor']
            print(f"✅ Loaded emission factor for {node_type}: {emission_factors[node_type]}")
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: could not load emission factor for {node_type} ({e}), defaulting to 1.0")
            emission_factors[node_type] = 1.0

    # --- Load hourly profiles from both real_data and synthetic_data sheets ---
    excel_file_path = path_files_node_flux / "emission_profile_emitters.xlsx"

    profiles_real      = None
    profiles_synthetic = None

    if excel_file_path.exists():
        xl = pd.ExcelFile(excel_file_path)
        if "real_data" in xl.sheet_names:
            profiles_real = pd.read_excel(xl, sheet_name="real_data")
        if "synthetic_data" in xl.sheet_names:
            profiles_synthetic = pd.read_excel(xl, sheet_name="synthetic_data")

    def get_profile(node_type: str, node_name: str) -> tuple[np.ndarray | None, str]:
        """
        Look up the hourly profile for a node.
        Priority: real_data sheet first, then synthetic_data sheet.
        Returns (array_8760, source_label) or (None, 'missing').
        """
        col = f"{node_type} - {node_name}"
        for df, label in [(profiles_real, "real_data"), (profiles_synthetic, "synthetic_data")]:
            if df is not None and col in df.columns:
                arr = df[col].head(8760).values.astype(float)
                if len(arr) != 8760:
                    raise ValueError(f"Expected 8760 rows for '{col}' in {label}, got {len(arr)}")
                if np.any(np.isnan(arr)):
                    raise ValueError(f"NaN values in profile '{col}' in {label}")
                return arr, label
        return None, "missing"

    # --- Process demands per node ---
    node_demands_hourly = {}
    node_demands_annual = {}
    nodes_with_real_data, nodes_with_synthetic_data, nodes_missing_profile = [], [], []

    for _, row in network_emission_flux.iterrows():
        node_name       = row['node_name']
        node_type       = row['node_type']
        annual_emission = row['annual_emission']

        if node_type in ('Storage', 'Transport') or annual_emission == 0:
            continue
        if node_type not in node_type_mapping:
            continue

        _, carrier_name = node_type_mapping[node_type]
        annual_demand_tonnes = round(annual_emission / emission_factors[node_type] / 1000.0, 2)

        hourly_demand_array, source = get_profile(node_type, node_name)

        if source == "real_data":
            annual_demand_tonnes = round(hourly_demand_array.sum(), 2)
            nodes_with_real_data.append(f"{node_name} ({node_type})")
            print(f"  ✅ Real profile      | {node_name} ({carrier_name}): {annual_demand_tonnes:.2f} t/yr")

        elif source == "synthetic_data":
            annual_demand_tonnes = round(hourly_demand_array.sum(), 2)
            nodes_with_synthetic_data.append(f"{node_name} ({node_type})")
            print(f"  🔧 Synthetic profile | {node_name} ({carrier_name}): {annual_demand_tonnes:.2f} t/yr")

        else:
            # No profile found in either sheet — fall back to flat and warn
            hourly_rate = round(annual_demand_tonnes / 8760.0, 6)
            hourly_demand_array = np.full(8760, hourly_rate)
            nodes_missing_profile.append(f"{node_name} ({node_type})")
            print(f"  ⚠️  No profile found  | {node_name} ({carrier_name}): falling back to flat {hourly_rate:.4f} t/hr")

        node_demands_hourly.setdefault(node_name, {}).setdefault(carrier_name, np.zeros(8760))
        node_demands_hourly[node_name][carrier_name] += hourly_demand_array

        node_demands_annual.setdefault(node_name, {}).setdefault(carrier_name, 0)
        node_demands_annual[node_name][carrier_name] = round(
            node_demands_annual[node_name][carrier_name] + annual_demand_tonnes, 2)

    # --- Apply demands ---
    for node_name, carriers in node_demands_hourly.items():
        for carrier_name, hourly_array in carriers.items():
            adopt.fill_carrier_data(input_data_path, value_or_data=hourly_array,
                                    columns=['Demand'], carriers=[carrier_name], nodes=[node_name])

    # --- Summary ---
    print(f"\n📊 CARRIER DATA UPDATE SUMMARY:")
    print(f"  Electricity import limit : {electricity_import_limit}")
    print(f"  Heat import limit        : {heat_import_limit}")
    print(f"  Elec. emission factor    : {co2_intensity_electricity} kg CO2/kWh")
    print(f"  Heat emission factor     : {co2_intensity_heat:.4f} kg CO2/kWh")
    print(f"  Electricity price points : {len(electricity_prices)}")
    print(f"  Nodes with real profiles      ({len(nodes_with_real_data):>2}): {nodes_with_real_data}")
    print(f"  Nodes with synthetic profiles ({len(nodes_with_synthetic_data):>2}): {nodes_with_synthetic_data}")
    if nodes_missing_profile:
        print(f"  ⚠️  Nodes with NO profile     ({len(nodes_missing_profile):>2}): {nodes_missing_profile}")

    if node_demands_annual:
        print(f"\nAnnual demand by node (tonnes/year):")
        for node_name, carriers in node_demands_annual.items():
            details = ', '.join(f"{c}: {d:.2f}t/yr" for c, d in carriers.items())
            print(f"  {node_name}: {details} (Total: {sum(carriers.values()):.2f}t/yr)")

    return True

# Add these debug enhancements to your utility functions:

# ===== Enhanced assign_ccs_technologies function with debugging =====
def assign_ccs_technologies_debug(network_location, network_emission_flux, path_data_case_study, input_data_path):
    """
    Enhanced version with comprehensive debugging
    """
    print("\n🔍 DEBUG: Starting assign_ccs_technologies with enhanced debugging...")

    # Debug input data
    print(f"🔍 Input data summary:")
    print(f"  network_location shape: {network_location.shape}")
    print(f"  network_emission_flux shape: {network_emission_flux.shape}")

    # Check for capacity column
    if 'emitter_capacity' not in network_emission_flux.columns:
        raise ValueError("'emitter_capacity' column not found. Please run calculate_emitter_capacities() first.")

    unique_nodes = network_location['node_name'].unique()
    print(f"  unique_nodes: {list(unique_nodes)}")

    for node_name in unique_nodes:
        print(f"\n🔍 Processing node: {node_name}")

        # Get all rows for this node
        node_rows = network_location[network_location['node_name'] == node_name]
        print(f"  node_location rows: {len(node_rows)}")

        existing_techs_dict = {}
        new_techs_list = []

        for idx, row in node_rows.iterrows():
            node_type = row['node_type']
            print(f"    Processing row {idx}: type = {node_type}")

            if node_type == "Storage":
                new_techs_list.append("PermanentStorage_CO2_simple")
                print(f"      Added storage technology")

            elif node_type == "Transport":
                print(f"      Transport node - no technologies")
                pass

            else:  # Emitter nodes
                emitter_row = network_emission_flux[
                    (network_emission_flux['node_name'] == node_name) &
                    (network_emission_flux['node_type'] == node_type)
                    ]

                print(f"      Emitter rows found: {len(emitter_row)}")

                if not emitter_row.empty:
                    capacity = float(emitter_row['emitter_capacity'].iloc[0])
                    print(f"      Capacity: {capacity} (type: {type(capacity)})")
                else:
                    capacity = 0.0
                    print(f"      Warning: No capacity data found")

                if node_type == "Waste":
                    existing_techs_dict["WasteToEnergyEmitter"] = capacity
                elif node_type == "Cement":
                    existing_techs_dict["CementEmitter"] = capacity
                elif node_type == "Refining":
                    existing_techs_dict["RefineryEmitter"] = capacity
                elif node_type == "Other":
                    existing_techs_dict["UnspecifiedEmitter"] = capacity

        # Data type validation and conversion
        print(f"  Before cleaning - existing: {existing_techs_dict}")
        print(f"  Before cleaning - new: {new_techs_list}")

        # Clean data types
        existing_techs_clean = {}
        for tech_name, capacity in existing_techs_dict.items():
            clean_name = str(tech_name)
            clean_capacity = float(capacity)
            existing_techs_clean[clean_name] = clean_capacity
            print(f"    Cleaned: {clean_name} = {clean_capacity} ({type(clean_capacity)})")

        new_techs_clean = [str(tech) for tech in set(new_techs_list)]
        print(f"    New techs cleaned: {new_techs_clean}")

        # Create final technologies dictionary
        technologies = {
            "existing": existing_techs_clean,
            "new": new_techs_clean,
        }

        print(f"  Final technologies dict:")
        print(f"    existing: {technologies['existing']}")
        print(f"    new: {technologies['new']}")

        # Write to file with validation
        tech_file_path = input_data_path / "period1" / "node_data" / node_name / "Technologies.json"
        print(f"  Writing to: {tech_file_path}")

        try:
            # Test JSON serialization first
            test_json = json.dumps(technologies, indent=2, ensure_ascii=False)
            print(f"  JSON serialization test: SUCCESS (length: {len(test_json)})")

            # Write to file
            with open(tech_file_path, "w", encoding='utf-8') as json_file:
                json.dump(technologies, json_file, indent=4, ensure_ascii=False)

            # Verify by reading back
            with open(tech_file_path, "r", encoding='utf-8') as json_file:
                read_back = json.load(json_file)

            print(f"  Read back verification:")
            print(f"    existing: {read_back.get('existing', 'NOT_FOUND')}")
            print(f"    new: {read_back.get('new', 'NOT_FOUND')}")

        except Exception as e:
            print(f"  ❌ Error writing/reading Technologies.json: {e}")
            print(f"  ❌ Error type: {type(e)}")
            import traceback
            traceback.print_exc()

    print("\n🔍 assign_ccs_technologies debugging completed")


# ===== Enhanced network matrix functions with debugging =====
def update_network_distance_matrix_debug(input_data_path, network_data_dict, network_types, decimal_places=2):
    """
    Enhanced version with comprehensive debugging
    """
    print("\n🔍 DEBUG: Starting distance matrix update with enhanced debugging...")

    # Load and inspect template
    template_path = input_data_path / "period1" / "network_topology" / "new" / "distance.csv"
    print(f"🔍 Template path: {template_path}")

    try:
        template_distance = pd.read_csv(template_path, sep=";", index_col=0)
        print(f"🔍 Template loaded successfully:")
        print(f"  Shape: {template_distance.shape}")
        print(f"  Index: {list(template_distance.index)}")
        print(f"  Columns: {list(template_distance.columns)}")
        print(f"  Dtypes: {template_distance.dtypes.unique()}")
        print(f"  Sample values: {template_distance.iloc[0:2, 0:2].values}")
    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return False

    # Process each network type
    for network_type in network_types:
        print(f"\n🔍 Processing network type: {network_type}")

        # Get network data
        data_mapping = {
            'CO2_Pipeline': 'pipeline',
            'CO2Truck': 'truck',
            'CO2Railway': 'railway',
            'CO2Ship': 'ship'
        }

        data_key = data_mapping.get(network_type)
        if not data_key:
            print(f"❌ No mapping for {network_type}")
            continue

        network_data = network_data_dict.get(data_key)
        if network_data is None:
            print(f"❌ No data for {data_key}")
            continue

        print(f"🔍 Network data inspection:")
        print(f"  Shape: {network_data.shape}")
        print(f"  Index: {list(network_data.index)}")
        print(f"  Columns: {list(network_data.columns)}")
        print(f"  Dtypes: {network_data.dtypes.unique()}")
        print(f"  Sample values: {network_data.iloc[0:2, 0:2].values}")
        print(f"  Contains NaN: {network_data.isna().any().any()}")
        print(f"  Contains inf: {np.isinf(network_data.select_dtypes(include=[np.number])).any().any()}")

        # Check for data alignment issues
        if template_distance.shape != network_data.shape:
            print(f"⚠️  Shape mismatch: template {template_distance.shape} vs data {network_data.shape}")

            # Check index/column differences
            template_nodes = set(template_distance.index)
            data_nodes = set(network_data.index)
            missing_in_data = template_nodes - data_nodes
            extra_in_data = data_nodes - template_nodes

            if missing_in_data:
                print(f"  Missing in data: {missing_in_data}")
            if extra_in_data:
                print(f"  Extra in data: {extra_in_data}")

        # Create output directory and save
        output_dir = input_data_path / "period1" / "network_topology" / "new" / network_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "distance.csv"

        try:
            # Process and save matrix
            updated_distance = template_distance.copy().astype(float)

            # Align data if needed
            if template_distance.shape != network_data.shape:
                network_data_aligned = network_data.reindex(
                    index=template_distance.index,
                    columns=template_distance.columns,
                    fill_value=0.0
                )
                print(f"  Data aligned from {network_data.shape} to {network_data_aligned.shape}")
                network_data = network_data_aligned

            # Update values
            network_data_numeric = pd.DataFrame(network_data).astype(float)
            updated_distance.iloc[:, :] = network_data_numeric.values
            updated_distance = updated_distance.round(decimal_places)

            print(f"🔍 Final matrix for {network_type}:")
            print(f"  Shape: {updated_distance.shape}")
            print(f"  Dtypes: {updated_distance.dtypes.unique()}")
            print(f"  Sample values: {updated_distance.iloc[0:2, 0:2].values}")

            # Save with explicit formatting
            updated_distance.to_csv(
                output_path,
                sep=";",
                float_format=f'%.{decimal_places}f',
                lineterminator='\n',
                encoding='utf-8'
            )

            # Verify saved file
            verification = pd.read_csv(output_path, sep=";", index_col=0)
            print(f"  Verification - saved file shape: {verification.shape}")
            print(f"  Verification - saved file dtypes: {verification.dtypes.unique()}")

        except Exception as e:
            print(f"❌ Error processing {network_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n🔍 Distance matrix debugging completed")
    return True


# ===== Add this debug function to check raw network data =====
def debug_raw_network_data(network_data_dict):
    """
    Debug the raw network data before processing
    """
    print("\n🔍 DEBUG: Raw network data inspection...")

    for data_type, data in network_data_dict.items():
        print(f"\n📊 {data_type.upper()} network data:")
        print(f"  Type: {type(data)}")
        print(f"  Shape: {data.shape}")
        print(f"  Index type: {type(data.index[0]) if len(data.index) > 0 else 'Empty'}")
        print(f"  Column type: {type(data.columns[0]) if len(data.columns) > 0 else 'Empty'}")
        print(f"  Index: {list(data.index)}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Data types: {data.dtypes.unique()}")

        # Check for problematic values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"  Min value: {data[numeric_cols].min().min()}")
            print(f"  Max value: {data[numeric_cols].max().max()}")
            print(f"  Contains NaN: {data[numeric_cols].isna().any().any()}")
            print(f"  Contains inf: {np.isinf(data[numeric_cols]).any().any()}")

        # Show sample data
        print(f"  Sample data (top-left 3x3):")
        print(data.iloc[0:3, 0:3])

        # Check for non-numeric data in supposedly numeric columns
        object_cols = data.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"  ⚠️  Object columns: {list(object_cols)}")
            for col in object_cols:
                unique_vals = data[col].unique()[:5]  # First 5 unique values
                print(f"    {col} unique values: {unique_vals}")


# ===== Enhanced copy_technology_data_custom with debugging =====
def copy_technology_data_custom_debug(input_data_path, path_files_technologies, network_emission_flux=None):
    """
    Enhanced version with comprehensive debugging
    """
    print("\n🔍 DEBUG: Starting technology file copying with enhanced debugging...")

    # Read topology
    with open(input_data_path / "Topology.json", "r") as f:
        topology = json.load(f)

    print(f"🔍 Topology loaded: {len(topology['nodes'])} nodes, {len(topology['investment_periods'])} periods")

    # Copy technology files for each node in each period
    for period in topology["investment_periods"]:
        print(f"\n📁 Processing period: {period}")

        for node in topology["nodes"]:
            print(f"\n🔧 Processing node: {node}")

            # Read Technologies.json for this node
            tech_file_path = input_data_path / period / "node_data" / node / "Technologies.json"

            if not tech_file_path.exists():
                print(f"  ❌ Technologies.json not found: {tech_file_path}")
                continue

            try:
                with open(tech_file_path, "r") as f:
                    technologies_at_node = json.load(f)

                print(f"  📄 Technologies.json loaded successfully")
                print(f"    existing: {technologies_at_node.get('existing', 'NOT_FOUND')}")
                print(f"    new: {technologies_at_node.get('new', 'NOT_FOUND')}")

            except Exception as e:
                print(f"  ❌ Error reading Technologies.json: {e}")
                continue

            # Get technology lists
            existing_techs = list(technologies_at_node["existing"].keys()) if technologies_at_node["existing"] else []

            if isinstance(technologies_at_node["new"], dict):
                new_techs = list(technologies_at_node["new"].keys())
            elif isinstance(technologies_at_node["new"], list):
                new_techs = technologies_at_node["new"]
            else:
                new_techs = []

            all_techs = existing_techs + new_techs
            print(f"    Total technologies to copy: {len(all_techs)} - {all_techs}")

            # Create technology_data directory
            tech_data_dir = input_data_path / period / "node_data" / node / "technology_data"
            tech_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"    Technology data directory: {tech_data_dir}")

            # Copy each technology file
            for tech_name in all_techs:
                print(f"      🔍 Looking for: {tech_name}")

                source_file = find_technology_file(tech_name, path_files_technologies)
                if source_file:
                    dest_file = tech_data_dir / f"{tech_name}.json"
                    try:
                        shutil.copy2(source_file, dest_file)

                        # Verify the copied file
                        if dest_file.exists():
                            file_size = dest_file.stat().st_size
                            print(f"        ✅ Copied successfully: {source_file} -> {dest_file} ({file_size} bytes)")

                            # Quick validation of JSON content
                            try:
                                with open(dest_file, 'r') as f:
                                    test_json = json.load(f)
                                print(f"        ✅ JSON validation passed")
                            except Exception as e:
                                print(f"        ⚠️  JSON validation failed: {e}")
                        else:
                            print(f"        ❌ File not found after copy")

                    except Exception as e:
                        print(f"        ❌ Copy failed: {e}")
                else:
                    print(f"        ❌ Source file not found in {path_files_technologies}")

    print("\n🔍 Technology file copying debugging completed")


# Add this function to your defined_functions.py file:

def convert_network_data_indices_to_names(network_data_dict, network_location):
    """
    Convert network data indices from node IDs to node names to match template structure.

    Parameters:
        - network_data_dict: Dictionary containing network data with numeric indices
        - network_location: DataFrame with node_id -> node_name mapping

    Returns:
        - network_data_dict: Updated dictionary with node names as indices
    """
    print("\n🔍 DEBUG: Converting network data indices from node IDs to node names...")

    # Create node_id to node_name mapping
    node_id_to_name = network_location['node_name'].to_dict()
    print(f"  Node ID to Name mapping: {node_id_to_name}")

    # Convert each network type
    for network_type, network_data in network_data_dict.items():
        print(f"\n  Processing {network_type} network:")
        print(f"    Original index: {list(network_data.index)}")
        print(f"    Original columns: {list(network_data.columns)}")

        # Map indices and columns from node IDs to node names
        new_index = [node_id_to_name.get(node_id, f"UNKNOWN_ID_{node_id}") for node_id in network_data.index]
        new_columns = [node_id_to_name.get(node_id, f"UNKNOWN_ID_{node_id}") for node_id in network_data.columns]

        # Update the DataFrame
        network_data.index = new_index
        network_data.columns = new_columns

        print(f"    Mapped index: {list(network_data.index)}")
        print(f"    Mapped columns: {list(network_data.columns)}")

        # Verify data integrity
        non_zero_count = (network_data != 0).sum().sum()
        print(f"    Non-zero values: {non_zero_count}")

        # Check for any unmapped nodes
        unknown_nodes = [node for node in new_index if node.startswith("UNKNOWN_ID_")]
        if unknown_nodes:
            print(f"    ⚠️  WARNING: Unmapped node IDs: {unknown_nodes}")

    print("\n✅ Network data index conversion completed")
    return network_data_dict


def apply_carbon_pricing_to_all_nodes(input_data_path, carbon_tax_euro_per_tonne, node_names):
    """
    Apply carbon pricing to all nodes by updating their CarbonCost.csv files.
    """

    print(f"\n💰 APPLYING CARBON PRICING TO ALL NODES")
    print(f"Carbon tax: €{carbon_tax_euro_per_tonne}/tonne CO2")
    print("=" * 50)


    # Create hourly carbon price array (8760 hours)
    carbon_price = np.ones(8760) * carbon_tax_euro_per_tonne

    successful_nodes = []
    failed_nodes = []

    for node_name in node_names:
        try:
            # Define path to CarbonCost.csv for this node
            carbon_cost_path = (
                    input_data_path / "period1" / "node_data" / node_name / "CarbonCost.csv"
            )

            # Check if CarbonCost.csv exists, create if not
            if not carbon_cost_path.exists():
                print(f"📝 {node_name}: Creating CarbonCost.csv template...")

                # Ensure directory exists
                carbon_cost_path.parent.mkdir(parents=True, exist_ok=True)

                # Create template
                template_data = {
                    't': list(range(1, 8761)),  # Hours 1 to 8760
                    'price': np.zeros(8760)  # Will be filled with actual carbon price
                }
                carbon_cost_df = pd.DataFrame(template_data)
                carbon_cost_df.to_csv(carbon_cost_path, sep=";", index=False)

            # Read the CarbonCost template
            carbon_cost_template = pd.read_csv(carbon_cost_path, sep=";", index_col=0, header=0)

            # Update the price column with carbon price
            carbon_cost_template["price"] = carbon_price

            # Reset index and save back to CSV
            carbon_cost_template = carbon_cost_template.reset_index()
            carbon_cost_template.to_csv(carbon_cost_path, sep=";", index=False)

            successful_nodes.append(node_name)
            print(f"✅ {node_name}: Carbon pricing applied")

        except Exception as e:
            failed_nodes.append(node_name)
            print(f"❌ {node_name}: Failed - {e}")

    print(f"\n📊 SUMMARY: ✅ {len(successful_nodes)} success, ❌ {len(failed_nodes)} failed")
    return len(failed_nodes) == 0

