import pandas as pd
import numpy as np
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

# CostsFun_Share is a sibling project (not an installed package) holding the
# Oeuvray et al. (2024) -based container-truck/train cost model used by
# update_capex_gamma2_per_arc() and compute_opex_var_arcs() below.
def _find_costsfun_share_path() -> Path:
    """Walk up from this file until a sibling 'CostsFun_Share' directory is found."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "CostsFun_Share"
        if candidate.is_dir():
            return candidate
    raise ImportError(
        "Could not find the 'CostsFun_Share' directory (expected as a sibling of the "
        "repo root, alongside 'adopt_net0/' and 'southern_europe/'). It provides "
        "truck_costs_per_capacity()/train_costs_per_capacity(), used by "
        "compute_opex_var_arcs() and update_capex_gamma2_per_arc()."
    )


_COSTSFUN_SHARE_PATH = _find_costsfun_share_path()
if str(_COSTSFUN_SHARE_PATH) not in sys.path:
    sys.path.insert(0, str(_COSTSFUN_SHARE_PATH))
from co2_container_transport_costs import (
    truck_costs_per_capacity,
    train_costs_per_capacity,
)

# Default emitter-technology-per-sector selection, matching the original hardcoded
# behavior (one fixed technology per sector, assigned as "existing"). Used as the
# fallback for callers (main.py, main_greece_test.py) that don't pass their own
# technology_selection -- main_italy.py passes an explicit, configurable one.
DEFAULT_TECHNOLOGY_SELECTION = {
    "Waste": ["WasteToEnergyEmitter"],
    "Cement": ["CementEmitter"],
    "Refining": ["RefineryEmitter"],
    "Other": ["UnspecifiedEmitter"],
}

# WasteCaL_CCS and CementHybridCCS bundle the host plant AND its CO2 capture
# unit into one self-contained technology block (unlike the generic bolt-on
# MEA retrofit, which is a separate technology from its host emitter). If
# assigned "existing" like a plain baseline emitter, their capture retrofit
# never gets charged real capex - existing + decommission="impossible" forces
# var_capex == 0 (see wasteToEnergy_CaL_ccs.py/cement_hybrid_ccs.py
# _define_capex_constraints), which is correct for a genuinely sunk host
# plant but wrong for a capture unit that should be a real new investment.
# So these two are always assigned "new" here, regardless of tech_as_existing
# - the host plant's size is still bounded by the technology's own
# size_min/size_max (matching or exceeding the node's real capacity), it's
# only the capex *accounting* that changes.
ALWAYS_NEW_TECHNOLOGIES = {"WasteCaL_CCS", "CementHybridCCS"}

# Default reference emission factor per sector (t CO2 emitted / t product output),
# used as the fallback for callers that don't pass their own sector_emission_factor.
# 1.0 everywhere reproduces the original "emissions == output" shortcut (every
# generic emitter's Performance.emission_factor set to 1) -- main_italy.py passes
# real, sector-specific values instead.
DEFAULT_SECTOR_EMISSION_FACTOR = {
    "Waste": 1.0,
    "Cement": 1.0,
    "Refining": 1.0,
    "Other": 1.0,
}

# Default flue-gas CO2 concentration per sector, used to size the generic bolt-on
# MEA CCS retrofit (assign_mea_technology) and to fill Performance.ccs.co2_concentration
# (update_emitter_ccs_references) when not already set. Fallback for callers that don't
# pass their own co2_concentration_by_type.
DEFAULT_CO2_CONCENTRATION_BY_TYPE = {
    "Waste": 0.07,
    "Cement": 0.20,
    "Refining": 0.15,
    "Other": 0.15,
}


def load_emission_profiles(path_files_node_flux):
    """
    Load hourly emission profiles from emission_profile_emitters.xlsx (real_data + synthetic_data sheets).
    Returns (profiles_real, profiles_synthetic) DataFrames (either can be None if missing).
    """
    excel_file_path = path_files_node_flux / "emission_profile_emitters.xlsx"

    profiles_real = None
    profiles_synthetic = None

    if excel_file_path.exists():
        xl = pd.ExcelFile(excel_file_path)
        if "real_data" in xl.sheet_names:
            profiles_real = pd.read_excel(xl, sheet_name="real_data")
        if "synthetic_data" in xl.sheet_names:
            profiles_synthetic = pd.read_excel(xl, sheet_name="synthetic_data")

    return profiles_real, profiles_synthetic


def get_profile(node_type: str, node_name: str, profiles_real, profiles_synthetic) -> tuple[np.ndarray | None, str]:
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


def calculate_annual_emission_values(network_emission_flux, path_files_node_flux):
    """
    Compute annual_emission (sum of hourly profile) and capacity (max of hourly profile)
    for each row, using the real_data/synthetic_data profiles.
    Skips calculation for 'Transport' and 'Storage' node types.
    """
    profiles_real, profiles_synthetic = load_emission_profiles(path_files_node_flux)

    annual_emissions = []
    capacities = []

    for _, row in network_emission_flux.iterrows():
        node_type = row['node_type']
        node_name = row['node_name']

        # Skip profile calculation for Transport and Storage
        if node_type in ['Transport', 'Storage']:
            annual_emissions.append(None)
            capacities.append(None)
            continue

        profile, source = get_profile(node_type, node_name, profiles_real, profiles_synthetic)

        if profile is None:
            raise ValueError(
                f"No profile found for '{node_type} - {node_name}' in real_data or synthetic_data sheets."
            )

        annual_emissions.append(profile.sum())
        capacities.append(profile.max()*1.01)

    network_emission_flux['annual_emission'] = annual_emissions
    network_emission_flux['capacity'] = capacities

    return network_emission_flux

    return network_emission_flux
def calculate_emitter_capacities(network_emission_flux, sector_emission_factor=None):
    """
    Set 'emitter_capacity' from the 'capacity' column already computed from the
    hourly profile (max value, from calculate_annual_emission_values).

    IMPORTANT -- units: emission_profile_emitters.xlsx (and therefore the 'capacity'
    column) is always in t CO2/h, regardless of sector -- it's an emissions profile,
    not a production profile. 'emitter_capacity' divides that by the sector's
    reference emission factor (t CO2 / t product) to get an actual PRODUCT output
    capacity (t clinker/h, t waste/h, ...), since that's what the generic Emitter
    technologies' size represents (size_based_on="output", main_output_carrier =
    the sector's product carrier). 'capacity' itself is left unscaled (still t CO2/h)
    for anything that needs the raw CO2 rate instead -- e.g. assign_mea_technology,
    which sizes CCS equipment against actual CO2 mass flow, not product output.

    Parameters:
        - network_emission_flux: DataFrame with 'annual_emission' and 'capacity' columns
        - sector_emission_factor: dict mapping node_type -> t CO2 / t product. Defaults
          to DEFAULT_SECTOR_EMISSION_FACTOR (all 1.0, i.e. no rescaling -- the original
          "emissions == output" shortcut).

    Returns:
        - network_emission_flux: Updated DataFrame with 'emitter_capacity' column
    """
    if sector_emission_factor is None:
        sector_emission_factor = DEFAULT_SECTOR_EMISSION_FACTOR

    if 'annual_emission' not in network_emission_flux.columns:
        raise ValueError("'annual_emission' column not found. Please run calculate_annual_emission_values() first.")
    if 'capacity' not in network_emission_flux.columns:
        raise ValueError("'capacity' column not found. Please run calculate_annual_emission_values() first.")

    capacities = []
    for _, row in network_emission_flux.iterrows():
        node_type = row['node_type']
        annual_emission = row['annual_emission']
        profile_capacity = row['capacity']  # t CO2/h, always

        if node_type in ('Storage', 'Transport') or annual_emission == 0:
            capacities.append(0.0)
            continue

        if pd.isna(profile_capacity) or profile_capacity <= 0:
            raise ValueError(
                f"Missing or invalid capacity for node '{row['node_name']}' ({node_type}): {profile_capacity}"
            )

        factor = sector_emission_factor.get(node_type, 1.0)
        capacities.append(round(profile_capacity / factor, 2))

    network_emission_flux['emitter_capacity'] = capacities

    return network_emission_flux


def assign_carriers_to_nodes(input_data_path, network_location, network_emission_flux,
                              technology_selection=None):
    """
    Assign appropriate carriers to each node based on their type(s).

    Carrier assignment rules:
    - All nodes get: electricity, heat, CO2captured (except Transport nodes)
    - Transport nodes get: electricity, CO2captured only (no heat)
    - Cement nodes also get: clinker
    - Waste nodes also get: waste
    - Refining nodes also get: refined_product
    - Other nodes also get: industrial_product
    - Nodes with multiple emitters get all relevant carriers from both emitter types
    - If WasteCaL_CCS is the selected Waste technology (technology_selection["Waste"]),
      also add "wasteIn" -- WasteCaL_CCS.json's own input_carrier, distinct from the
      generic "waste" product carrier.

    Parameters:
        - input_data_path: Path to the input data directory
        - network_location: DataFrame containing node information with node_name and node_type
        - network_emission_flux: DataFrame containing emission data with node_name and node_type
        - technology_selection: dict mapping node_type -> list of technology names
          (see assign_ccs_technologies_debug); used only to detect whether
          WasteCaL_CCS is in play. Defaults to DEFAULT_TECHNOLOGY_SELECTION.

    Returns:
        - None (updates Topology.json file)
    """
    if technology_selection is None:
        technology_selection = DEFAULT_TECHNOLOGY_SELECTION

    # Get all unique nodes
    all_nodes = network_location['node_name'].unique().tolist()

    # Base carriers that most nodes get
    base_carriers = ["electricity", "heat", "CO2captured"]
    transport_carriers = ["electricity", "CO2captured"]  # Transport nodes don't get heat

    # Mapping from emitter node_type to specific carriers
    emitter_carriers = {
        'Cement': 'clinker',
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

    if "WasteCaL_CCS" in technology_selection.get("Waste", []):
        all_carriers.add("wasteIn")

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


def assign_mea_technology(network_emission_flux, path_data_case_study, co2_concentration_by_type=None):
    """
    Determines appropriate MEA (Monoethanolamine) carbon capture technology scale
    for emitter nodes based on their CO2 emission rate (t CO2/h).

    This function analyzes capacity data for each node and determines the appropriate
    MEA technology scale (small, medium, large), adding it to a new column 'mea_technology'.

    IMPORTANT -- units: uses the 'capacity' column (raw t CO2/h from the emission
    profile), NOT 'emitter_capacity'. MEA equipment is sized against actual CO2 mass
    flow, so it must stay in CO2 units even after 'emitter_capacity' gets rescaled to
    product units by calculate_emitter_capacities().

    Parameters:
        - network_emission_flux: DataFrame containing node information and 'capacity' column (t CO2/h)
        - path_data_case_study: Path to the case study data directory
        - co2_concentration_by_type: dict mapping node_type -> flue-gas CO2 concentration.
          Defaults to DEFAULT_CO2_CONCENTRATION_BY_TYPE.

    Returns:
        - network_emission_flux: Updated DataFrame with mea_technology column added
    """
    if co2_concentration_by_type is None:
        co2_concentration_by_type = DEFAULT_CO2_CONCENTRATION_BY_TYPE

    # Ensure capacity column exists
    if 'capacity' not in network_emission_flux.columns:
        raise ValueError("'capacity' column not found. Please run calculate_annual_emission_values() first.")

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
        node_type = row['node_type']

        # Skip non-emitter nodes (Storage and Transport)
        if node_type in ["Storage", "Transport"]:
            continue

        # Get the node's CO2 emission rate (t CO2/h) -- NOT emitter_capacity, see docstring
        capacity = row["capacity"]

        # Determine CO2 concentration based on emitter type
        co2_concentration = co2_concentration_by_type.get(node_type, 0.15)

        # Calculate CO2 capacity ranges (t/h) for each MEA scale based on technology specs
        mea_ranges = {}
        for scale, data in mea_data.items():
            min_co2 = co2_concentration * data["size_min"]
            max_co2 = co2_concentration * data["size_max"]
            mea_ranges[scale] = (min_co2, max_co2)

        # Find the MEA scale that matches the node's capacity range
        suitable_mea = None
        for scale, (min_co2, max_co2) in mea_ranges.items():
            if min_co2 <= capacity <= max_co2:
                suitable_mea = scale
                break

        # If no exact match found, choose the closest scale
        if suitable_mea is None:
            distances = {}
            for scale, (min_co2, max_co2) in mea_ranges.items():
                if capacity < min_co2:
                    distances[scale] = min_co2 - capacity
                elif capacity > max_co2:
                    distances[scale] = capacity - max_co2

            suitable_mea = min(distances, key=distances.get)

        # Store the suitable MEA technology in the mea_technology column
        mea_tech_path = str(path_data_case_study / f"technologies/CCSTechnologies/MEA_{suitable_mea}.json")
        network_emission_flux.at[idx, 'mea_technology'] = mea_tech_path

    return network_emission_flux


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


def update_emitter_ccs_references(input_data_path, network_emission_flux, technology_selection=None,
                                   co2_concentration_by_type=None):
    """
    Updates the CCS references in emitter technology files to match the determined MEA technology size.
    This function ensures that each emitter's CCS section points to the correct MEA technology
    as determined by the assign_mea_technology function.

    Only technologies whose copied JSON already defines a Performance.ccs block are touched --
    e.g. CementHybridCCS has no "ccs" key (its capture is built into the technology model itself,
    not the generic bolt-on MEA retrofit), so it is left untouched rather than having a fake ccs
    block force-created on it.

    Parameters:
        - input_data_path: Path to the input data directory
        - network_emission_flux: DataFrame containing emission data and MEA technology assignments
        - technology_selection: dict mapping node_type ("Waste", "Cement", "Refining", "Other") to
          the list of technology names assigned to that sector (see assign_ccs_technologies_debug)
        - co2_concentration_by_type: dict mapping node_type -> flue-gas CO2 concentration, used to
          fill Performance.ccs.co2_concentration when not already set. Defaults to
          DEFAULT_CO2_CONCENTRATION_BY_TYPE (should match what was passed to assign_mea_technology).
    """
    if technology_selection is None:
        technology_selection = DEFAULT_TECHNOLOGY_SELECTION
    if co2_concentration_by_type is None:
        co2_concentration_by_type = DEFAULT_CO2_CONCENTRATION_BY_TYPE

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
                if node_type not in technology_selection:
                    continue

                mea_tech = emission_row.get('mea_technology')
                if not pd.notna(mea_tech):
                    continue

                mea_tech_name = Path(mea_tech).stem  # e.g., 'MEA_medium' or 'MEA_large'

                for emitter_tech_name in technology_selection[node_type]:
                    tech_file_path = input_data_path / period / "node_data" / node / "technology_data" / f"{emitter_tech_name}.json"

                    if not tech_file_path.exists():
                        print(f"  ❌ Technology file not found: {tech_file_path}")
                        continue

                    # Read the technology file
                    with open(tech_file_path, 'r') as f:
                        tech_data = json.load(f)

                    # Technologies without a pre-existing "ccs" block (e.g. CementHybridCCS)
                    # manage their own capture internally -- don't force one onto them.
                    if "Performance" not in tech_data or "ccs" not in tech_data["Performance"]:
                        print(f"  ⏭️  Skipped {emitter_tech_name} at {node} (no generic CCS retrofit block)")
                        continue

                    # Update the CCS section with the determined MEA technology
                    tech_data["Performance"]["ccs"]["possible"] = 1
                    tech_data["Performance"]["ccs"]["ccs_type"] = mea_tech_name

                    # Set CO2 concentration based on emitter type (if not already set)
                    if "co2_concentration" not in tech_data["Performance"]["ccs"]:
                        tech_data["Performance"]["ccs"]["co2_concentration"] = co2_concentration_by_type[node_type]

                    # Write back the updated file
                    with open(tech_file_path, 'w') as f:
                        json.dump(tech_data, f, indent=2)

                    print(f"  ✅ Updated {emitter_tech_name} at {node} to use CCS: {mea_tech_name}")

    print("CCS reference updates completed.")


def load_sector_reference_values(path_files_technologies, reference_technologies, json_path):
    """
    Reads a reference value per sector directly out of a fixed group of "reference"
    technologies' SOURCE JSON files (e.g. italy_data/technologies/Emitter/CementEmitter.json)
    -- these JSON files are the single source of truth; this just surfaces one field
    from them into a plain per-sector dict for use elsewhere in the pipeline (e.g.
    calculate_emitter_capacities(), update_carrier_data(), assign_mea_technology()),
    instead of hand-duplicating the same number as a separate literal in main_italy.py.

    IMPORTANT: reference_technologies is deliberately a FIXED group (see
    REFERENCE_EMITTER_TECHNOLOGIES), independent of technology_selection (main_italy.py's
    live choice of which technology to actually build/use per sector). The values read
    here (emission_factor, co2_concentration) describe physical/sector properties --
    t CO2 per t product, flue-gas concentration -- that don't change depending on which
    technology is currently selected for the optimization, so they must always be read
    from the same fixed set regardless of tech_for_cement/tech_for_waste/etc.

    For each sector, uses the first technology in reference_technologies[sector] whose
    JSON actually defines the requested field -- not every technology needs to (e.g.
    CementHybridCCS has no "ccs" block at all, since its capture is built into the
    technology model rather than the generic bolt-on MEA retrofit, so it's skipped
    when looking up ("Performance", "ccs", "co2_concentration") but still checked for
    ("Performance", "emission_factor")). If the technologies for a sector disagree on
    the value, warns and uses the first one found. If NONE of a sector's reference
    technologies define the field, warns and omits that sector from the returned dict
    -- callers should already fall back sanely via dict.get(...) (as
    calculate_emitter_capacities/update_carrier_data/assign_mea_technology all do).

    Parameters:
        - path_files_technologies: Path to the source technology files directory
          (e.g. italy_data/technologies)
        - reference_technologies: dict mapping node_type -> list of technology names
          to read the value from (typically REFERENCE_EMITTER_TECHNOLOGIES, NOT the
          live technology_selection)
        - json_path: sequence of keys to walk into each technology's JSON, e.g.
          ("Performance", "emission_factor") or ("Performance", "ccs", "co2_concentration")

    Returns:
        - dict mapping node_type -> value (only for sectors where a value was found)
    """
    field_label = ".".join(json_path)
    values = {}

    for node_type, tech_names in reference_technologies.items():
        found = []
        for tech_name in tech_names:
            source_file = find_technology_file(tech_name, path_files_technologies)
            if source_file is None:
                continue
            with open(source_file, "r") as f:
                tech_data = json.load(f)

            value = tech_data
            for key in json_path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]

            if value is not None:
                found.append((tech_name, value))

        if not found:
            print(f"  ⚠️  Sector '{node_type}': no selected technology ({tech_names}) defines "
                  f"{field_label} -- omitted, caller's own default will apply")
            continue

        distinct_values = {v for _, v in found}
        if len(distinct_values) > 1:
            print(f"  ⚠️  Sector '{node_type}': selected technologies disagree on {field_label}: "
                  f"{found} -- using {found[0][1]} (from {found[0][0]})")

        values[node_type] = found[0][1]
        print(f"  ✅ Sector '{node_type}': {field_label} = {found[0][1]} (from {found[0][0]})")

    return values


def update_cement_hybrid_ccs_capacities(input_data_path, network_emission_flux, tech_name="CementHybridCCS"):
    """
    Writes each Cement node's fixed clinker production capacity into its copied
    CementHybridCCS.json (Performance.prod_capacity_clinker) -- used when
    Performance.size_is_fixed == 1 to pin the technology's size to that node's own
    real installed capacity, instead of leaving prod_capacity_clinker at whatever
    single generic value the source template happened to have.

    Units: network_emission_flux['emitter_capacity'] is already in product units
    (t clinker/h) by this point -- calculate_emitter_capacities() converts it from the
    raw CO2-emission-rate profile using the sector's reference emission factor (see
    that function's docstring). So prod_capacity_clinker is simply set equal to
    emitter_capacity here; no further conversion happens in this function. (Note:
    this relies on sector_emission_factor["Cement"], passed to
    calculate_emitter_capacities(), being kept consistent with this JSON's own
    Performance.performance.tCO2_tclinker -- they represent the same physical
    quantity but are two separate fields.)

    Only touches nodes that actually have a copied {tech_name}.json under
    technology_data/ (i.e. nodes where it was included in technology_selection for
    "Cement" -- see assign_ccs_technologies_debug) -- other Cement nodes are skipped
    with a note, not treated as an error.

    Validates the computed capacity:
      - against the node's own size_min/size_max: since size_is_fixed pins var_size
        to prod_capacity_clinker via an equality constraint, a value outside those
        bounds makes the model infeasible for that node -- raises ValueError rather
        than writing a value that would silently break the solve.
      - against the oxyfuel piecewise capex curve's highest breakpoint
        (Economics.piecewise_capex.bp_x[-1]): capex is computed via linear
        interpolation (np.interp), which silently flat-extrapolates beyond the
        curve's defined domain instead of raising -- exceeding it means capex would
        be UNDERESTIMATED for that node. Only warns for this, since it's a
        data-completeness issue in cement_sheet.xlsx (extend the breakpoints), not
        something to block the run on.

    Parameters:
        - input_data_path: Path to the case-study input data directory
        - network_emission_flux: DataFrame with node_name, node_type, emitter_capacity
        - tech_name: technology filename (without ".json") to patch; defaults to
          "CementHybridCCS"
    """
    with open(input_data_path / "Topology.json", "r") as f:
        topology = json.load(f)

    print(f"Updating {tech_name} fixed clinker capacities per node...")

    cement_rows = network_emission_flux[network_emission_flux["node_type"] == "Cement"]

    for period in topology["investment_periods"]:
        for _, row in cement_rows.iterrows():
            node_name = row["node_name"]
            emitter_capacity = float(row["emitter_capacity"])

            tech_file_path = (
                input_data_path / period / "node_data" / node_name / "technology_data" / f"{tech_name}.json"
            )
            if not tech_file_path.exists():
                print(f"  ⏭️  Skipped {node_name} ({tech_name}.json not selected/copied at this node)")
                continue

            with open(tech_file_path, "r") as f:
                tech_data = json.load(f)

            prod_capacity_clinker = round(emitter_capacity, 3)

            size_min = tech_data.get("size_min", 0)
            size_max = tech_data.get("size_max")
            if size_max is not None and not (size_min <= prod_capacity_clinker <= size_max):
                raise ValueError(
                    f"{node_name}: emitter_capacity={prod_capacity_clinker} t clinker/h "
                    f"is outside {tech_name}.json's [size_min={size_min}, size_max={size_max}]. Since "
                    f"Performance.size_is_fixed pins var_size == prod_capacity_clinker, this would make "
                    f"the model infeasible for this node. Widen size_max (or check the capacity data)."
                )

            bp_x_oxy = tech_data["Economics"]["piecewise_capex"]["bp_x"]
            if prod_capacity_clinker > max(bp_x_oxy):
                print(
                    f"  ⚠️  {node_name}: prod_capacity_clinker={prod_capacity_clinker} t/h exceeds the "
                    f"oxyfuel piecewise capex curve's highest breakpoint ({max(bp_x_oxy)} t/h) -- capex "
                    f"for this node will be flat-extrapolated (underestimated) via np.interp. Consider "
                    f"extending cement_sheet.xlsx's capex_cpu_oxyfuel breakpoints."
                )

            tech_data["Performance"]["prod_capacity_clinker"] = prod_capacity_clinker

            with open(tech_file_path, "w") as f:
                json.dump(tech_data, f, indent=2)

            print(f"  ✅ {node_name}: prod_capacity_clinker = {prod_capacity_clinker} t/h")

    print(f"{tech_name} capacity updates completed.")


def update_wastecal_ccs_capacities(input_data_path, network_emission_flux, tech_name="WasteCaL_CCS"):
    """
    Writes each Waste node's fixed waste-processing capacity into its copied
    WasteCaL_CCS.json (Performance.prod_capacity_wte) -- used when
    Performance.size_is_fixed == 1 to pin var_size (max wasteIn throughput) to
    that node's own real capacity. Mirrors update_cement_hybrid_ccs_capacities's
    prod_capacity_clinker mechanism for CementHybridCCS; see
    wasteToEnergy_CaL_ccs.py's construct_tech_model (const_size_wte) for the
    corresponding Pyomo constraint.

    Needed because WasteCaL_CCS is always assigned "new" regardless of
    tech_as_existing (see ALWAYS_NEW_TECHNOLOGIES in assign_ccs_technologies_debug)
    so its capex reflects a real capture-unit investment instead of being zeroed
    out by the existing-technology capex constraint -- but "new" alone leaves
    var_size as a free variable bounded only by the technology's generic
    size_min/size_max, unrelated to any specific node's real throughput.

    Units: network_emission_flux['emitter_capacity'] is already in product units
    (t waste/h) by this point (see calculate_emitter_capacities).

    Only touches nodes that actually have a copied {tech_name}.json under
    technology_data/ (i.e. nodes where it was included in technology_selection
    for "Waste" -- see assign_ccs_technologies_debug) -- other Waste nodes are
    skipped with a note, not treated as an error.

    Validates the computed capacity against the node's own size_min/size_max:
    since size_is_fixed pins var_size to prod_capacity_wte via an equality
    constraint, a value outside those bounds makes the model infeasible for
    that node -- raises ValueError rather than writing a value that would
    silently break the solve. (Unlike update_cement_hybrid_ccs_capacities,
    this does NOT also check against the CaL capex curve's breakpoints -- those
    are computed at runtime from wasteCaL_sheet.xlsx by
    _define_capex_parameters, not stored statically in this JSON, so aren't
    available to check here.)

    Parameters:
        - input_data_path: Path to the case-study input data directory
        - network_emission_flux: DataFrame with node_name, node_type, emitter_capacity
        - tech_name: technology filename (without ".json") to patch; defaults to
          "WasteCaL_CCS"
    """
    with open(input_data_path / "Topology.json", "r") as f:
        topology = json.load(f)

    print(f"Updating {tech_name} fixed waste-processing capacities per node...")

    waste_rows = network_emission_flux[network_emission_flux["node_type"] == "Waste"]

    for period in topology["investment_periods"]:
        for _, row in waste_rows.iterrows():
            node_name = row["node_name"]
            emitter_capacity = float(row["emitter_capacity"])

            tech_file_path = (
                input_data_path / period / "node_data" / node_name / "technology_data" / f"{tech_name}.json"
            )
            if not tech_file_path.exists():
                print(f"  ⏭️  Skipped {node_name} ({tech_name}.json not selected/copied at this node)")
                continue

            with open(tech_file_path, "r") as f:
                tech_data = json.load(f)

            prod_capacity_wte = round(emitter_capacity, 3)

            size_min = tech_data.get("size_min", 0)
            size_max = tech_data.get("size_max")
            if size_max is not None and not (size_min <= prod_capacity_wte <= size_max):
                raise ValueError(
                    f"{node_name}: emitter_capacity={prod_capacity_wte} t waste/h "
                    f"is outside {tech_name}.json's [size_min={size_min}, size_max={size_max}]. Since "
                    f"Performance.size_is_fixed pins var_size == prod_capacity_wte, this would make "
                    f"the model infeasible for this node. Widen size_max (or check the capacity data)."
                )

            tech_data["Performance"]["prod_capacity_wte"] = prod_capacity_wte

            with open(tech_file_path, "w") as f:
                json.dump(tech_data, f, indent=2)

            print(f"  ✅ {node_name}: prod_capacity_wte = {prod_capacity_wte} t/h")

    print(f"{tech_name} capacity updates completed.")


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


# Pipeline size classes curated via
# visualisation/pipeline_class_connections_dashboard.py (see
# load_pipeline_class_connection_matrix below).
PIPELINE_SIZE_CLASSES = ["small", "medium", "large"]


def load_pipeline_class_connection_matrix(overrides_path, size_class, base_pipeline_df):
    """
    Apply a per-arc on/off mask for one pipeline size class to
    base_pipeline_df, zeroing out any arc the user disabled for that class -
    e.g. so a 'large' pipeline is never built on an arc that only ever
    carries a small emitter's flow, even though the arc physically exists.

    The mask is curated via visualisation/pipeline_class_connections_dashboard.py
    and saved to pipeline_size_class_connections.xlsx (one sheet per size
    class, a node_id x node_id matrix of 1=enabled/0=disabled). An arc
    missing from the mask, or the override file/sheet not existing at all,
    defaults to enabled - i.e. identical to every size class sharing the
    same flat pipeline connectivity, which was the behaviour before
    per-class connection curation existed.

    Parameters:
    - overrides_path: Path to pipeline_size_class_connections.xlsx (may not exist)
    - size_class: one of PIPELINE_SIZE_CLASSES ("small" | "medium" | "large")
    - base_pipeline_df: node_id-indexed pipeline matrix (0 = no connection,
      >0 = connected, value = distance)

    Returns:
    - pd.DataFrame: copy of base_pipeline_df with disabled arcs zeroed out
    """
    result = base_pipeline_df.copy()
    overrides_path = Path(overrides_path)
    if not overrides_path.exists():
        return result

    try:
        xl = pd.ExcelFile(overrides_path)
        if size_class not in xl.sheet_names:
            return result
        mask_df = pd.read_excel(overrides_path, sheet_name=size_class, index_col=0)
        mask_df.index = mask_df.index.astype(int)
        mask_df.columns = mask_df.columns.astype(int)
    except Exception as e:
        print(f"⚠️  Could not read pipeline class connection overrides for '{size_class}': {e}")
        return result

    mask_aligned = mask_df.reindex(index=result.index, columns=result.columns, fill_value=1).fillna(1)
    disabled = mask_aligned.values == 0
    n_disabled = int(disabled.sum())
    if n_disabled:
        values = result.values.astype(float)
        values[disabled] = 0.0
        result.iloc[:, :] = values
        print(f"🖊️  Pipeline class '{size_class}': {n_disabled} arc(s) disabled via connection overrides")

    return result


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

    # Mapping from network data keys to network type folders. The three
    # pipeline size classes each get their own data key ('pipeline_small' /
    # '_medium' / '_large') so a connection can be disabled for one class
    # without affecting the others (see
    # load_pipeline_class_connection_matrix) - they default to the same
    # connectivity as flat 'pipeline' unless a per-class override exists.
    network_type_mapping = [
        ('pipeline', 'CO2_Pipeline'),
        ('pipeline_small', 'CO2_Pipeline_small'),
        ('pipeline_medium', 'CO2_Pipeline_medium'),
        ('pipeline_large', 'CO2_Pipeline_large'),
        ('truck', 'CO2Truck'),
        ('railway', 'CO2Railway'),
        ('ship', 'CO2Ship'),
    ]

    # Process each network data type
    for data_key, network_type in network_type_mapping:
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # FIXED: Ensure proper CSV formatting
        try:
            updated_connection.to_csv(output_path, sep=";", lineterminator='\n', encoding='utf-8')
            print(f"✅ Successfully saved connection matrix for {network_type} to {output_path}")
        except Exception as e:
            print(f"❌ Error saving connection matrix for {network_type}: {e}")
            continue

    return True


def update_network_size_max_arcs(input_data_path, network_data_dict, max_transport_capacity,
                                  per_network_type_capacity=None):
    """
    Update size_max_arcs matrices using a predefined transport capacity value.
    Uses the max_transport_capacity parameter from scenario parameterization.

    Parameters:
    - input_data_path: Path object pointing to the input data directory
    - network_data_dict: Dictionary with keys 'pipeline', 'truck', 'railway', 'ship' containing network data
    - max_transport_capacity: Default maximum transport capacity in tonnes/hour, used for
      any network type not present in per_network_type_capacity
    - per_network_type_capacity: Optional dict {network_type: max_capacity_t_h} overriding
      max_transport_capacity for specific network types. Needed for the pipeline size
      classes: since size_max_defined_per_arc=1 in their JSONs, the JSON's own scalar
      size_max is never read by the model (see network.py::fit_network_performance) -
      size_max_arcs.csv (written here) is the ONLY thing that actually bounds a built
      arc's capacity. Leaving all three classes at the same flat max_transport_capacity
      would let e.g. a 'small' pipe (calibrated for ~11-104 t/h) get built at, say,
      2000 t/h using its extrapolated - and wrong outside that range - gamma1/gamma2,
      which defeats the purpose of having distinct calibrated size classes.
    """
    per_network_type_capacity = per_network_type_capacity or {}

    # Load the template size_max_arcs CSV
    template_size_max = pd.read_csv(input_data_path / "period1" / "network_topology" / "new" / "size_max_arcs.csv",
                                    sep=";", index_col=0)

    print(f"🔍 PREDEFINED Network sizing:")
    print(f"  Default transport capacity: {max_transport_capacity:.2f} tonnes/hour")
    if per_network_type_capacity:
        print(f"  Per-network-type overrides: {per_network_type_capacity}")

    # Mapping from network data keys to network type folders. The three
    # pipeline size classes each get their own data key so a connection can
    # be disabled for one class without affecting the others - see
    # update_network_connection_matrix's mapping above for details.
    network_type_mapping = [
        ('pipeline', 'CO2_Pipeline'),
        ('pipeline_small', 'CO2_Pipeline_small'),
        ('pipeline_medium', 'CO2_Pipeline_medium'),
        ('pipeline_large', 'CO2_Pipeline_large'),
        ('truck', 'CO2Truck'),
        ('railway', 'CO2Railway'),
        ('ship', 'CO2Ship'),
    ]

    # Process each network data type with its own (or the default) predefined capacity
    for data_key, network_type in network_type_mapping:
        if data_key not in network_data_dict:
            print(f"Warning: {data_key} data not found in network_data_dict")
            continue

        network_capacity = per_network_type_capacity.get(network_type, max_transport_capacity)
        network_data = network_data_dict[data_key]

        print(f"🔍 Processing {network_type} with capacity {network_capacity:.2f} tonnes/hour")

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
            size_max_values = connection_matrix * network_capacity

            # Update the template matrix
            updated_size_max = updated_size_max.astype(float)
            updated_size_max.iloc[:, :] = size_max_values

            # Count connections for this network type
            num_connections = np.count_nonzero(connection_matrix)
            print(f"   {num_connections} connections found with capacity {network_capacity:.2f} tonnes/hour each")

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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            updated_size_max.to_csv(output_path, sep=";", float_format='%.2f',
                                    lineterminator='\n', encoding='utf-8')
            print(f"✅ Successfully saved size_max matrix for {network_type} to {output_path}")
        except Exception as e:
            print(f"❌ Error saving size_max matrix for {network_type}: {e}")
            continue

    print(f"\n✅ Network sizing complete (default {max_transport_capacity:.2f} t/h"
          f"{', per-type overrides: ' + str(per_network_type_capacity) if per_network_type_capacity else ''})")

    return True




def compute_opex_var_arcs(
    path_node_metrics: Path, path_output_root: Path, discount_rate: float = 0.08,
    target_year: int = None,
) -> None:
    """
    Reads truck and railway distance matrices from node_metrics.xlsx and writes
    opex_var_arcs.csv to the respective CO2Truck and CO2Railway output folders.

    Truck: uses the mechanistic, usage-based variable-opex rate (fuel,
    maintenance, driver time, per-km HGVT) from
    CostsFun_Share/co2_container_transport_costs.py::truck_costs_per_capacity(),
    itself calibrated against the Oeuvray et al. (2024) published fit
    (Table C.1). This replaced the previous flat-fit formula
    (5.58/d + 0.15)*d, which conflated capacity cost (capex) and usage cost
    (opex_var) into a single number; the two are now split, with the
    capacity part going into gamma2 (see
    update_capex_gammas_truck_railway()).

    Railway: written as all zeros. For container-based train, the
    equivalent usage-based costs (transshipment, weighing, rail linehaul)
    are folded into gamma2/gamma4 instead of kept as a separate per-tonne
    rate (rail capacity is normally bought as a single bundled per-isotainer
    service rather than operated/fuelled directly) - see
    update_capex_gammas_truck_railway().

    Zero-distance entries (no arc / same node) are left as 0.

    Parameters
    ----------
    path_node_metrics : Path
        Full path to node_metrics.xlsx.
    path_output_root : Path
        Root folder that contains CO2Truck/ and CO2Railway/ sub-folders.
    discount_rate : float
        Discount rate passed to truck_costs_per_capacity() (does not affect
        variable opex, kept for consistency with update_capex_gammas_truck_railway()).
    target_year : int, optional
        If given, escalates the underlying EUR_2021 (Oeuvray et al. 2024)
        costs to EUR_{target_year} via truck_costs_per_capacity()'s
        target_year argument (Eurostat producer-price-index based).
    """

    def _opex_truck(d: float) -> float:
        if d <= 0:
            return 0.0
        return truck_costs_per_capacity(d, discount_rate, target_year=target_year)["variable_opex_eur_per_t"]

    def _opex_railway(d: float) -> float:
        return 0.0

    # Build a mapping from node_id -> node_name
    nodes_df = pd.read_excel(path_node_metrics, sheet_name="nodes")
    id_to_name = nodes_df.set_index('node_id')['node_name'].to_dict()

    truck_dist = pd.read_excel(path_node_metrics, sheet_name="truck", index_col=0)
    rail_dist = pd.read_excel(path_node_metrics, sheet_name="railway", index_col=0)

    # Rename both index and columns using the id -> name mapping
    truck_dist = truck_dist.rename(index=id_to_name, columns=id_to_name)
    rail_dist = rail_dist.rename(index=id_to_name, columns=id_to_name)

    truck_opex = truck_dist.map(_opex_truck)
    rail_opex = rail_dist.map(_opex_railway)

    path_truck_out = path_output_root / "CO2Truck" / "opex_var_arcs.csv"
    path_railway_out = path_output_root / "CO2Railway" / "opex_var_arcs.csv"

    path_truck_out.parent.mkdir(parents=True, exist_ok=True)
    path_railway_out.parent.mkdir(parents=True, exist_ok=True)

    truck_opex.to_csv(path_truck_out, sep=';')
    rail_opex.to_csv(path_railway_out, sep=';')

    print(f"Saved truck   opex_var_arcs -> {path_truck_out}")
    print(f"Saved railway opex_var_arcs -> {path_railway_out}")


def update_capex_gamma2_per_arc(
    path_node_metrics: Path,
    path_output_root: Path,
    path_network_data: Path,
    discount_rate: float = 0.08,
    target_year: int = None,
) -> None:
    """
    Writes per-arc gamma1.csv/gamma2.csv/gamma3.csv/gamma4.csv for CO2Truck
    and CO2Railway, using the capacity-based capex + fixed-opex from
    CostsFun_Share/co2_container_transport_costs.py, and sets
    capex_defined_per_arc=1 in CO2Truck.json/CO2Railway.json so the solver
    reads these files instead of the (distance-independent) global gamma
    scalars in the JSON.

    Background: the network cost model (adopt_net0/components/networks/network.py)
    computes each arc's capex as
        gamma1 + gamma2*size + gamma3*distance + gamma4*size*distance
    where size is in t/h. Since gamma2, gamma3, gamma4 must vary per arc
    with that arc's own distance (a longer trip needs proportionally more
    trucks/isotainers per unit of guaranteed t/h capacity - it's not just a
    fixed per-unit-capacity cost), the whole (capex + fixed_opex) at that
    arc's distance is put directly into gamma2 (evaluated AT that distance),
    with gamma1=gamma3=gamma4=0 (this cost model has no size-independent or
    pure-per-km term, so there's nothing to put there).

    Units: for this case study, CO2Truck.json/CO2Railway.json use
    discount_rate=0, lifetime=1 with fraction_of_year_modelled=1 (main_italy.py
    runs start_period=0, end_period=8759, i.e. a full year), so
    annualize(0, 1, 1) == 1: gamma2 is used with NO extra annualization by
    the solver, matching the fact that truck_costs_per_capacity()/
    train_costs_per_capacity() already return annualized EUR/(t/h)/y values
    directly.

    Parameters
    ----------
    path_node_metrics : Path
        Full path to node_metrics.xlsx (same source as compute_opex_var_arcs).
    path_output_root : Path
        Root folder that contains CO2Truck/ and CO2Railway/ sub-folders,
        typically input_data_path / "period1" / "network_topology" / "new".
    path_network_data : Path
        Folder containing the (already copied, e.g. via adopt.copy_network_data)
        CO2Truck.json and CO2Railway.json for this run, typically
        input_data_path / "period1" / "network_data".
    discount_rate : float
        Discount rate passed to truck_costs_per_capacity()/train_costs_per_capacity().
    target_year : int, optional
        If given, escalates the underlying EUR_2021 (Oeuvray et al. 2024)
        costs to EUR_{target_year} via truck_costs_per_capacity()/
        train_costs_per_capacity()'s target_year argument (Eurostat
        producer-price-index based).
    """

    def _capex_plus_fixed_opex(d: float, cost_fn) -> float:
        if d <= 0:
            return 0.0
        r = cost_fn(d, discount_rate, target_year=target_year)
        return r["capex_eur_per_tph_y"] + r["fixed_opex_eur_per_tph_y"]

    # Build a mapping from node_id -> node_name (same as compute_opex_var_arcs)
    nodes_df = pd.read_excel(path_node_metrics, sheet_name="nodes")
    id_to_name = nodes_df.set_index('node_id')['node_name'].to_dict()

    truck_dist = pd.read_excel(path_node_metrics, sheet_name="truck", index_col=0)
    rail_dist = pd.read_excel(path_node_metrics, sheet_name="railway", index_col=0)
    truck_dist = truck_dist.rename(index=id_to_name, columns=id_to_name)
    rail_dist = rail_dist.rename(index=id_to_name, columns=id_to_name)

    truck_gamma2 = truck_dist.map(lambda d: _capex_plus_fixed_opex(d, truck_costs_per_capacity))
    rail_gamma2 = rail_dist.map(lambda d: _capex_plus_fixed_opex(d, train_costs_per_capacity))

    for subfolder, dist_df, gamma2_df in [
        ("CO2Truck", truck_dist, truck_gamma2),
        ("CO2Railway", rail_dist, rail_gamma2),
    ]:
        out_dir = path_output_root / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

        zeros_df = pd.DataFrame(0.0, index=dist_df.index, columns=dist_df.columns)
        zeros_df.to_csv(out_dir / "gamma1.csv", sep=';')
        gamma2_df.to_csv(out_dir / "gamma2.csv", sep=';')
        zeros_df.to_csv(out_dir / "gamma3.csv", sep=';')
        zeros_df.to_csv(out_dir / "gamma4.csv", sep=';')
        print(f"Saved per-arc gamma1-4 for {subfolder} -> {out_dir}")

        json_path = path_network_data / f"{subfolder}.json"
        with open(json_path, "r") as f:
            data = json.load(f)
        data["capex_defined_per_arc"] = 1
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Set capex_defined_per_arc=1 -> {json_path}")

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
    valid_modes = ["pipeline", "pipeline_small", "pipeline_medium", "pipeline_large"]
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
        "pipeline_small": "CO2_Pipeline_small",
        "pipeline_medium": "CO2_Pipeline_medium",
        "pipeline_large": "CO2_Pipeline_large",
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
                        cop_hp, levelized_capex_hp, path_files_node_flux,
                        electricity_import_limit=100, heat_import_limit=200,
                        sector_emission_factor=None, wasteIn_import_limit=1000,
                        flatten_profiles=False):
    """
    :param bool flatten_profiles: if True, every hourly demand and electricity/heat
        price profile is replaced by its own annual average repeated across all 8760
        hours (so annual totals are unchanged, but hour-to-hour variance is removed).
        Emitter capacities (sized from the raw, un-flattened profile in
        calculate_emitter_capacities) are unaffected. Used for the "timeless"
        scenario variant that isolates the effect of hourly variability.
    """

    import adopt_net0 as adopt

    if sector_emission_factor is None:
        sector_emission_factor = DEFAULT_SECTOR_EMISSION_FACTOR

    co2_intensity_heat = round(co2_intensity_electricity / cop_hp, 4)

    # --- Import limits ---
    adopt.fill_carrier_data(input_data_path, value_or_data=electricity_import_limit,
                            columns=['Import limit'], carriers=['electricity'], nodes=node_names)
    adopt.fill_carrier_data(input_data_path, value_or_data=heat_import_limit,
                            columns=['Import limit'], carriers=['heat'], nodes=node_names)

    # WasteCaL_CCS's "wasteIn" input carrier (raw feedstock waste) is only present in
    # Topology.json's carrier list -- and therefore only has a carrier_data/wasteIn.csv
    # per node -- when WasteCaL_CCS was actually selected (see assign_carriers_to_nodes).
    with open(input_data_path / "Topology.json", "r") as f:
        _topology_carriers = json.load(f)["carriers"]
    if "wasteIn" in _topology_carriers:
        adopt.fill_carrier_data(input_data_path, value_or_data=wasteIn_import_limit,
                                columns=['Import limit'], carriers=['wasteIn'], nodes=node_names)

    # --- Emission factors ---
    adopt.fill_carrier_data(input_data_path, value_or_data=co2_intensity_electricity,
                            columns=['Import emission factor'], carriers=['electricity'], nodes=node_names)
    adopt.fill_carrier_data(input_data_path, value_or_data=co2_intensity_heat,
                            columns=['Import emission factor'], carriers=['heat'], nodes=node_names)

    # --- Electricity & heat prices ---
    electricity_prices = electricity_price_data['Day-ahead Price (EUR/MWh)'].values
    if flatten_profiles:
        electricity_prices = np.full_like(electricity_prices, electricity_prices.mean(), dtype=float)
    heat_prices = np.round(levelized_capex_hp + electricity_prices / cop_hp, 2)

    adopt.fill_carrier_data(input_data_path, value_or_data=electricity_prices,
                            columns=['Import price'], carriers=['electricity'], nodes=node_names)
    adopt.fill_carrier_data(input_data_path, value_or_data=heat_prices,
                            columns=['Import price'], carriers=['heat'], nodes=node_names)

    # --- Sector -> product carrier mapping ---
    node_type_mapping = {
        'Waste':    ('Emitter/WasteToEnergyEmitter.json', 'waste'),
        'Cement':   ('Emitter/CementEmitter.json',        'clinker'),
        'Refining': ('Emitter/RefineryEmitter.json',      'refined_product'),
        'Other':    ('Emitter/UnspecifiedEmitter.json',   'industrial_product'),
    }

    # --- Load hourly profiles (shared helper) ---
    profiles_real, profiles_synthetic = load_emission_profiles(path_files_node_flux)

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

        hourly_demand_array, source = get_profile(node_type, node_name, profiles_real, profiles_synthetic)

        # emission_profile_emitters.xlsx is always in t CO2/h; rescale to t product/h
        # (matching the carrier's real units) using the sector's reference emission
        # factor -- same conversion as calculate_emitter_capacities().
        hourly_demand_array = hourly_demand_array / sector_emission_factor.get(node_type, 1.0)

        if flatten_profiles:
            # Same length (8760) in and out, so the annual sum (and therefore
            # annual emissions) is exactly preserved -- only the hour-to-hour
            # shape is removed.
            hourly_demand_array = np.full_like(hourly_demand_array, hourly_demand_array.mean())

        if source == "real_data":
            annual_demand_tonnes = round(hourly_demand_array.sum(), 2)
            nodes_with_real_data.append(f"{node_name} ({node_type})")
            print(f"  ✅ Real profile      | {node_name} ({carrier_name}): {annual_demand_tonnes:.2f} t/yr")

        elif source == "synthetic_data":
            annual_demand_tonnes = round(hourly_demand_array.sum(), 2)
            nodes_with_synthetic_data.append(f"{node_name} ({node_type})")
            print(f"  🔧 Synthetic profile | {node_name} ({carrier_name}): {annual_demand_tonnes:.2f} t/yr")
        else:
            nodes_missing_profile.append(f"{node_name} ({node_type})")
            raise ValueError(
                f"❌ No profile found for {node_name} ({carrier_name}). "
                f"Unsupported or missing source type: '{source}'."
            )

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
def assign_ccs_technologies_debug(network_location, network_emission_flux, path_data_case_study, input_data_path,
                                   technology_selection=None, tech_as_existing=True):
    """
    Enhanced version with comprehensive debugging

    :param dict technology_selection: maps node_type ("Waste", "Cement", "Refining",
        "Other") to a list of technology names (matching filenames under
        italy_data/technologies/Emitter/, without ".json") to make available at that
        sector's nodes.
    :param bool tech_as_existing: if True, each sector's (single) selected technology
        is assigned as "existing" with size fixed at the node's current emitter
        capacity (the original/sunk-asset behavior). If False, all selected
        technologies are assigned as "new" (buildable, freely sized between their own
        JSON size_min/size_max) and compete to meet the sector's carrier demand.
        Technologies in ALWAYS_NEW_TECHNOLOGIES (WasteCaL_CCS, CementHybridCCS) are
        always assigned "new" regardless of this flag - see that constant's comment.
    """
    if technology_selection is None:
        technology_selection = DEFAULT_TECHNOLOGY_SELECTION

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

                selected_techs = technology_selection.get(node_type, [])
                print(f"      Selected technologies for {node_type}: {selected_techs} "
                      f"(as {'existing' if tech_as_existing else 'new'}, "
                      f"forced new regardless: {[t for t in selected_techs if t in ALWAYS_NEW_TECHNOLOGIES]})")
                for tech_name in selected_techs:
                    if tech_as_existing and tech_name not in ALWAYS_NEW_TECHNOLOGIES:
                        existing_techs_dict[tech_name] = capacity
                    else:
                        new_techs_list.append(tech_name)

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

        # Get network data. The three pipeline size classes each get their
        # own data key so a connection can be disabled for one class without
        # affecting the others - see update_network_connection_matrix's
        # mapping for details. Distance itself isn't class-specific (it's a
        # physical fact), but reusing the same masked matrix here keeps a
        # disabled arc's distance at 0 too, which is harmless since
        # connection.csv already gates whether the arc can be built.
        data_mapping = {
            'CO2_Pipeline': 'pipeline',
            'CO2_Pipeline_small': 'pipeline_small',
            'CO2_Pipeline_medium': 'pipeline_medium',
            'CO2_Pipeline_large': 'pipeline_large',
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


# # ===== Enhanced copy_technology_data_custom with debugging =====
# def copy_technology_data_custom_debug(input_data_path, path_files_technologies, network_emission_flux=None):
#     """
#     Enhanced version with comprehensive debugging
#     """
#     print("\n🔍 DEBUG: Starting technology file copying with enhanced debugging...")
#
#     # Read topology
#     with open(input_data_path / "Topology.json", "r") as f:
#         topology = json.load(f)
#
#     print(f"🔍 Topology loaded: {len(topology['nodes'])} nodes, {len(topology['investment_periods'])} periods")
#
#     # Copy technology files for each node in each period
#     for period in topology["investment_periods"]:
#         print(f"\n📁 Processing period: {period}")
#
#         for node in topology["nodes"]:
#             print(f"\n🔧 Processing node: {node}")
#
#             # Read Technologies.json for this node
#             tech_file_path = input_data_path / period / "node_data" / node / "Technologies.json"
#
#             if not tech_file_path.exists():
#                 print(f"  ❌ Technologies.json not found: {tech_file_path}")
#                 continue
#
#             try:
#                 with open(tech_file_path, "r") as f:
#                     technologies_at_node = json.load(f)
#
#                 print(f"  📄 Technologies.json loaded successfully")
#                 print(f"    existing: {technologies_at_node.get('existing', 'NOT_FOUND')}")
#                 print(f"    new: {technologies_at_node.get('new', 'NOT_FOUND')}")
#
#             except Exception as e:
#                 print(f"  ❌ Error reading Technologies.json: {e}")
#                 continue
#
#             # Get technology lists
#             existing_techs = list(technologies_at_node["existing"].keys()) if technologies_at_node["existing"] else []
#
#             if isinstance(technologies_at_node["new"], dict):
#                 new_techs = list(technologies_at_node["new"].keys())
#             elif isinstance(technologies_at_node["new"], list):
#                 new_techs = technologies_at_node["new"]
#             else:
#                 new_techs = []
#
#             all_techs = existing_techs + new_techs
#             print(f"    Total technologies to copy: {len(all_techs)} - {all_techs}")
#
#             # Create technology_data directory
#             tech_data_dir = input_data_path / period / "node_data" / node / "technology_data"
#             tech_data_dir.mkdir(parents=True, exist_ok=True)
#             print(f"    Technology data directory: {tech_data_dir}")
#
#             # Copy each technology file
#             for tech_name in all_techs:
#                 print(f"      🔍 Looking for: {tech_name}")
#
#                 source_file = find_technology_file(tech_name, path_files_technologies)
#                 if source_file:
#                     dest_file = tech_data_dir / f"{tech_name}.json"
#                     try:
#                         shutil.copy2(source_file, dest_file)
#
#                         # Verify the copied file
#                         if dest_file.exists():
#                             file_size = dest_file.stat().st_size
#                             print(f"        ✅ Copied successfully: {source_file} -> {dest_file} ({file_size} bytes)")
#
#                             # Quick validation of JSON content
#                             try:
#                                 with open(dest_file, 'r') as f:
#                                     test_json = json.load(f)
#                                 print(f"        ✅ JSON validation passed")
#                             except Exception as e:
#                                 print(f"        ⚠️  JSON validation failed: {e}")
#                         else:
#                             print(f"        ❌ File not found after copy")
#
#                     except Exception as e:
#                         print(f"        ❌ Copy failed: {e}")
#                 else:
#                     print(f"        ❌ Source file not found in {path_files_technologies}")
#
#     print("\n🔍 Technology file copying debugging completed")


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

