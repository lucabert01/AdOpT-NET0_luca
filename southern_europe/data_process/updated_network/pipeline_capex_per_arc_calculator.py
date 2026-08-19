import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utilities/')))

try:
    from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import \
        CO2_Pipeline_CostModel as EnhancedModel
except ImportError as e:
    print(f"⚠️  Warning: Could not import EnhancedModel: {e}")
    print("   This is required for gamma calculations. Please check your adopt_net0 installation.")

# Import shared functions from arc_specific_functions
from arc_specific_functions import (
    load_network_data,
    load_intersection_data,
    get_node_emission,
    calculate_total_annual_emission,
    calculate_global_max_massflow,
    calculate_global_min_massflow,
    calculate_arc_gammas,
    create_gamma_matrices,
    create_zero_gamma_matrices,
    get_pipeline_length,
    validate_pipeline_transport,
    get_all_possible_arcs,
    create_base_options,
    add_geographical_options,
    print_section_header,
    save_to_excel,
    determine_arc_terrain,
    load_massflow_overrides,
)

# ============================================================================
# PIPELINE SIZE CLASSES
# ============================================================================
# Three separate CO2_Pipeline_{small,medium,large} network technologies (see
# italy_data/networks/CO2_Pipeline_{small,medium,large}.json) each get their
# own gamma1/gamma2 matrix, computed from a FIXED mass-flow range (kg/s)
# applied to every arc for that class - unlike the old single-CO2_Pipeline
# approach, which derived a different min/max per arc from that arc's own
# emissions.
#
# Derived from the arcs actually built in
# Results_CCSchainOptimization/20260711184007_emissions_minC-1 (41 arcs, used
# as a realistic sample of pipeline sizes since geography drives topology far
# more than the exact cost coefficients). For each, the TRUE capex at that
# arc's real flow was recomputed with the bug-fixed Oeuvray model and
# normalized by length (capex_per_km vs flow_kg_s), giving
# capex_per_km ~= 199,080 * flow_kg_s^0.433 (log-log R2=0.73) - strong
# economies of scale, steep at small flow and flattening at large flow. The
# two breakpoints below minimize the piecewise-linear approximation error of
# that curve (i.e. chosen so each class's straight-line gamma1+gamma2*size
# fit is a good local approximation), then padded to the network's true
# floor (~3.1 kg/s, smallest single emitter) and ceiling (~466 kg/s, total
# network emissions) so nothing falls outside all three ranges.
SIZE_CLASS_MASSFLOW_RANGES_KG_S = {
    "small": (3.1, 29.0),
    "medium": (29.0, 133.0),
    "large": (133.0, 470.0),
}


def compute_gammas_for_size_class(size_class, fixed_range_kg_s, data_dict, possible_arcs,
                                   output_dir, dashboard_overrides_path):
    """
    Compute gamma1/gamma2 for every arc for one pipeline size class and save
    to capex_defined_per_arc_{size_class}.xlsx.

    Args:
        size_class: "small" | "medium" | "large" (used for the output filename)
        fixed_range_kg_s: (min_kg_s, max_kg_s) applied to every arc for this
            class, or None to fall back to the automatic per-arc derivation
            (plus any dashboard overrides).
        data_dict: Network data dict from load_network_data (with
            intersection_data already attached)
        possible_arcs: List of (from_node, to_node) tuples
        output_dir: network_capex_metrics directory
        dashboard_overrides_path: Path to massflow_overrides_per_arc.xlsx,
            only consulted when fixed_range_kg_s is None

    Returns:
        Path to the saved capex_defined_per_arc_{size_class}.xlsx file
    """
    print_section_header(f"PIPELINE SIZE CLASS: {size_class.upper()}")

    if fixed_range_kg_s is not None:
        min_kg_s, max_kg_s = fixed_range_kg_s
        print(f"🖊️  Fixed mass-flow range for '{size_class}': {min_kg_s:.3f} - {max_kg_s:.3f} kg/s "
              f"(applied to all {len(possible_arcs)} arcs)")
        massflow_overrides = {arc: (min_kg_s, max_kg_s) for arc in possible_arcs}
    else:
        print(f"⚠️  No fixed range set for '{size_class}' yet (SIZE_CLASS_MASSFLOW_RANGES_KG_S) - "
              f"falling back to automatic per-arc min/max derivation")
        massflow_overrides = load_massflow_overrides(dashboard_overrides_path)
        if massflow_overrides:
            print(f"🖊️  Loaded {len(massflow_overrides)} manual mass-flow override(s) from {dashboard_overrides_path}")

    gamma1_matrix, gamma2_matrix = create_gamma_matrices(
        data_dict, terrain_function=determine_arc_terrain, massflow_overrides=massflow_overrides
    )
    gamma3_matrix, gamma4_matrix = create_zero_gamma_matrices(gamma1_matrix)

    gamma1_nonzero = (gamma1_matrix != 0).sum().sum()
    gamma2_nonzero = (gamma2_matrix != 0).sum().sum()
    print(f"\n📊 '{size_class}' gamma calculation results:")
    print(f"   Gamma1 non-zero values: {gamma1_nonzero} / {len(possible_arcs)} arcs")
    print(f"   Gamma2 non-zero values: {gamma2_nonzero} / {len(possible_arcs)} arcs")

    if gamma1_nonzero > 0:
        v1 = gamma1_matrix.values.flatten(); v1 = v1[v1 != 0]
        v2 = gamma2_matrix.values.flatten(); v2 = v2[v2 != 0]
        print(f"   Gamma1: min={v1.min():,.0f}, max={v1.max():,.0f}, mean={v1.mean():,.0f} EUR")
        print(f"   Gamma2: min={v2.min():.3f}, max={v2.max():.3f}, mean={v2.mean():.3f} EUR/(t/h)")

    output_file = output_dir / f"capex_defined_per_arc_{size_class}.xlsx"
    save_to_excel(gamma1_matrix, gamma2_matrix, gamma3_matrix, gamma4_matrix, filename=str(output_file))
    print(f"📁 Saved: {output_file.absolute()}")

    return output_file


def main():
    """Main execution function"""

    print_section_header("FULL NETWORK ARC GAMMA CALCULATOR (3 PIPELINE SIZE CLASSES)")
    print("This script calculates gamma1/gamma2 for every possible arc, once per")
    print("pipeline size class (small/medium/large), and saves each as its own")
    print("capex_defined_per_arc_{size_class}.xlsx. Gamma3/gamma4 are zeros.")
    print("\n🌊 SPECIAL CONFIGURATION: Arc Eni S.p.A Casalborsetti -> Porto Corsini will be processed as OFFSHORE terrain")
    print("🏞️  All other arcs will be processed as ONSHORE terrain")
    print("\n🔇 VERBOSE OUTPUT: Cost model calculations are suppressed for cleaner output")
    print("📊 UPDATED BEHAVIOR: Failed calculations will be set to 0 instead of NaN")

    try:
        # Load all data ONCE (geo data / electricity prices / emissions are
        # shared across all three size classes)
        print("\n🔄 Loading network data...")
        data_dict = load_network_data("../../italy_data")

        print(f"\n📊 Data verification:")
        print(f"   Nodes: {len(data_dict['network_nodes'])}")
        print(f"   Distance matrix: {data_dict['network_distance'].shape}")
        print(f"   Pipeline matrix: {data_dict['network_pipeline'].shape}")

        if 'annual_emission' not in data_dict['network_emission_flux'].columns:
            print("⚠️  Warning: Annual emission data not found. Please check data loading.")
            print(f"   Available columns in emission data: {list(data_dict['network_emission_flux'].columns)}")
            return

        print("\n🔄 Loading intersection data...")
        intersection_file_path = Path("../../italy_data/geographical_feature/route_grid_intersections.xlsx")
        possible_arcs = get_all_possible_arcs(data_dict['network_pipeline'])
        pipeline_names = [f"{from_node}_{to_node}" for from_node, to_node in possible_arcs]
        print(f"   Found {len(possible_arcs)} possible arcs")

        data_dict['intersection_data'] = load_intersection_data(intersection_file_path, pipeline_names)

        offshore_arcs = [(f, t) for f, t in possible_arcs if determine_arc_terrain(f, t, data_dict) == "Offshore"]
        print(f"   🌊 Offshore arcs: {offshore_arcs}" if offshore_arcs else "   ℹ️  No offshore arcs found")

        output_dir = Path("../../italy_data/network_capex_metrics")
        output_dir.mkdir(parents=True, exist_ok=True)
        dashboard_overrides_path = output_dir / "massflow_overrides_per_arc.xlsx"

        output_files = {}
        for size_class, fixed_range_kg_s in SIZE_CLASS_MASSFLOW_RANGES_KG_S.items():
            output_files[size_class] = compute_gammas_for_size_class(
                size_class, fixed_range_kg_s, data_dict, possible_arcs,
                output_dir, dashboard_overrides_path,
            )

        print_section_header("ANALYSIS COMPLETE")
        for size_class, path in output_files.items():
            print(f"   {size_class:8s} -> {path}")
        print("\n🔢 Missing/failed calculations are set to 0 instead of NaN")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

        print(f"\n🔍 Debugging information:")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Data path exists: {Path('../../italy_data').exists()}")
        print(
            f"   Excel file exists: {Path('../../italy_data/geographical_feature/node_metrics.xlsx').exists()}")
        print(f"   Output directory: {Path('../../italy_data/network_capex_metrics').exists()}")

        try:
            from adopt_net0.database.components.networks.enhanced_co2_pipelines_cost_model import CO2_Pipeline_CostModel
            print(f"   Enhanced cost model: Available")
        except ImportError:
            print(f"   Enhanced cost model: NOT available (required for gamma calculations)")

        try:
            from southern_europe.data_process.utilities.defined_functions import calculate_annual_emission_values
            print(f"   Defined functions module: Available")
        except ImportError:
            print(f"   Defined functions module: NOT available")


if __name__ == "__main__":
    main()
