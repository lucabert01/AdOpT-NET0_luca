from pyomo.environ import SolverFactory
import json
import pandas as pd
from pathlib import Path


def get_gurobi_parameters(solveroptions: dict):
    """
    Initiates the gurobi solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory(solveroptions["solver"]["value"], solver_io="python")
    solver.options["TimeLimit"] = solveroptions["timelim"]["value"] * 3600
    solver.options["MIPGap"] = solveroptions["mipgap"]["value"]
    solver.options["MIPFocus"] = solveroptions["mipfocus"]["value"]
    solver.options["Threads"] = solveroptions["threads"]["value"]
    solver.options["NodefileStart"] = solveroptions["nodefilestart"]["value"]
    solver.options["Method"] = solveroptions["method"]["value"]
    solver.options["Heuristics"] = solveroptions["heuristics"]["value"]
    solver.options["Presolve"] = solveroptions["presolve"]["value"]
    solver.options["BranchDir"] = solveroptions["branchdir"]["value"]
    solver.options["LPWarmStart"] = solveroptions["lpwarmstart"]["value"]
    solver.options["IntFeasTol"] = solveroptions["intfeastol"]["value"]
    solver.options["FeasibilityTol"] = solveroptions["feastol"]["value"]
    solver.options["Cuts"] = solveroptions["cuts"]["value"]
    solver.options["NumericFocus"] = solveroptions["numericfocus"]["value"]

    return solver


def get_glpk_parameters(solveroptions: dict):
    """
    Initiates the glpk solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory("glpk")

    return solver


def get_set_t(config: dict, model_block):
    """
    Returns the correct set_t for different clustering options

    :param dict config: config dict
    :param model_block: pyomo block holding set_t_full and set_t_clustered
    :return: set_t
    """
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        return model_block.set_t_full
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        return model_block.set_t_clustered
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        return model_block.set_t_full


def get_data_for_investment_period(
    data, investment_period: str, aggregation_model: str
) -> dict:
    """
    Gets data from DataHandle for specific investement_period. Writes it to a dict.

    :param data: data to use
    :param str investment_period: investment period
    :param str aggregation_model: aggregation type
    :return: data of respective investment period
    :rtype: dict
    """
    data_period = {}
    data_period["period_name"] = investment_period
    data_period["topology"] = data.topology
    data_period["technology_data"] = data.technology_data[investment_period]
    data_period["time_series"] = data.time_series[aggregation_model].loc[
        :, investment_period
    ]
    data_period["network_data"] = data.network_data[investment_period]
    data_period["energybalance_options"] = data.energybalance_options[investment_period]
    data_period["config"] = data.model_config
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_period["k_means_specs"] = data.k_means_specs[investment_period]
        # data_period["averaged_specs"] = data.averaged_specs[investment_period]
    if data.model_config["performance"]["pressure"]["pressure_on"]["value"] == 1:
        data_period["compressor_data"] = data.compressor_data[investment_period]

    # Hour multiplication factors
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] == 0:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 1:
        data_period["hour_factors"] = data_period["k_means_specs"]["factors"]
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 2:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )

    # Nr timesteps averaged
    if data.model_config["optimization"]["timestaging"]["value"] != 0:
        data_period["nr_timesteps_averaged"] = data.model_config["optimization"][
            "timestaging"
        ]["value"]
    else:
        data_period["nr_timesteps_averaged"] = 1

    return data_period


def determine_flow_existing_compressors(self, compressor, b_period, node):
    """
    Determines the flow capacity of an existing compressor connection by returning
    the minimum available capacity between the output and input components

    :param compressor: tuple with carrier, component1, component 2
    :param b_period: pyomo block data for period
    :param node: pyomo block data for node
    :return float: minimum capacity between input and output component
    """
    component_output_bound = float("inf")
    component_input_bound = float("inf")
    period_name = b_period.name.split("[")[-1].rstrip("]")
    type_component = [compressor.output_type, compressor.input_type]

    if type_component[0] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.output_component]
            .var_output
        )
        component_output_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[0] == "Network":
        component_output_bound = next(
            iter(
                b_period.network_block[
                    compressor.output_component
                ].para_size_initial.values()
            )
        )
    elif type_component[0] == "Import":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Import limit"]
        )

    elif type_component[0] == "Generic production":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Generic production"]
        )

    if type_component[1] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.input_component]
            .var_output
        )
        component_input_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[1] == "Network":
        component_input_bound = next(
            iter(
                b_period.network_block[
                    compressor.input_component
                ].para_size_initial.values()
            )
        )
    elif type_component[1] == "Demand":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Demand"]
        )
    elif type_component[1] == "Export":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Export limit"]
        )

    size = min(component_output_bound, component_input_bound)

    return size


def create_csv_database_from_json():
    """
    Creates an Excel database from all JSON files in the database/templates directory.
    The Excel file contains comprehensive information about all components (technologies and networks)
    and is saved as 'Components database.xlsx' in the database/data directory.
    """
    # Define paths
    base_path = Path(__file__).parent / "database" / "templates"
    output_path = (
        Path(__file__).parent / "database" / "data" / "Components database.xlsx"
    )

    # Initialize lists to store processed data
    all_components = []

    # Process technology data files
    tech_data_path = base_path / "technology_data"
    if tech_data_path.exists():
        for json_file in tech_data_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract basic information
                component_info = {
                    "Component_Type": "Technology",
                    "Category": json_file.parent.name,
                    "File_Name": json_file.name,
                    "Component_Name": data.get("tec_type", "N/A"),
                    "Decommission": data.get("decommission", "N/A"),
                    "Size_Is_Int": data.get("size_is_int", "N/A"),
                    "Size_Min": data.get("size_min", "N/A"),
                    "Size_Max": data.get("size_max", "N/A"),
                }

                # Extract Economics data
                economics = data.get("Economics", {})
                component_info.update(
                    {
                        "CAPEX_Model": economics.get("capex_model", "N/A"),
                        "Unit_CAPEX": economics.get("unit_capex", "N/A"),
                        "Fix_CAPEX": economics.get("fix_capex", "N/A"),
                        "OPEX_Variable": economics.get("opex_variable", "N/A"),
                        "OPEX_Fixed": economics.get("opex_fixed", "N/A"),
                        "Discount_Rate": economics.get("discount_rate", "N/A"),
                        "Lifetime": economics.get("lifetime", "N/A"),
                        "Decommission_Cost": economics.get("decommission_cost", "N/A"),
                    }
                )

                # Extract Performance data
                performance = data.get("Performance", {})
                component_info.update(
                    {
                        "Input_Carrier": (
                            ", ".join(performance.get("input_carrier", []))
                            if isinstance(performance.get("input_carrier"), list)
                            else performance.get("input_carrier", "N/A")
                        ),
                        "Output_Carrier": (
                            ", ".join(performance.get("output_carrier", []))
                            if isinstance(performance.get("output_carrier"), list)
                            else performance.get("output_carrier", "N/A")
                        ),
                        "Main_Input_Carrier": performance.get(
                            "main_input_carrier", "N/A"
                        ),
                        "Rated_Power": performance.get("rated_power", "N/A"),
                        "Efficiency": performance.get("efficiency", "N/A"),
                        "Emission_Factor": performance.get("emission_factor", "N/A"),
                    }
                )

                # Extract Units data
                units = data.get("Units", {})
                component_info.update(
                    {
                        "Size_Unit": units.get("size", "N/A"),
                    }
                )

                all_components.append(component_info)

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue

    # Process network data files
    network_data_path = base_path / "network_data"
    if network_data_path.exists():
        for json_file in network_data_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract basic information
                component_info = {
                    "Component_Type": "Network",
                    "Category": "Network",
                    "File_Name": json_file.name,
                    "Component_Name": data.get("network_type", "N/A"),
                    "Decommission": data.get("decommission", "N/A"),
                    "Size_Is_Int": data.get("size_is_int", "N/A"),
                    "Size_Min": data.get("size_min", "N/A"),
                    "Size_Max": data.get("size_max", "N/A"),
                }

                # Extract Economics data
                economics = data.get("Economics", {})
                component_info.update(
                    {
                        "CAPEX_Model": "N/A",  # Networks use gamma coefficients
                        "Unit_CAPEX": "N/A",
                        "Fix_CAPEX": "N/A",
                        "OPEX_Variable": economics.get("opex_variable", "N/A"),
                        "OPEX_Fixed": economics.get("opex_fixed", "N/A"),
                        "Discount_Rate": economics.get("discount_rate", "N/A"),
                        "Lifetime": economics.get("lifetime", "N/A"),
                        "Decommission_Cost": economics.get("decommission_cost", "N/A"),
                    }
                )

                # Extract Performance data
                performance = data.get("Performance", {})
                component_info.update(
                    {
                        "Input_Carrier": performance.get("carrier", "N/A"),
                        "Output_Carrier": performance.get("carrier", "N/A"),
                        "Main_Input_Carrier": performance.get("carrier", "N/A"),
                        "Rated_Power": "N/A",
                        "Efficiency": f"Loss: {performance.get('loss', 'N/A')}",
                        "Emission_Factor": performance.get("emissionfactor", "N/A"),
                    }
                )

                # Extract Units data
                units = data.get("Units", {})
                component_info.update(
                    {
                        "Size_Unit": units.get("size", "N/A"),
                    }
                )

                # Add network-specific gamma coefficients
                component_info.update(
                    {
                        "Gamma1": economics.get("gamma1", "N/A"),
                        "Gamma2": economics.get("gamma2", "N/A"),
                        "Gamma3": economics.get("gamma3", "N/A"),
                        "Gamma4": economics.get("gamma4", "N/A"),
                    }
                )

                all_components.append(component_info)

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue

    # Create DataFrame
    if all_components:
        df = pd.DataFrame(all_components)

        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Write all components to main sheet
            df.to_excel(writer, sheet_name="All Components", index=False)

            # Create separate sheets for technologies and networks
            tech_df = df[df["Component_Type"] == "Technology"]
            network_df = df[df["Component_Type"] == "Network"]

            if not tech_df.empty:
                tech_df.to_excel(writer, sheet_name="Technologies", index=False)

            if not network_df.empty:
                network_df.to_excel(writer, sheet_name="Networks", index=False)

            # Create summary sheet with category breakdown
            summary_data = []
            if not tech_df.empty:
                tech_summary = tech_df["Category"].value_counts().to_dict()
                for category, count in tech_summary.items():
                    summary_data.append(
                        {"Type": "Technology", "Category": category, "Count": count}
                    )

            if not network_df.empty:
                summary_data.append(
                    {"Type": "Network", "Category": "Network", "Count": len(network_df)}
                )

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

        print(f"Database created successfully with {len(all_components)} components!")
        print(f"Excel file saved to: {output_path}")
        return output_path

    else:
        print("No components found to process!")
        return None
