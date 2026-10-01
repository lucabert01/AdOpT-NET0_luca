import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import os

# Technology selection (MEA vs oxyfuel, with the option of a small MEA) as a function of the capex of the
# capture technologies.
# x-axis: capex of MEA relative to its baseline (1 = baseline, 1.5 = +50%, ...). It is applied to the MEA of
#         CementEmitter and to the small MEA + compressor of CementHybridCCS
# y-axis: capex of oxyfuel relative to its baseline. It is applied to the oxyfuel + CPU of CementHybridCCS
# The baseline data (json files in dataCaseStudy_Cement and cement_sheet.xlsx) are never modified: the multipliers
# are only written in the json files copied to the case study folder.

# All paths are relative to the repository, independently of where the script is launched from
os.chdir(Path(__file__).resolve().parent)

# Specify the path to your input data
casepath = Path("./CaseStudy_Cement_capex_sensitivity")
json_files_path = Path("dataCaseStudy_Cement/technologies_json")
json_files_path_network = Path("dataCaseStudy_Cement/network_json")
result_path = Path("dataCaseStudy_Cement/raw_results/capex_sensitivity")

os.makedirs(casepath, exist_ok=True)
adopt.create_optimization_templates(casepath)

# Import data from the json file
info_cement = json.loads((json_files_path / "CementEmitter.json").read_text())
ccs_type = info_cement["Performance"]["ccs"]["ccs_type"]

# General input data
possible_plants = ["Vernasca", "Robilante", "Monselice", "Fanna"]
plant_analyzed = "Vernasca"
carbon_tax = 200
explored_mea_capex_multiplier = [1, 1.5, 2]
explored_oxy_capex_multiplier = [1, 1.5, 2]
skip_solved_cases = 1  # do not re-run cases that already have results in result_path
distance_to_stor = 100
dymanics_on = 0
end_period = 8760
cost_extra_fuel = 15
pyhub = {}

# Baseline electricity price (northern Italy)
path_processed_data = Path("dataCaseStudy_Cement/dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
electricity_price = electricity_price_data["el_price_itNord"]
av_el_price = electricity_price.mean()

clinker_data = pd.read_excel(path_processed_data, sheet_name="clinker_production")
clinker_demand = clinker_data[f"clinker_{plant_analyzed}"]


# Matrix of the capex multipliers
capex_matrix = []
for oxy_multiplier in explored_oxy_capex_multiplier:
    for mea_multiplier in explored_mea_capex_multiplier:
        capex_matrix.append(
            {
                "case_name": f"mea_{mea_multiplier:.2f}_oxy_{oxy_multiplier:.2f}",
                "mea_capex_multiplier": mea_multiplier,
                "oxy_capex_multiplier": oxy_multiplier,
            }
        )
capex_matrix = pd.DataFrame(capex_matrix)
print(capex_matrix.to_string())

os.makedirs(result_path, exist_ok=True)
capex_matrix.to_csv(result_path / "capex_matrix.csv", sep=";", index=False)
with open(result_path / "capex_matrix_info.json", "w") as json_file:
    json.dump(
        {
            "carbon_tax": carbon_tax,
            "av_el_price": av_el_price,
            "plant_analyzed": plant_analyzed,
        },
        json_file,
        indent=4,
    )


def case_is_solved(case_name):
    return any((d / "optimization_results.h5").exists() for d in result_path.glob(f"*_{case_name}"))


def run_case(case_name, mea_capex_multiplier, oxy_capex_multiplier):
    """
    Builds and solves one case

    :param str case_name: name of the case (used for the result folder)
    :param float mea_capex_multiplier: multiplier of the baseline capex of MEA
    :param float oxy_capex_multiplier: multiplier of the baseline capex of oxyfuel
    """
    # Load json template
    with open(casepath / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    # Nodes
    topology["nodes"] = ["storage", "industrial_cluster"]
    # Carriers:
    topology["carriers"] = [
        "electricity",
        "CO2captured",
        "heat",
        "extra_fuel",
        "gas",
        "clinker",
        "limestone"
    ]
    # Investment periods:
    topology["investment_periods"] = ["period1"]
    # Save json template
    with open(casepath / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)

    # Load json template
    with open(casepath / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)
    # Change objective
    configuration["optimization"]["objective"]["value"] = "costs"
    # Set MILP gap
    configuration["solveroptions"]["mipgap"]["value"] = 0.02
    configuration["performance"]["dynamics"]["value"] = dymanics_on
    # change save options
    configuration['reporting']['save_summary_path']['value'] = str(result_path)
    configuration['reporting']['save_path']['value'] = str(result_path)
    # Save json template
    with open(casepath / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)

    adopt.create_input_data_folder_template(casepath)

    node_location = pd.read_csv(casepath / "NodeLocations.csv", sep=";", index_col=0, header=0)
    for node in topology["nodes"]:
        node_location.at[node, "lon"] = 10
        node_location.at[node, "lat"] = 10
        node_location.at[node, "alt"] = 10
    node_location = node_location.reset_index()
    node_location.to_csv(casepath / "NodeLocations.csv", sep=";", index=False)

    # Add technologies
    with open(casepath / "period1" / "node_data" / "storage" / "Technologies.json", "r") as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["PermanentStorage_CO2_simple"]

    with open(casepath / "period1" / "node_data" / "storage" / "Technologies.json", "w") as json_file:
        json.dump(technologies, json_file, indent=4)

    with open(
        casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "r"
    ) as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["CementHybridCCS", "CementEmitter", "HeatPump"]

    with open(
        casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
    ) as json_file:
        json.dump(technologies, json_file, indent=4)

    # Copy over technology files
    adopt.copy_technology_data(casepath, json_files_path)

    # Scale the capex in the copied json files. All of them are written starting from the baseline data, as
    # copy_technology_data does not overwrite the json file of the CCS if it is already in the folder
    tec_data_path = casepath / "period1" / "node_data" / "industrial_cluster" / "technology_data"

    case_mea = json.loads((json_files_path / f"{ccs_type}.json").read_text())
    for capex_parameter in ["unit_capex", "capex_kappa", "capex_lambda", "capex_zeta"]:
        case_mea["Economics"][capex_parameter] = case_mea["Economics"][capex_parameter] * mea_capex_multiplier
    (tec_data_path / f"{ccs_type}.json").write_text(json.dumps(case_mea, indent=4))

    case_oxy_ccs = json.loads((json_files_path / "CementHybridCCS.json").read_text())
    case_oxy_ccs["Economics"]["other_economics"]["capex_multiplier_oxy"] = oxy_capex_multiplier
    case_oxy_ccs["Economics"]["other_economics"]["capex_multiplier_MEA"] = mea_capex_multiplier
    (tec_data_path / "CementHybridCCS.json").write_text(json.dumps(case_oxy_ccs, indent=4))

    # Add networks
    with open(casepath / "period1" / "Networks.json", "r") as json_file:
        networks = json.load(json_file)
    networks["new"] = ["CO2PipelineOnshore"]

    with open(casepath / "period1" / "Networks.json", "w") as json_file:
        json.dump(networks, json_file, indent=4)

    adopt.copy_network_data(casepath, json_files_path_network)

    # Make a new folder for the new network
    os.makedirs(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore", exist_ok=True)
    # max size arc
    arc_size = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "size_max_arcs.csv", sep=";",
                           index_col=0)
    arc_size.loc["industrial_cluster", "storage"] = 10000
    arc_size.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "size_max_arcs.csv",
                    sep=";")

    # Use the templates, fill and save them to the respective directory
    # Connection
    connection = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "connection.csv", sep=";", index_col=0)
    connection.loc["industrial_cluster", "storage"] = 1
    connection.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "connection.csv",
                      sep=";")

    # Delete the template
    os.remove(casepath / "period1" / "network_topology" / "new" / "connection.csv")

    # Distance
    distance = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "distance.csv", sep=";", index_col=0)
    distance.loc["industrial_cluster", "storage"] = distance_to_stor
    distance.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "distance.csv", sep=";")

    # Delete the template
    os.remove(casepath / "period1" / "network_topology" / "new" / "distance.csv")

    # Delete the max_size_arc template
    os.remove(casepath / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # Set import limits/cost
    adopt.fill_carrier_data(
        casepath,
        value_or_data=5000,
        columns=["Import limit"],
        carriers=["electricity"],
        nodes=["industrial_cluster", "storage"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=5000,
        columns=["Import limit"],
        carriers=["extra_fuel"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=5000,
        columns=["Import limit"],
        carriers=["limestone"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=cost_extra_fuel,
        columns=["Import price"],
        carriers=["extra_fuel"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=electricity_price,
        columns=["Import price"],
        carriers=["electricity"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=clinker_demand,
        columns=["Demand"],
        carriers=["clinker"],
        nodes=["industrial_cluster"],
    )

    carbon_price = np.ones(8760) * carbon_tax
    carbon_cost_path = (
        casepath / "period1" / "node_data" / "industrial_cluster" / "CarbonCost.csv"
    )
    carbon_cost_template = pd.read_csv(carbon_cost_path, sep=";", index_col=0, header=0)
    carbon_cost_template["price"] = carbon_price
    carbon_cost_template = carbon_cost_template.reset_index()
    carbon_cost_template.to_csv(carbon_cost_path, sep=";", index=False)

    # Construct and solve the model
    pyhub[case_name] = adopt.ModelHub()
    pyhub[case_name].read_data(casepath, start_period=0, end_period=end_period)
    pyhub[case_name].data.model_config['reporting']['case_name']['value'] = case_name
    pyhub[case_name].construct_model()
    pyhub[case_name].construct_balances()
    pyhub[case_name].solve()


for _, case in capex_matrix.iterrows():
    if skip_solved_cases and case_is_solved(case["case_name"]):
        print(f"Skipping {case['case_name']}: already solved")
        continue
    run_case(case["case_name"], case["mea_capex_multiplier"], case["oxy_capex_multiplier"])
    # free the memory of the solved model
    del pyhub[case["case_name"]]
