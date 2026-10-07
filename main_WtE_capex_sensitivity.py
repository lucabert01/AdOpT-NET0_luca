import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import os

# Technology selection (MEA vs CaL) as a function of the capex of the two capture technologies.
# x-axis: capex of MEA relative to its baseline (1 = baseline, 1.5 = +50%, ...)
# y-axis: capex of CaL relative to its baseline (1 = baseline, 0.5 = -50%, ...)
# The baseline data (json files in dataCaseStudy_WtE and wasteCaL_sheet.xlsx) are never modified: the multipliers
# are only written in the json files copied to the case study folder.

# All paths are relative to the repository, independently of where the script is launched from
os.chdir(Path(__file__).resolve().parent)

# Specify the path to your input data
casepath = Path("CaseStudies_WtE/capex_sensitivity")
json_files_path = Path("./dataCaseStudy_WtE/technologies_json")
json_files_path_network = Path("./dataCaseStudy_WtE/network_json")
result_path = Path("dataCaseStudy_WtE/raw_results/capex_sensitivity")

os.makedirs(casepath, exist_ok=True)
adopt.create_optimization_templates(casepath)

# Import data from the json file
info_wasteCHP = json.loads((json_files_path / "WasteCHP.json").read_text())
info_WasteCaL_CCS = json.loads((json_files_path / "WasteCaL_CCS.json").read_text())
ccs_type = info_wasteCHP["Performance"]["ccs"]["ccs_type"]
lhv = info_wasteCHP["Performance"]["LHV"]
th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
emission_factor = info_wasteCHP["Performance"]["emission_factor"]
path_processed_data = Path("./dataCaseStudy_WtE/dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)

# General input data
carbon_tax = 150
dh_ratio = 0.5
explored_mea_capex_multiplier = [1, 1.5, 2]
explored_cal_capex_multiplier = [1, 0.5, 0.25]
skip_solved_cases = 1  # do not re-run cases that already have results in result_path
plant_analyzed = "PAIP" # one between: "silla2", "gerbido", "PAIP", "piacenza"
gas_price = 40
import_price_RDF = 20
existing_boiler_size = max(data[f"emission_{plant_analyzed}"])/emission_factor*lhv*th_efficiency
wte_demand_is_averaged = 0
heat_demand_is_averaged = 0
rolling_av_hours = 24*7
co2_concentration = data["co2_concentration_"+plant_analyzed]
distance_to_stor = 100
end_period = 8760
pyhub = {}

# Baseline electricity price (northern Italy)
path_processed_data = Path("dataCaseStudy_Cement/dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
electricity_price = electricity_price_data["el_price_itNord"]
av_el_price = electricity_price.mean()


# Matrix of the capex multipliers
capex_matrix = []
for cal_multiplier in explored_cal_capex_multiplier:
    for mea_multiplier in explored_mea_capex_multiplier:
        capex_matrix.append(
            {
                "case_name": f"mea_{mea_multiplier:.2f}_cal_{cal_multiplier:.2f}_ctax_{carbon_tax}",
                "mea_capex_multiplier": mea_multiplier,
                "cal_capex_multiplier": cal_multiplier,
            }
        )
capex_matrix = pd.DataFrame(capex_matrix)
no_ccs_case_name = f"noCCS_ctax_{carbon_tax}"
print(capex_matrix.to_string())

os.makedirs(result_path, exist_ok=True)
capex_matrix.to_csv(result_path / "capex_matrix.csv", sep=";", index=False)
with open(result_path / "capex_matrix_info.json", "w") as json_file:
    json.dump(
        {
            "carbon_tax": carbon_tax,
            "av_el_price": av_el_price,
            "dh_ratio": dh_ratio,
            "no_ccs_case_name": no_ccs_case_name,
        },
        json_file,
        indent=4,
    )


def case_is_solved(case_name):
    return any((d / "optimization_results.h5").exists() for d in result_path.glob(f"*_{case_name}"))


def run_case(case_name, mea_capex_multiplier, cal_capex_multiplier, ccs_possible):
    """
    Builds and solves one case

    :param str case_name: name of the case (used for the result folder)
    :param float mea_capex_multiplier: multiplier of the baseline capex of MEA
    :param float cal_capex_multiplier: multiplier of the baseline capex of CaL
    :param int ccs_possible: if 0, the WtE plant without CCS is optimized (benchmark for the cost of avoidance)
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
        "wasteIn",
        "wasteProcessed",
        "wasteInRDF",
        "gas",
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
    technologies["new"] = ["WasteCHP", "WasteCaL_CCS"] if ccs_possible else ["WasteCHP"]
    technologies["existing"] = {"Boiler_Industrial_NG": existing_boiler_size}

    with open(
            casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
    ) as json_file:
        json.dump(technologies, json_file, indent=4)

    # Copy over technology files
    adopt.copy_technology_data(casepath, json_files_path)

    # Scale the capex in the copied json files. All of them are written starting from the baseline data, as
    # copy_technology_data does not overwrite the json file of the CCS if it is already in the folder
    tec_data_path = casepath / "period1" / "node_data" / "industrial_cluster" / "technology_data"

    case_wasteCHP = json.loads((json_files_path / "WasteCHP.json").read_text())
    case_wasteCHP["Performance"]["ccs"]["possible"] = ccs_possible
    (tec_data_path / "WasteCHP.json").write_text(json.dumps(case_wasteCHP, indent=4))

    case_mea = json.loads((json_files_path / f"{ccs_type}.json").read_text())
    for capex_parameter in ["unit_capex", "capex_kappa", "capex_lambda", "capex_zeta"]:
        case_mea["Economics"][capex_parameter] = case_mea["Economics"][capex_parameter] * mea_capex_multiplier
    (tec_data_path / f"{ccs_type}.json").write_text(json.dumps(case_mea, indent=4))

    if ccs_possible:
        case_WasteCaL_CCS = json.loads((json_files_path / "WasteCaL_CCS.json").read_text())
        case_WasteCaL_CCS["Economics"]["capex_multiplier"] = cal_capex_multiplier
        (tec_data_path / "WasteCaL_CCS.json").write_text(json.dumps(case_WasteCaL_CCS, indent=4))

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
    arc_size.to_csv(
        casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "size_max_arcs.csv",
        sep=";")

    # Use the templates, fill and save them to the respective directory
    # Connection
    connection = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                             index_col=0)
    connection.loc["industrial_cluster", "storage"] = 1
    connection.to_csv(
        casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "connection.csv",
        sep=";")

    # Delete the template
    os.remove(casepath / "period1" / "network_topology" / "new" / "connection.csv")

    # Distance
    distance = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                           index_col=0)
    distance.loc["industrial_cluster", "storage"] = distance_to_stor
    distance.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "distance.csv",
                    sep=";")

    # Delete the template
    os.remove(casepath / "period1" / "network_topology" / "new" / "distance.csv")

    # Delete the max_size_arc template
    os.remove(casepath / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

    # Import hourly profiles
    if wte_demand_is_averaged:
        wasteProcessed_demand = data[f"waste_in_{plant_analyzed}"].rolling(window=rolling_av_hours,
                                                                           min_periods=1).mean()
    else:
        wasteProcessed_demand = data[f"waste_in_{plant_analyzed}"]
    norm_heat_demand = data["normalized_heat_demand_milan"]
    max_useful_heat_output = (wasteProcessed_demand.mean() * info_WasteCaL_CCS["Performance"]["LHV"]
                              * info_WasteCaL_CCS["Performance"]["th_efficiency"])
    peak_heat_demand = dh_ratio * max_useful_heat_output
    if heat_demand_is_averaged:
        heat_demand = (norm_heat_demand * peak_heat_demand).rolling(window=rolling_av_hours, min_periods=1).mean()
    else:
        heat_demand = (norm_heat_demand * peak_heat_demand)

    # Set import limits/cost
    adopt.fill_carrier_data(
        casepath,
        value_or_data=1000,
        columns=["Export limit"],
        carriers=["electricity"],
        nodes=["industrial_cluster"],
    )
    adopt.fill_carrier_data(
        casepath,
        value_or_data=1000,
        columns=["Export limit"],
        carriers=["heat"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=electricity_price,
        columns=["Export price"],
        carriers=["electricity"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=1000,
        columns=["Import limit"],
        carriers=["wasteIn"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=1000,
        columns=["Import limit"],
        carriers=["wasteInRDF"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=5000,
        columns=["Import limit"],
        carriers=["gas"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=gas_price,
        columns=["Import price"],
        carriers=["gas"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=import_price_RDF,
        columns=["Import price"],
        carriers=["wasteInRDF"],
        nodes=["industrial_cluster"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=wasteProcessed_demand,
        columns=["Demand"],
        carriers=["wasteProcessed"],
        nodes=["industrial_cluster"],
    )
    adopt.fill_carrier_data(
        casepath,
        value_or_data=heat_demand,
        columns=["Demand"],
        carriers=["heat"],
        nodes=["industrial_cluster"],
    )
    adopt.fill_carrier_data(
        casepath,
        value_or_data=1000,
        columns=["Import limit"],
        carriers=["electricity"],
        nodes=["storage"],
    )

    adopt.fill_carrier_data(
        casepath,
        value_or_data=0,
        columns=["Import price"],
        carriers=["electricity"],
        nodes=["storage"],
    )

    tech_with_hourly_co2_concentration = ["WasteCHP", "WasteCaL_CCS"]
    climate_data_file = (
            casepath / "period1" / "node_data" / "industrial_cluster" / "ClimateData.csv"
    )
    climate_data = pd.read_csv(climate_data_file)
    for tech in tech_with_hourly_co2_concentration:
        climate_data["co2_concentration_"+ tech] = co2_concentration.values
    climate_data.to_csv(climate_data_file, index=False, sep=";")

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


# Benchmark without CCS (does not depend on the capex of the capture technologies)
if not (skip_solved_cases and case_is_solved(no_ccs_case_name)):
    run_case(no_ccs_case_name, 1, 1, ccs_possible=0)

for _, case in capex_matrix.iterrows():
    if skip_solved_cases and case_is_solved(case["case_name"]):
        print(f"Skipping {case['case_name']}: already solved")
        continue
    run_case(case["case_name"], case["mea_capex_multiplier"], case["cal_capex_multiplier"], ccs_possible=1)
    # free the memory of the solved model
    del pyhub[case["case_name"]]
