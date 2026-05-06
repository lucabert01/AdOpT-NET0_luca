# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import os
from dataCaseStudy_Cement.process_data_cement_emission import create_norm_el_price_profiles


# Specify the path to your input data
casepath = Path("./CaseStudy_Cement_flexible")
json_files_path = Path("dataCaseStudy_Cement/technologies_json_flexible")
json_files_path_network = Path("dataCaseStudy_Cement/network_json")
result_path = "./dataCaseStudy_Cement/raw_results/flexible_ops"

adopt.create_optimization_templates(casepath)

# General input data
objective_function = "costs" # "emissions_net", "emissions_minC", "costs"
possible_plants = ["Vernasca", "Robilante", "Monselice", "Fanna"]
plant_analyzed = "Vernasca"
explored_std = [1, 2]
explored_el_price = [50, 107, 150] # average el prices explored in the analysis
explored_technologies = ["mea","mea_inflex"]
distance_to_stor = 100
carbon_tax = 200
dymanics_on = 1

cost_extra_fuel = 15
path_processed_data = Path("dataCaseStudy_Cement/dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
el_price_base = electricity_price_data["el_price_itNord"]
generated_el_profiles = pd.read_excel(path_processed_data, sheet_name="el_prices_norm_stds")
create_norm_el_price_profiles(el_price_base, explored_std)

clinker_data = pd.read_excel(path_processed_data, sheet_name="clinker_production")
clinker_demand = clinker_data[f"clinker_{plant_analyzed}"]

pyhub = {}

for tec_name in explored_technologies:

    if "mea" in tec_name:
        techs = ["CementEmitter"]
    elif "oxy" in tec_name:
        techs = ["CementHybridCCS"]


    techs += ["HeatPump"]

    if "inflex" not in tec_name:
        techs += ["ClinkerStorage"]

    pyhub[tec_name] = {}
    for std in explored_std:
        name_profile = f"el_price_norm_{std}"
        electricity_price_norm = generated_el_profiles[name_profile]
        pyhub_std = f"std_{std}"
        pyhub[tec_name][pyhub_std] = {}

        for av_el_price in explored_el_price:
            pyhub_el_price = f"el_price_{av_el_price}"
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

            end_period = 8760


            # Load json template
            with open(casepath / "ConfigModel.json", "r") as json_file:
                configuration = json.load(json_file)
            # Change objective
            configuration["optimization"]["objective"]["value"] = "costs"
            # Set MILP gap
            configuration["solveroptions"]["mipgap"]["value"] = 0.03
            # Dymanics on/off
            configuration["performance"]["dynamics"]["value"] = dymanics_on
            # change save options
            configuration['reporting']['save_summary_path']['value'] = result_path
            configuration['reporting']['save_path']['value'] = result_path
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
            technologies["new"] = techs

            with open(
                casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
            ) as json_file:
                json.dump(technologies, json_file, indent=4)

            # Copy over technology files
            adopt.copy_technology_data(casepath, json_files_path)

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
            print("Max size per arc:", arc_size)

            # Use the templates, fill and save them to the respective directory
            # Connection
            connection = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "connection.csv", sep=";", index_col=0)
            connection.loc["industrial_cluster", "storage"] = 1
            connection.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "connection.csv",
                              sep=";")
            print("Connection:", connection)

            # Delete the template
            os.remove(casepath / "period1" / "network_topology" / "new" / "connection.csv")

            # Distance
            distance = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "distance.csv", sep=";", index_col=0)
            distance.loc["industrial_cluster", "storage"] = distance_to_stor
            distance.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "distance.csv", sep=";")
            print("Distance:", distance)

            # Delete the template
            os.remove(casepath / "period1" / "network_topology" / "new" / "distance.csv")

            # Delete the max_size_arc template
            os.remove(casepath / "period1" / "network_topology" / "new" / "size_max_arcs.csv")


            # Import hourly profiles
            electricity_price = electricity_price_norm * av_el_price

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
                value_or_data=200,
                columns=["Import price"],
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
            pyhub[tec_name][pyhub_std][pyhub_el_price] = adopt.ModelHub()
            pyhub[tec_name][pyhub_std][pyhub_el_price].read_data(casepath, start_period=0, end_period=end_period)

            pyhub[tec_name][pyhub_std][pyhub_el_price].data.model_config['reporting']['case_name'][
                'value'] = f"{tec_name}_{pyhub_std}_{pyhub_el_price}"
            # pyhub[pyhub_el_price].data.time_series['full']['period1', 'industrial_cluster', 'CarrierData', 'heat', 'Demand'] = heat_demand

            pyhub[tec_name][pyhub_std][pyhub_el_price].construct_model()
            pyhub[tec_name][pyhub_std][pyhub_el_price].construct_balances()
            pyhub[tec_name][pyhub_std][pyhub_el_price].solve()
