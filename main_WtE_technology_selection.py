# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
import os

# Specify the path to your input data
casepath= Path("CaseStudies_WtE/technology_selection")
json_files_path = Path("./dataCaseStudy_WtE/technologies_json")
json_files_path_network = Path("./dataCaseStudy_WtE/network_json")
result_path = "./dataCaseStudy_WtE/raw_results/technology_selection"

adopt.create_optimization_templates(casepath)

# Import data from the json file
json_wasteCHP = Path("./dataCaseStudy_WtE/technologies_json/WasteCHP.json")
info_wasteCHP = json.loads(json_wasteCHP.read_text())
lhv = info_wasteCHP["Performance"]["LHV"]
th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
el_efficiency = info_wasteCHP["Performance"]["el_efficiency"]
emission_factor = info_wasteCHP["Performance"]["emission_factor"]
path_processed_data = Path("./dataCaseStudy_WtE/dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)

# General input data
objective_function = "costs" # "emissions_net", "emissions_minC", "costs"
explored_carbon_tax = [50, 75, 100,125, 150]
explored_el_price = [25, 50, 75, 100,125, 150, 175] # average el prices explored in the analysis
explored_dh_ratio = [0.5, 0.75]
plant_analyzed = "PAIP" # one between: "silla2", "gerbido", "PAIP", "piacenza"
gas_price = 40
import_price_RDF = 20
existing_boiler_size = max(data[f"emission_{plant_analyzed}"])/emission_factor*lhv*th_efficiency
wte_demand_is_averaged = 0
heat_demand_is_averaged = 0
rolling_av_hours = 24*7
co2_concentration = data["co2_concentration_"+plant_analyzed]
distance_to_stor = 100
pyhub = {}

path_processed_data = Path("dataCaseStudy_Cement/dataSources/data_processed.xlsx")
electricity_price_data = pd.read_excel(path_processed_data, sheet_name="electricity_prices")
av_el_price = electricity_price_data["el_price_itNord"].mean()
electricity_price_norm = electricity_price_data["el_price_itNord"]/av_el_price



for dh_ratio in explored_dh_ratio:
    pyhub_dh_ratio = f"dh_{dh_ratio}"
    pyhub[pyhub_dh_ratio] = {}

    for carbon_tax in explored_carbon_tax:
        pyhub_carbon_tax = f"ctax_{carbon_tax}"
        pyhub[pyhub_dh_ratio][pyhub_carbon_tax] = {}

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

            end_period = 8760


            # Load json template
            with open(casepath / "ConfigModel.json", "r") as json_file:
                configuration = json.load(json_file)
            # Change objective
            configuration["optimization"]["objective"]["value"] = "costs"
            # Set MILP gap
            configuration["solveroptions"]["mipgap"]["value"] = 0.02
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
            technologies["new"] = ["WasteCHP", "WasteCaL_CCS"]  # ,"WasteCHP"
            technologies["existing"] = {"Boiler_Industrial_NG": existing_boiler_size}

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
            arc_size.to_csv(
                casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "size_max_arcs.csv",
                sep=";")
            print("Max size per arc:", arc_size)

            # Use the templates, fill and save them to the respective directory
            # Connection
            connection = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "connection.csv", sep=";",
                                     index_col=0)
            connection.loc["industrial_cluster", "storage"] = 1
            connection.to_csv(
                casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "connection.csv",
                sep=";")
            print("Connection:", connection)

            # Delete the template
            os.remove(casepath / "period1" / "network_topology" / "new" / "connection.csv")

            # Distance
            distance = pd.read_csv(casepath / "period1" / "network_topology" / "new" / "distance.csv", sep=";",
                                   index_col=0)
            distance.loc["industrial_cluster", "storage"] = distance_to_stor
            distance.to_csv(casepath / "period1" / "network_topology" / "new" / "CO2PipelineOnshore" / "distance.csv",
                            sep=";")
            print("Distance:", distance)

            # Delete the template
            os.remove(casepath / "period1" / "network_topology" / "new" / "distance.csv")

            # Delete the max_size_arc template
            os.remove(casepath / "period1" / "network_topology" / "new" / "size_max_arcs.csv")

            # Import hourly profiles
            electricity_price = electricity_price_norm * av_el_price
            if wte_demand_is_averaged:
                emissions = data[f"emission_{plant_analyzed}"].rolling(window=rolling_av_hours, min_periods=1).mean()
            else:
                emissions = data[f"emission_{plant_analyzed}"]
            norm_heat_demand = data["normalized_heat_demand_milan"]
            json_wasteCHP = Path("./dataCaseStudy_WtE/technologies_json/WasteCHP.json")
            info_wasteCHP = json.loads(json_wasteCHP.read_text())
            lhv = info_wasteCHP["Performance"]["LHV"]
            th_efficiency = info_wasteCHP["Performance"]["th_efficiency"]
            emission_factor = info_wasteCHP["Performance"]["emission_factor"]
            wasteProcessed_demand = emissions / emission_factor
            max_useful_heat_output = max(wasteProcessed_demand) * lhv * th_efficiency
            peak_heat_demand = dh_ratio * max_useful_heat_output
            heat_demand = (norm_heat_demand * peak_heat_demand).rolling(window=rolling_av_hours, min_periods=1).mean()
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
                value_or_data=30,
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
            pyhub[pyhub_dh_ratio][pyhub_carbon_tax][pyhub_el_price] = adopt.ModelHub()
            pyhub[pyhub_dh_ratio][pyhub_carbon_tax][pyhub_el_price].read_data(casepath, start_period=0, end_period=end_period)

            pyhub[pyhub_dh_ratio][pyhub_carbon_tax][pyhub_el_price].data.model_config['reporting']['case_name'][
                'value'] = f"{pyhub_dh_ratio}_{pyhub_carbon_tax}_{pyhub_el_price}"
            # pyhub[pyhub_el_price].data.time_series['full']['period1', 'industrial_cluster', 'CarrierData', 'heat', 'Demand'] = heat_demand

            pyhub[pyhub_dh_ratio][pyhub_carbon_tax][pyhub_el_price].construct_model()
            pyhub[pyhub_dh_ratio][pyhub_carbon_tax][pyhub_el_price].construct_balances()
            pyhub[pyhub_dh_ratio][pyhub_carbon_tax][pyhub_el_price].solve()
