# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
from adopt_net0.data_preprocessing import load_climate_data_from_api


# Specify the path to your input data
casepath= Path("CaseStudy_WtE_DHratio_minEmission")
json_files_path = Path("./dataCaseStudy_WtE/technologies_json")
result_path = "./dataCaseStudy_WtE/raw_results"

adopt.create_optimization_templates(casepath)


# General input data
possible_plants = ["silla2", "gerbido", "PAIP", "piacenza"]
plant_analyzed = possible_plants[0]
carbon_tax = 200
max_dh_ratio = [0, 0.25, 0.5, 0.75, 1] # ratio of peak DH demand to supply compared to peak heat prod. from WtE
pyhub = {}

for dh_ratio in max_dh_ratio:
    pyhub_dh_ratio = f"dh_ratio_{dh_ratio}"
    # Load json template
    with open(casepath / "Topology.json", "r") as json_file:
        topology = json.load(json_file)
    # Nodes
    topology["nodes"] = ["industrial_cluster"]
    # Carriers:
    topology["carriers"] = [
        "electricity",
        "CO2captured",
        "heat",
        "wasteFuel",
        "wasteProcessed",
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
    configuration["optimization"]["objective"]["value"] = "emissions_net"
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
    node_location.at["industrial_cluster", "lon"] = 10
    node_location.at["industrial_cluster", "lat"] = 10
    node_location.at["industrial_cluster", "alt"] = 10
    node_location = node_location.reset_index()
    node_location.to_csv(casepath / "NodeLocations.csv", sep=";", index=False)
    
    with open(
        casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "r"
    ) as json_file:
        technologies = json.load(json_file)
    technologies["new"] = ["WasteCHP", "Boiler_Industrial_NG"]
    
    with open(
        casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
    ) as json_file:
        json.dump(technologies, json_file, indent=4)
    
    # Copy over technology files
    adopt.copy_technology_data(casepath, json_files_path)
    
    
    # Import hourly profiles
    path_processed_data = Path("./dataCaseStudy_WtE/dataSources/hourly_data_casestudy.xlsx")
    data = pd.read_excel(path_processed_data)
    electricity_price = data["el_price_itNord"]
    emissions = data[f"emission_{plant_analyzed}"]
    norm_heat_demand = data["normalized_heat_demand_milan"]
    json_wasteCHP = Path("./dataCaseStudy_WtE/technologies_json/WasteCHP.json")
    info_wasteCHP = json.loads(json_wasteCHP.read_text())
    lhv = info_wasteCHP["Performance"]["LHV"]
    emission_factor = info_wasteCHP["Performance"]["emission_factor"]
    wasteProcessed_demand = emissions/emission_factor
    max_heat_output = max(wasteProcessed_demand)*lhv
    peak_heat_demand = dh_ratio*max_heat_output
    heat_demand = norm_heat_demand * peak_heat_demand
    
    # Set import limits/cost
    adopt.fill_carrier_data(
        casepath,
        value_or_data=5000,
        columns=["Export limit"],
        carriers=["electricity"],
        nodes=["industrial_cluster"],
    )
    adopt.fill_carrier_data(
        casepath,
        value_or_data=5000,
        columns=["Export limit"],
        carriers=["CO2captured"],
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
        carriers=["wasteFuel"],
        nodes=["industrial_cluster"],
    )
    adopt.fill_carrier_data(
        casepath,
        value_or_data=1000,
        columns=["Import limit"],
        carriers=["gas"],
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
    
    
    carbon_price = np.ones(8760) * carbon_tax
    carbon_cost_path = (
        casepath / "period1" / "node_data" / "industrial_cluster" / "CarbonCost.csv"
    )
    carbon_cost_template = pd.read_csv(carbon_cost_path, sep=";", index_col=0, header=0)
    carbon_cost_template["price"] = carbon_price
    carbon_cost_template = carbon_cost_template.reset_index()
    carbon_cost_template.to_csv(carbon_cost_path, sep=";", index=False)
    
    load_climate_data_from_api(folder_path=casepath)
    
    # Construct and solve the model
    pyhub[pyhub_dh_ratio] = adopt.ModelHub()
    pyhub[pyhub_dh_ratio].read_data(casepath, start_period=0, end_period=end_period)

    pyhub[pyhub_dh_ratio].data.model_config['reporting']['case_name'][
        'value'] = pyhub_dh_ratio
    pyhub[pyhub_dh_ratio].data.time_series['full']['period1', 'industrial_cluster', 'CarrierData', 'heat', 'Demand'] = heat_demand
    
    
    pyhub[pyhub_dh_ratio].construct_model()
    pyhub[pyhub_dh_ratio].construct_balances()
    pyhub[pyhub_dh_ratio].solve()
