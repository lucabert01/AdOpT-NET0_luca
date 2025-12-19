# from adopt_net0.model_configuration import ModelConfiguration
import adopt_net0 as adopt
import json
import pandas as pd
from pathlib import Path
import numpy as np
from adopt_net0.data_preprocessing import load_climate_data_from_api

# Specify the path to your input data
casepath= Path("CaseStudies_WtE/CaL")
json_files_path = Path("./dataCaseStudy_WtE/technologies_json")
result_path = "./dataCaseStudy_WtE/raw_results/CaL"

adopt.create_optimization_templates(casepath)


# General input data
objective_function = "costs" # "emissions_net", "emissions_minC", "costs"
plant_analyzed = "PAIP" # one between: "silla2", "gerbido", "PAIP", "piacenza"
carbon_tax = 200
gas_price = 40
import_price_RDF = 20
explored_el_price = [25, 50, 75, 100,125] # average el prices explored in the analysis
existing_boiler_size = 250
dh_ratio = 0.5
wte_demand_is_averaged = 0
heat_demand_is_averaged = 0
rolling_av_hours = 24*7
path_processed_data = Path("./dataCaseStudy_WtE/dataSources/hourly_data_casestudy.xlsx")
data = pd.read_excel(path_processed_data)
av_el_price = data["el_price_itNord"].mean()
electricity_price_norm = data["el_price_itNord"]/av_el_price
co2_concentration = data["co2_concentration_"+plant_analyzed]
pyhub = {}

for av_el_price in explored_el_price:
    pyhub_el_price = f"el_price_{av_el_price}"
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
        "wasteIn",
        "wasteInRDF",
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
    configuration["optimization"]["objective"]["value"] = objective_function
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
    technologies["new"] = [ "WasteCaL_CCS"]
    technologies["existing"] = {"Boiler_Industrial_NG": existing_boiler_size}

    with open(
        casepath / "period1" / "node_data" / "industrial_cluster" / "Technologies.json", "w"
    ) as json_file:
        json.dump(technologies, json_file, indent=4)
    
    # Copy over technology files
    adopt.copy_technology_data(casepath, json_files_path)
    
    
    # Import hourly profiles
    electricity_price = electricity_price_norm*av_el_price
    if wte_demand_is_averaged:
        emissions = data[f"emission_{plant_analyzed}"].rolling(window=rolling_av_hours, min_periods=1).mean()
    else:
        emissions = data[f"emission_{plant_analyzed}"]
    # emissions  = emissions.to_numpy()
    norm_heat_demand = data["normalized_heat_demand_milan"]
    json_WasteCaL_CCS = Path("./dataCaseStudy_WtE/technologies_json/WasteCaL_CCS.json")
    info_WasteCaL_CCS = json.loads(json_WasteCaL_CCS.read_text())
    lhv = info_WasteCaL_CCS["Performance"]["LHV"]
    th_efficiency = info_WasteCaL_CCS["Performance"]["th_efficiency"]
    emission_factor = info_WasteCaL_CCS["Performance"]["emission_factor"]
    wasteProcessed_demand = emissions/emission_factor
    max_useful_heat_output = max(wasteProcessed_demand)*lhv*th_efficiency
    peak_heat_demand = dh_ratio*max_useful_heat_output
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
        value_or_data=500,
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

    tech_with_hourly_co2_concentration = ["WasteCaL_CCS"]
    climate_data_file = (
            casepath / "period1" / "node_data" / "industrial_cluster" / "ClimateData.csv"
    )
    climate_data = pd.read_csv(climate_data_file)
    for tech in tech_with_hourly_co2_concentration:
        climate_data["co2_concentration_" + tech] = co2_concentration.values
    climate_data.to_csv(climate_data_file, index=False, sep=";")
    
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
    pyhub[pyhub_el_price] = adopt.ModelHub()
    pyhub[pyhub_el_price].read_data(casepath, start_period=0, end_period=end_period)

    pyhub[pyhub_el_price].data.model_config['reporting']['case_name'][
        'value'] = f"{pyhub_el_price}_{objective_function}"
    # pyhub[pyhub_el_price].data.time_series['full']['period1', 'industrial_cluster', 'CarrierData', 'heat', 'Demand'] = heat_demand
    
    
    pyhub[pyhub_el_price].construct_model()
    pyhub[pyhub_el_price].construct_balances()
    pyhub[pyhub_el_price].solve()
    a=1

