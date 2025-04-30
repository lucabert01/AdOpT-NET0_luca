import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def convert_emissions_to_clinker():
    """Function that converts the daily data of the emissions from selected cement plants into hourly
    data of clinker production """
    path_emissions_database = Path("../dataCaseStudy_Cement/dataSources/DS.06.01_emission profiles_v2.xlsx")
    emissions_data = pd.read_excel(path_emissions_database, sheet_name=None)
    names = ["Vernasca", "Robilante", "Monselice", "Fanna"]
    for name_plant in names:
        emissions = emissions_data["Cement-" + name_plant]
        emissions_daily = emissions["CO2 flowrate\ndaily average t_CO2/day"]
        total_count = len(emissions_daily)
        value_counts = emissions_daily.value_counts()
        frequencies = value_counts / total_count
        frequency_threshold = 0.1
        # Trying to understand the rated capacity of the plant
        frequent_values_emissions = frequencies[frequencies > frequency_threshold]
        print(frequent_values_emissions)
        non_zero_frequent_values = frequent_values_emissions[frequent_values_emissions.index != 0]
        number_of_plant_lines = len(non_zero_frequent_values)

        for day in range(0, 365):
            if number_of_plant_lines == 1:
                if emissions_daily[day]> frequent_values_emissions.index[0]*0.5:
                    emissions_daily[day] = frequent_values_emissions.index[0]
                else:
                    emissions_daily[day] = 0
            elif number_of_plant_lines == 2:
                if emissions_daily[day]> frequent_values_emissions.index[0]*0.75:
                    emissions_daily[day] = frequent_values_emissions.index[0]
                elif emissions_daily[day]< frequent_values_emissions.index[0]*0.75 and emissions_daily[day]> frequent_values_emissions.index[1]*0.25:
                    emissions_daily[day] = frequent_values_emissions.index[1]
                else:
                    emissions_daily[day] = 0
        # TODO transform from daily to hourly + add last day
        # TODO convert to clinker demand
        # TODO export to file excel data_processed

convert_emissions_to_clinker()