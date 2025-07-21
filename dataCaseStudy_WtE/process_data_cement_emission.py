import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

import pandas as pd
import numpy as np
from pathlib import Path


def convert_emissions_to_clinker():
    path_emissions_database = Path("../dataCaseStudy_Cement/dataSources/DS.06.01_emission profiles_v2.xlsx")
    emissions_data = pd.read_excel(path_emissions_database, sheet_name=None)
    names = ["Vernasca", "Robilante", "Monselice", "Fanna"]
    clinker_df = pd.DataFrame()

    for name_plant in names:
        emissions = emissions_data["Cement-" + name_plant]
        emissions_daily = emissions["CO2 flowrate\ndaily average t_CO2/day"].copy()
        total_count = len(emissions_daily)
        value_counts = emissions_daily.value_counts()
        frequencies = value_counts / total_count
        frequent_values_emissions = frequencies[frequencies > 0.1]
        non_zero_frequent_values = frequent_values_emissions[frequent_values_emissions.index != 0]
        number_of_plant_lines = len(non_zero_frequent_values)

        for day in range(len(emissions_daily)):
            value = emissions_daily.iloc[day]
            if number_of_plant_lines == 1:
                emissions_daily.iloc[day] = frequent_values_emissions.index[0] if value > \
                                                                                  frequent_values_emissions.index[
                                                                                      0] * 0.5 else 0
            elif number_of_plant_lines == 2:
                if value > frequent_values_emissions.index[1] * 0.75:
                    emissions_daily.iloc[day] = frequent_values_emissions.index[1]
                elif value > frequent_values_emissions.index[1] * 0.25 and value < frequent_values_emissions.index[1] * 0.75:
                    emissions_daily.iloc[day] = frequent_values_emissions.index[0]
                else:
                    emissions_daily.iloc[day] = 0

        emissions_hourly = np.repeat(emissions_daily.values / 24, 24)
        clinker_hourly = emissions_hourly / 0.833
        clinker_df[f"clinker_{name_plant}"] = clinker_hourly

    path_processed_data = Path("../dataCaseStudy_Cement/data_processed.xlsx")
    clinker_df.to_excel(path_processed_data, index=False)


convert_emissions_to_clinker()