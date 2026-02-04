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

    path_processed_data = Path("./dataSources/data_processed.xlsx")
    clinker_df.to_excel(path_processed_data, index=False)


# convert_emissions_to_clinker()

def create_norm_el_price_profiles(el_price_base, std_factors):
    # Anchor to the project folder for robustness
    BASE_DIR = Path(__file__).resolve().parents[1]  # one level up from this script
    path_processed_data = BASE_DIR / "dataCaseStudy_Cement" / "dataSources" / "data_processed.xlsx"
    sheet_name = "el_prices_norm_stds"

    av_el_price = el_price_base.mean()
    profiles = {}

    for std_el in std_factors:
        name_profile = f"el_price_norm_{std_el}"
        el_price_new = av_el_price + std_el * (el_price_base - av_el_price)
        norm_el_price_new = el_price_new / el_price_new.mean()
        profiles[name_profile] = norm_el_price_new

    df_profiles = pd.DataFrame(profiles)

    if path_processed_data.exists():
        # Append mode → sheet replacement allowed
        with pd.ExcelWriter(
                path_processed_data,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace"
        ) as writer:
            df_profiles.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Write mode → no if_sheet_exists
        with pd.ExcelWriter(
                path_processed_data,
                engine="openpyxl",
                mode="w"
        ) as writer:
            df_profiles.to_excel(writer, sheet_name=sheet_name, index=False)



